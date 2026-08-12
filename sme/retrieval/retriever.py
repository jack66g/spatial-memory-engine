"""Two-stage retrieval.

Stage 1 - Region Retrieval: rank spatial regions by query relevance.
Stage 2 - Memory Retrieval: hybrid search INSIDE the top regions.

The hybrid memory search combines:
    - vector similarity  (embedding cosine)
    - keyword similarity (BM25-style, IDF weighted token overlap)
    - metadata filter    (exact match on the query's metadata filters)
    - region contribution (memories in better regions get a boost)

The final ranking then applies the MemoryRanker (semantic / importance /
freshness / weight / decay / hits / recency / region).

Hybrid scoring inside a region:
    hybrid = vector_weight * vector_score
           + keyword_weight * keyword_score
           + metadata_weight * metadata_match
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sme.config import RetrievalConfig
from sme.models import RegionHit, SearchHit
from sme.ranking import MemoryRanker
from sme.utils import now, tokenize


@dataclass
class SearchQuery:
    text: str
    top_k: int = 10
    top_regions: Optional[int] = None
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    tags: Optional[list[str]] = None
    include_archived: bool = False
    region_retrieval: Optional[str] = None  # top1 | top3 | top5
    graph_expand: int = 0  # walk the memory graph for related memories (0=off)


class BM25Index:
    """Incremental BM25-style keyword index over memory texts.

    Document-frequency statistics are maintained incrementally on
    add/remove, so query-time scoring is O(candidates x query tokens)
    with no full rebuild (unlike the previous set_documents design).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 cjk_bigram: bool = True) -> None:
        self.k1 = k1
        self.b = b
        self.cjk_bigram = cjk_bigram  # CJK 1-2 gram tokenization (iteration 1.2)
        self._doc_tokens: dict[str, list[str]] = {}
        self._df: dict[str, int] = {}
        self._postings: dict[str, set[str]] = {}
        self._sum_len = 0
        self._n_docs = 0

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def _tok(self, text: str) -> list[str]:
        return tokenize(text, cjk_bigram=self.cjk_bigram)

    def set_documents(self, documents: dict[str, str]) -> None:
        """Replace the whole index (used after deserialization)."""
        self._doc_tokens = {mid: self._tok(text) for mid, text in documents.items()}
        self._df = {}
        self._postings = {}
        self._sum_len = 0
        self._n_docs = 0
        for mid, toks in self._doc_tokens.items():
            self._n_docs += 1
            self._sum_len += len(toks)
            for tok in set(toks):
                self._df[tok] = self._df.get(tok, 0) + 1
                self._postings.setdefault(tok, set()).add(mid)

    def add_document(self, memory_id: str, text: str) -> None:
        toks = self._tok(text)
        old = self._doc_tokens.get(memory_id)
        if old is not None:
            for tok in set(old):
                self._df[tok] -= 1
                if self._df[tok] <= 0:
                    self._df.pop(tok, None)
                posting = self._postings.get(tok)
                if posting is not None:
                    posting.discard(memory_id)
                    if not posting:
                        self._postings.pop(tok, None)
            self._sum_len -= len(old)
        else:
            self._n_docs += 1
        self._doc_tokens[memory_id] = toks
        self._sum_len += len(toks)
        for tok in set(toks):
            self._df[tok] = self._df.get(tok, 0) + 1
            self._postings.setdefault(tok, set()).add(memory_id)

    def remove_document(self, memory_id: str) -> None:
        old = self._doc_tokens.pop(memory_id, None)
        if old is None:
            return
        self._n_docs -= 1
        self._sum_len -= len(old)
        for tok in set(old):
            self._df[tok] -= 1
            if self._df[tok] <= 0:
                self._df.pop(tok, None)
            posting = self._postings.get(tok)
            if posting is not None:
                posting.discard(memory_id)
                if not posting:
                    self._postings.pop(tok, None)

    @property
    def _avg_len(self) -> float:
        return self._sum_len / max(1, self._n_docs)

    def scores(self, query_tokens: list[str], memory_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {mid: 0.0 for mid in memory_ids}
        if not query_tokens or not memory_ids:
            return out
        # inverted-index pruning: only documents containing at least one
        # query token can have a non-zero score
        candidate_set: set[str] | None = None
        for qt in query_tokens:
            post = self._postings.get(qt)
            if not post:
                continue
            if candidate_set is None:
                candidate_set = set(post)
            else:
                candidate_set.update(post)
        if not candidate_set:
            return out
        requested = set(memory_ids)
        to_score = requested & candidate_set
        if not to_score:
            return out

        avg_len = self._avg_len
        n_docs = self._n_docs
        df = self._df
        # hoist the per-token idf (independent of the candidate document)
        idf: dict[str, float] = {}
        for qt in query_tokens:
            idf[qt] = math.log(
                1 + (n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5)
            )
        for mid in to_score:
            toks = self._doc_tokens.get(mid, [])
            doc_len = len(toks)
            freq: dict[str, int] = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            score = 0.0
            for qt in query_tokens:
                f = freq.get(qt, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / max(avg_len, 1))
                score += idf[qt] * (f * (self.k1 + 1)) / max(denom, 1e-9)
            out[mid] = score
        if max(out.values()) > 0:
            peak = max(out.values())
            out = {k: v / peak for k, v in out.items()}
        return out


class TwoStageRetriever:
    def __init__(self, config: RetrievalConfig, ranker: MemoryRanker) -> None:
        self.config = config
        self.ranker = ranker
        self.bm25 = BM25Index(cjk_bigram=config.cjk_bigram)

    # ------------------------------------------------------------------ #
    def refresh_keywords(self, engine: object) -> None:
        # keep the tokenizer in sync with the (possibly re-bound) config
        self.bm25.cjk_bigram = self.config.cjk_bigram
        texts = {
            mid: m.text
            for mid, m in engine.memories.items()
            if not m.archived
        }
        self.bm25.set_documents(texts)

    def index_memory(self, memory: object) -> None:
        """Incrementally add/update a memory in the keyword index."""
        if memory.archived:
            self.bm25.remove_document(memory.id)
        else:
            self.bm25.add_document(memory.id, memory.text)

    def drop_memory(self, memory_id: str) -> None:
        self.bm25.remove_document(memory_id)

    def warm_keywords(self, engine: object) -> None:
        """Build the global keyword index once (after load / import)."""
        if self.bm25.n_docs > 0 or not engine.memories:
            return
        self.refresh_keywords(engine)

    # ------------------------------------------------------------------ #
    def search(self, engine: object, query: SearchQuery) -> list[SearchHit]:
        """Two-stage search: regions first, then memories inside them."""
        ref = now()
        self.warm_keywords(engine)
        query_vec = engine.embeddings.embed_one(query.text)

        # stage 1: region retrieval
        # explicit query override -> region_retrieval string -> config default
        k_regions = query.top_regions
        if k_regions is None and query.region_retrieval:
            try:
                k_regions = int(query.region_retrieval.replace("top", ""))
            except ValueError:
                k_regions = None
        if k_regions is None:
            k_regions = self.config.top_regions
        region_hits = engine.space.query_regions(query_vec, k_regions)

        # stage 2: hybrid memory retrieval inside candidate regions
        candidate_ids: set[str] = set()
        region_scores: dict[str, float] = {
            hit.region.id: hit.score for hit in region_hits
        }
        for rhit in region_hits:
            candidate_ids.update(engine.space.candidates_in_region(rhit.region.id))

        # budgeted global supplement: always add the best globally-similar
        # memories so a strong memory inside a mediocre region is never
        # missed by the region gate. The pool target is max(top_k*2,
        # candidate_window) (iteration 2.2): template sentences crowding the
        # top-k no longer cut off a real memory that ranks just outside.
        pool_target = max(query.top_k * 2, self.config.candidate_window)
        if len(candidate_ids) < pool_target:
            budget = pool_target - len(candidate_ids)
            pool = [
                (mid, m)
                for mid, m in engine.memories.items()
                if mid not in candidate_ids
                and (not m.archived or query.include_archived)
                and (engine.policy.allows_retrieval(m) or query.include_archived)
                and m.embedding is not None
            ]
            if pool:
                pool_ids = [mid for mid, _ in pool]
                q = query_vec.reshape(1, -1)
                # chunked matmul: identical results, bounded peak memory even
                # when the global supplement has to scan a very large store
                sims = np.empty(len(pool), dtype=np.float64)
                step = 4096
                for start in range(0, len(pool), step):
                    part = pool[start : start + step]
                    mat = np.stack([m.embedding for _, m in part])
                    mat = mat / np.clip(
                        np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None
                    )
                    sims[start : start + len(part)] = (mat @ q.T).reshape(-1)
                order = np.argsort(sims)[::-1][:budget]
                for idx in order:
                    mid = pool_ids[int(idx)]
                    candidate_ids.add(mid)
                    rid = engine.space.region_for(mid)
                    # these memories come from unranked regions, so they get
                    # the flat dampening boost (keyed by their region id)
                    region_scores.setdefault(rid or "", self.config.region_dampening)

        if not candidate_ids:
            return []

        candidates = [engine.memories[mid] for mid in candidate_ids]
        candidates = [m for m in candidates if not m.archived or query.include_archived]
        # apply the global memory policy (full_memory / importance filter);
        # archived memories may bypass it when explicitly requested
        candidates = [
            m
            for m in candidates
            if engine.policy.allows_retrieval(m) or query.include_archived
        ]

        # metadata + tag filters
        candidates = [
            m
            for m in candidates
            if self._metadata_matches(m, query.metadata_filters)
            and self._tags_match(m, query.tags)
        ]

        # hybrid scores
        vector_scores = self._vector_scores(query_vec, candidates)
        keyword_scores = self._keyword_scores(query.text, candidates)
        metadata_matches = {
            m.id: self._metadata_matches(m, query.metadata_filters) for m in candidates
        }

        scored: list[SearchHit] = []
        for memory in candidates:
            vector_score = vector_scores.get(memory.id, 0.0)
            keyword_score = keyword_scores.get(memory.id, 0.0)
            region_id = engine.space.region_for(memory.id) or ""
            region_score = region_scores.get(region_id, 0.0)
            meta_score = 1.0 if metadata_matches.get(memory.id) else 0.0

            hybrid = (
                self.config.vector_weight * vector_score
                + self.config.keyword_weight * keyword_score
                + self.config.metadata_weight * meta_score
            )
            final, _ = self.ranker.score(
                memory,
                query_vec,
                region_score,
                engine,
                reference=ref,
                semantic=hybrid,
                detailed=False,
            )
            if memory.source == "summary":
                # summary memories are navigation aids: keep them retrievable
                # but let real facts outrank their generic template text
                final *= self.config.summary_penalty
            scored.append(
                SearchHit(
                    memory=memory,
                    score=final,
                    region_id=region_id,
                    region_score=region_score,
                    keyword_score=round(keyword_score, 4),
                    vector_score=round(vector_score, 4),
                    metadata_match=metadata_matches.get(memory.id, True),
                )
            )

        scored.sort(key=lambda hit: hit.score, reverse=True)
        top = scored[: query.top_k]
        # graph expansion: pull in memories that are graph-related to the
        # hits even when their embeddings are far from the query (P0)
        if query.graph_expand > 0 and engine.graph.edges:
            top = self._graph_expand(
                engine, query, top, scored, query_vec, region_scores, ref
            )
        # compute the transparent breakdown only for the final top-k
        for hit in top:
            _, breakdown = self.ranker.score(
                hit.memory,
                query_vec,
                hit.region_score,
                engine,
                reference=ref,
                semantic=(
                    self.config.vector_weight * hit.vector_score
                    + self.config.keyword_weight * hit.keyword_score
                    + self.config.metadata_weight * (1.0 if hit.metadata_match else 0.0)
                ),
                detailed=True,
            )
            if hit.memory.source == "summary":
                # keep breakdown.final consistent with the scored (penalized) score
                breakdown.final *= self.config.summary_penalty
            hit.breakdown = breakdown
        return top

    # ------------------------------------------------------------------ #
    def _graph_expand(
        self,
        engine: object,
        query: SearchQuery,
        top: list[SearchHit],
        scored: list[SearchHit],
        query_vec: np.ndarray,
        region_scores: dict[str, float],
        ref: float,
    ) -> list[SearchHit]:
        """BFS the memory graph from the current hits (depth-gated).

        Expanded memories are scored with the same hybrid pipeline but their
        final score is decayed by 0.5**depth so related-but-far memories
        surface without drowning the vector-similar ones. The expansion is
        capped so a dense neighbor graph cannot explode the candidate pool.
        """
        if query.graph_expand <= 0:
            return top
        # only the final top-k counts as "already surfaced"; a memory that
        # was scored but lost the ranking may still be rescued by the graph
        included: set[str] = {h.memory.id for h in top}
        expanded: list[tuple[str, int]] = []
        seen: set[str] = set(included)
        frontier = [h.memory.id for h in top]
        cap = max(query.top_k * 6, 30)
        for depth in range(1, query.graph_expand + 1):
            if len(expanded) >= cap:
                break
            nxt: list[str] = []
            for mid in frontier:
                for nb in engine.graph.neighbors_of(mid):
                    if nb in seen or nb not in engine.memories:
                        continue
                    nb_memory = engine.memories[nb]
                    if nb_memory.archived and not query.include_archived:
                        # archived nodes neither surface nor relay the chain
                        seen.add(nb)
                        continue
                    seen.add(nb)
                    expanded.append((nb, depth))
                    nxt.append(nb)
                    if len(expanded) >= cap:
                        break
                if len(expanded) >= cap:
                    break
            frontier = nxt
            if not frontier:
                break

        depth_by_id = dict(expanded)
        new_mems = [engine.memories[mid] for mid in depth_by_id]
        new_mems = [
            m
            for m in new_mems
            if (not m.archived or query.include_archived)
            and (engine.policy.allows_retrieval(m) or query.include_archived)
            and self._metadata_matches(m, query.metadata_filters)
            and self._tags_match(m, query.tags)
        ]
        if not new_mems:
            return top

        vector_scores = self._vector_scores(query_vec, new_mems)
        keyword_scores = self._keyword_scores(query.text, new_mems)
        hits: list[SearchHit] = []
        for memory in new_mems:
            depth = depth_by_id[memory.id]
            rid = engine.space.region_for(memory.id) or ""
            region_score = region_scores.get(rid, self.config.region_dampening)
            vector_score = vector_scores.get(memory.id, 0.0)
            keyword_score = keyword_scores.get(memory.id, 0.0)
            meta_score = 1.0 if self._metadata_matches(memory, query.metadata_filters) else 0.0
            hybrid = (
                self.config.vector_weight * vector_score
                + self.config.keyword_weight * keyword_score
                + self.config.metadata_weight * meta_score
            )
            final, _ = self.ranker.score(
                memory,
                query_vec,
                region_score,
                engine,
                reference=ref,
                semantic=hybrid,
                detailed=False,
            )
            final *= 0.5 ** depth  # depth decay: hops lose relevance
            if memory.source == "summary":
                # summary penalty applies to graph-recovered nodes too
                final *= self.config.summary_penalty
            hits.append(
                SearchHit(
                    memory=memory,
                    score=final,
                    region_id=rid,
                    region_score=region_score,
                    keyword_score=round(keyword_score, 4),
                    vector_score=round(vector_score, 4),
                    metadata_match=meta_score > 0,
                )
            )
        if not hits:
            return scored[: query.top_k]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        # reserve a few top-k slots for expanded memories: pure score-based
        # merging would starve graph-recovered nodes (their decayed scores
        # are far below vector hits), so at least `reserved` slots always
        # surface related-but-far memories
        reserved = max(1, query.top_k // 5)
        merged = scored[: max(0, query.top_k - reserved)] + hits[:reserved]
        merged.sort(key=lambda hit: hit.score, reverse=True)
        return merged[: query.top_k]

    # ------------------------------------------------------------------ #
    def search_regions(self, engine: object, text: str, top_k: int) -> list[RegionHit]:
        """Region-only retrieval (stage 1 standalone)."""
        query_vec = engine.embeddings.embed_one(text)
        return engine.space.query_regions(query_vec, top_k)

    # ------------------------------------------------------------------ #
    def _vector_scores(
        self, query_vec: np.ndarray, candidates: list
    ) -> dict[str, float]:
        """Batch cosine similarity via one matmul instead of per-item calls."""
        with_emb = [(m.id, m.embedding) for m in candidates if m.embedding is not None]
        out: dict[str, float] = {m.id: 0.0 for m in candidates}
        if not with_emb:
            return out
        ids = [mid for mid, _ in with_emb]
        mat = np.stack([emb for _, emb in with_emb])
        q = query_vec.reshape(1, -1)
        # normalize defensively, then one matmul == cosine similarities
        mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
        q = q / np.clip(np.linalg.norm(q), 1e-12, None)
        sims = (mat @ q.T).reshape(-1)
        sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
        for mid, sim in zip(ids, sims):
            out[mid] = float(sim)
        return out

    def _keyword_scores(self, query_text: str, candidates: list) -> dict[str, float]:
        # the query must be tokenized with the SAME scheme as the index
        # (cjk_bigram): otherwise document-side CJK bigrams can never match
        tokens = tokenize(query_text, cjk_bigram=self.bm25.cjk_bigram)
        if not tokens:
            return {m.id: 0.0 for m in candidates}
        ids = [m.id for m in candidates]
        return self.bm25.scores(tokens, ids)

    @staticmethod
    def _metadata_matches(memory, filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in memory.metadata:
                return False
            if isinstance(value, (list, set, tuple)):
                if memory.metadata[key] not in value:
                    return False
            elif memory.metadata[key] != value:
                return False
        return True

    @staticmethod
    def _tags_match(memory, tags: Optional[list[str]]) -> bool:
        if not tags:
            return True
        return any(tag in memory.tags for tag in tags)
