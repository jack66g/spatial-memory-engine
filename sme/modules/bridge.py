"""v2 integration: write-pipeline stages and search hooks.

The engine owns one ``V2Bridge`` object which routes writes through the
pluggable ``WritePipeline`` (pipeline.py) and wraps searches with the v2
stages (qapair replay / factgraph multi-hop / noise / profile boost /
factversion penalty). Every stage is driven by its module's ``enabled``
flag - all disabled => pure v1 behavior.

Stage order (v2 模块设计 5.1)::

    extraction -> canonical -> storage -> answer_capture
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from sme.models import Fact, SearchHit
from sme.modules.factversion import ADD_BATCH_KEY
from sme.modules.namespaces import NS_KEY
from sme.modules.pipeline import WriteContext, WritePipeline
from sme.utils import cosine_similarity

# score penalty applied to bare question memories outside qapair replay
QUESTION_RANK_PENALTY = 0.05

# how many extra candidates the retriever fetches when a post-rerank module
# (noise / factversion / qapair question-penalty) needs a real pool to work
# with. v1 truncated to top_k *before* search_post, so the stale penalty and
# noise re-ranking could only shuffle a pre-cut list - the buried complete
# fact never had a chance (battle misled attribution, iteration 2.6).
# Floor 40: hybrid (vector+keyword+region) ordering buries pure-vector detail
# facts (measured: the rent/commute fact at cosine 0.64 missing from a 20-pool
# full of 0.55-0.57 keyword hits), the post-rerank pass then surfaces them.
POOL_FACTOR = 4
POOL_MIN = 40
POOL_REGIONS = 6

# near-duplicate collapse threshold for the result list (module 06 "duplication
# crowding"): paraphrase restatements of one fact (measured 0.82-0.96 on
# Qwen3-Embedding) occupy every top-k slot and crowd out the single complete
# fact + the superseded old statement. A hit is demoted (not dropped) when it
# is this close to an already-kept hit AND carries no digit the keeper lacks
# (digit guard: "月租2300" vs "月租1900" cosine 0.95 but must both surface).
REDUNDANCY_COS = 0.80

# keep_raw sentences are the highest-fidelity record (the extracted fragments
# are rewrites of them), but their length dilutes the embedding so short
# fragments outscore them on detail questions. A mild completeness bonus
# pulls the raw into the visible top-k (battle acc attribution, iter 2.6).
RAW_COMPLETENESS_BONUS = 1.08

# fraction of the stale penalty applied to an old statement that carries
# digits no live hit has (visible history, e.g. the superseded cat-age fact)
STALE_UNIQUE_SHARE = 0.3


def _digits(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def _current_batch(engine: Any) -> Optional[str]:
    """Write-batch id for the running add() call (see V2Bridge.add)."""
    return getattr(engine._v2, "_current_batch", None) if hasattr(engine, "_v2") else None


def _is_question(text: str) -> bool:
    from sme.modules.qapair import looks_like_question

    return looks_like_question(text)


# --------------------------------------------------------------------------- #
# write stages (module 01 / 05 / 02 / 03 / 04)
# --------------------------------------------------------------------------- #
class ExtractionStage:
    """Stage 01: fact extraction + correction markers + cosine dedup."""

    name = "extraction"

    def enabled(self, engine: Any) -> bool:
        return engine.extraction.enabled

    def run(self, engine: Any, ctx: WriteContext) -> WriteContext:
        ctx.facts = engine.extraction.extract(ctx.text, assistant=ctx.assistant)
        if ctx.facts and engine.factversion.enabled:
            # 纠正语气检测必须基于【原始用户文本】——LLM 提取可能剥离
            # “其实/不对”等语气词，导致纠错句退化为普通事实
            correction = engine.factversion.has_correction_marker(ctx.text)
            for fact in ctx.facts:
                if correction or (
                    fact.kind == "fact"
                    and engine.factversion.has_correction_marker(fact.text)
                ):
                    fact.kind = "correction"
        if not ctx.facts:
            ctx.drop = True
        else:
            ctx.facts = engine.extraction.dedup(ctx.facts, engine)
            # opt-in dual write (getattr: not a declared config field, default
            # off => zero behavior drift). LLM extraction rewrites sever the
            # scene->entity context ("为了对付成都春天的雾霾" is lost from the
            # extracted fact), so paraphrased queries can no longer reach the
            # complete fact. Keeping the raw sentence restores that bridge.
            if (
                not ctx.assistant
                and getattr(engine.extraction.config, "keep_raw", False)
            ):
                ctx.extra["keep_raw"] = ctx.text
        return ctx


class CanonicalStage:
    """Stages 01/05/02: fact versioning and the "what gets stored" decision.

    Always runs: with extraction off it passes the raw text through
    (v1 behavior), with extraction on it decides per-fact what to keep.
    """

    name = "canonical"

    def enabled(self, engine: Any) -> bool:
        return True

    def run(self, engine: Any, ctx: WriteContext) -> WriteContext:
        canonical: list[Any] = []
        if not engine.extraction.enabled:
            # no extraction module: raw text flows through (module 01 off),
            # but a user question still routes to the QA store when module 02
            # is enabled (module 02 must work standalone, without module 01)
            if ctx.assistant and not engine.extraction.config.store_assistant:
                ctx.drop = True
            elif (
                not ctx.assistant
                and engine.qapair.enabled
                and _is_question(ctx.text)
            ):
                canonical.append("__question__")
            else:
                canonical.append(ctx.text)
        elif ctx.drop:
            # extraction dropped everything: a bare question may still go
            # to the QA store when module 02 is enabled
            if not ctx.assistant and engine.qapair.enabled and _is_question(ctx.text):
                canonical.append("__question__")
        else:
            for fact in ctx.facts:
                if fact.kind == "question":
                    canonical.append(fact)
                elif engine.factversion.enabled:
                    mem, is_new = engine.factversion.resolve(fact, engine)
                    if mem is not None and not is_new:
                        continue  # duplicate -> already remembered
                    canonical.append(mem if mem is not None else fact)
                else:
                    canonical.append(fact)
        ctx.extra["canonical"] = canonical
        return ctx


class StorageStage:
    """Stages 02/03/04: store canonical items + QA pair / graph / profile."""

    name = "storage"

    def enabled(self, engine: Any) -> bool:
        return True  # runs whenever there is something to store

    def run(self, engine: Any, ctx: WriteContext) -> WriteContext:
        batch = _current_batch(engine)
        for item in ctx.extra.get("canonical", []):
            if item == "__question__":
                if not engine.qapair.enabled:
                    continue
                mem = self._store_question(engine, ctx)
            elif isinstance(item, str):
                mem = engine.memory_manager.add_memory(
                    text=item, metadata=self._with_batch(ctx.metadata, batch),
                    tags=ctx.tags,
                    importance=ctx.importance, source=ctx.source,
                    link_to=ctx.link_to, link_kind=ctx.link_kind,
                    embedding=ctx.embedding,
                )
            elif isinstance(item, Fact):
                if item.kind == "question":
                    if not engine.qapair.enabled:
                        continue  # questions only stored when module 02 is on
                    mem = self._store_question(engine, ctx, fact=item)
                else:
                    # plain extracted fact -> clean memory record
                    mem = engine.memory_manager.add_memory(
                        text=item.text,
                        metadata={
                            **ctx.metadata,
                            "fact_kind": "fact",
                            "fact_subject": item.subject,
                            "confidence": item.confidence,
                            ADD_BATCH_KEY: batch,
                        },
                        tags=list(set(ctx.tags + ["fact", "extracted"])),
                        importance=0.55,
                        source=ctx.source,
                    )
            else:
                mem = item  # already stored by factversion
            ctx.primary = ctx.primary or mem
            self._after_store(engine, ctx, mem)
        self._store_raw(engine, ctx, batch)
        return ctx

    @staticmethod
    def _with_batch(metadata: dict, batch: Optional[str]) -> dict:
        if batch is None:
            return dict(metadata)
        return {**metadata, ADD_BATCH_KEY: batch}

    def _store_raw(self, engine: Any, ctx: WriteContext, batch) -> None:
        """keep_raw dual write: also store the original sentence (opt-in).

        Skipped when the raw is a near-duplicate of an existing memory (e.g.
        a one-fact round where raw ~= extracted fact, measured cosine 0.81 <
        dedup threshold is fine to keep; >= 0.92 is a true repeat).
        """
        raw = ctx.extra.get("keep_raw")
        if not raw or ctx.assistant:
            return
        try:
            vec = engine.embeddings.embed_one(raw)
            sims = engine.extraction._top_similar(engine, vec, k=5)
            if sims and sims[0] >= engine.extraction.config.dedup_threshold:
                return  # raw already remembered verbatim
        except Exception:  # noqa: BLE001 - best effort dedup guard
            pass
        engine.memory_manager.add_memory(
            text=raw,
            metadata=self._with_batch(
                {**ctx.metadata, "kind": "raw"}, batch
            ),
            tags=list(set(ctx.tags + ["raw"])),
            importance=ctx.importance,
            source=ctx.source,
        )

    def _store_question(self, engine: Any, ctx: WriteContext, fact=None) -> Any:
        text = fact.text if fact is not None else ctx.text
        mem = engine.memory_manager.add_memory(
            text=text,
            metadata={**ctx.metadata, "kind": "question"},
            tags=list(set(ctx.tags + ["question"])),
            importance=0.5,
            source=ctx.source,
        )
        if engine.qapair.enabled:
            ctx.pending_question = {
                "question": text,
                "question_memory_id": mem.id,
                "answer_memory_id": None,
                "ns": ctx.metadata.get(NS_KEY),
            }
        return mem

    def _after_store(self, engine: Any, ctx: WriteContext, mem: Any) -> None:
        """Stages 4-6: factgraph / profile for one stored memory."""
        if mem is None:
            return
        if mem.metadata.get("kind") == "question":
            if engine.profile.enabled:
                engine.profile.upsert(mem)
            return
        if engine.factgraph.enabled and mem.metadata.get("fact_kind") == "fact":
            try:
                entities, relations = engine.factgraph_extractor.extract(mem.text)
                engine.factgraph.add_fact(entities, relations, mem.id)
            except Exception:  # noqa: BLE001
                pass
        if engine.profile.enabled:
            engine.profile.upsert(mem)
        return None


class AnswerCaptureStage:
    """Stage 02: assistant answer fulfilling a pending question -> QA pair."""

    name = "answer_capture"

    def __init__(self, bridge: "V2Bridge") -> None:
        self._bridge = bridge

    def enabled(self, engine: Any) -> bool:
        return bool(engine.qapair.enabled)

    def run(self, engine: Any, ctx: WriteContext) -> WriteContext:
        bridge = self._bridge
        if ctx.pending_question is not None:
            # 跨 add 调用传递：问题在上一轮 user 消息里，答案在本轮
            bridge.pending_question = ctx.pending_question
        if ctx.assistant and bridge.pending_question:
            self._capture_answer(engine, ctx, ctx.primary)
        elif not ctx.assistant and not ctx.primary:
            # a user utterance that produced nothing (question w/o qapair)
            bridge.pending_question = None
        return ctx

    def _capture_answer(self, engine: Any, ctx: WriteContext, primary: Any) -> None:
        bridge = self._bridge
        if not engine.qapair.enabled or not bridge.pending_question:
            return
        pq = bridge.pending_question
        answer_mid = None
        if primary is not None and getattr(primary, "kind", None) != "question":
            answer_mid = primary.id
        try:
            engine.qapair.put(
                question=pq["question"],
                answer_text=ctx.text,
                question_memory_id=pq.get("question_memory_id"),
                answer_memory_id=answer_mid,
                ns=pq.get("ns"),
                engine=engine,
            )
        except Exception:  # noqa: BLE001
            pass
        bridge.pending_question = None


# --------------------------------------------------------------------------- #
class V2Bridge:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.pending_question: dict | None = None
        self._current_batch: str | None = None   # add-batch id (stale propagation)
        self._pipeline = WritePipeline()
        self._pipeline.register(ExtractionStage())
        self._pipeline.register(CanonicalStage())
        self._pipeline.register(StorageStage())
        self._pipeline.register(AnswerCaptureStage(self))

    # ------------------------------------------------------------------ #
    # write path
    # ------------------------------------------------------------------ #
    def active(self) -> bool:
        e = self.engine
        return any(
            [
                e.extraction.enabled,
                e.factversion.enabled,
                e.qapair.enabled,
                e.factgraph.enabled,
                e.profile.enabled,
                e.wal.enabled,
            ]
        )

    def add(self, text: str, metadata=None, tags=None, importance=0.5,
            source="user", link_to=None, link_kind="reference", embedding=None) -> Optional[Any]:
        """Pipeline write. Returns the primary stored memory (or None).

        Called by ``engine.add`` only when at least one v2 write-side module
        is active (``self.active()``); with everything off the engine routes
        straight to v1, so no bypass branch is needed here.
        """
        e = self.engine
        ctx = WriteContext(
            engine=e,
            text=text,
            metadata=metadata or {},
            tags=list(tags or []),
            importance=importance,
            source=source,
            link_to=link_to,
            link_kind=link_kind,
            embedding=embedding,
            assistant=(source == "assistant"),
        )
        # one batch id per add() call: factversion stale propagation marks
        # the whole old statement (sibling fragments + raw) in one shot
        self._current_batch = uuid.uuid4().hex[:12]
        try:
            self._pipeline.run(e, ctx)
        finally:
            self._current_batch = None
        return ctx.primary

    # ------------------------------------------------------------------ #
    # search path
    # ------------------------------------------------------------------ #
    def _pool_widen(self, query: Any) -> None:
        """Widen the retriever's cut so post-rerank modules get a real pool.

        v1 truncated the candidate list to top_k *before* search_post, so the
        stale penalty / noise re-ranking could only shuffle a pre-cut list.
        Only active when a post-side module is enabled; with every module off
        the query is untouched (zero drift). search_post restores top_k.

        Also queries more regions: with static region evolution (kb presets)
        the top-3 regions can already fill the whole candidate pool, so the
        global vector supplement never triggers and a detail fact living in
        the 4th region is unreachable no matter how good its cosine is
        (battle acc attribution: rent/commute fact at cosine 0.64).
        """
        e = self.engine
        post_modules = e.noise.enabled or e.factversion.enabled or e.qapair.enabled
        if post_modules and query.top_k and not getattr(query, "_v2_pooled", False):
            query._v2_pooled = True
            query._v2_orig_top_k = query.top_k
            query.top_k = max(query.top_k * POOL_FACTOR, POOL_MIN)
            if query.top_regions is None and not query.region_retrieval:
                query._v2_orig_top_regions = None
                query.top_regions = max(
                    e.retriever.config.top_regions, POOL_REGIONS
                )

    def search_pre(self, query: Any) -> list[SearchHit]:
        """Module 02: direct question/answer replay (before v1 retrieval)."""
        e = self.engine
        self._pool_widen(query)
        if not e.qapair.enabled:
            return []
        if not query.text or not query.text.strip():
            return []
        # module 12 isolation: replay only pairs of the queried namespace
        ns = (query.metadata_filters or {}).get(NS_KEY)
        pairs = e.qapair.lookup(query.text, e, ns=ns)
        if not pairs:
            return []
        return e.qapair.answer_hits(pairs, e)

    def search_post(self, query: Any, hits: list[SearchHit]) -> list[SearchHit]:
        """Modules 03/04/05/06 after the v1 retrieval pipeline."""
        e = self.engine
        orig_k = getattr(query, "_v2_orig_top_k", None) or query.top_k
        try:
            if not hits:
                return hits
            hits = list(hits)
            if e.factgraph.enabled:
                hits = self._factgraph_expand(query, hits)
            if e.noise.enabled:
                hits = e.noise.apply(hits, e)
            if e.profile.enabled:
                hits = e.profile.boost(hits)
            if e.factversion.enabled:
                penalty = e.factversion.config.stale_penalty
                # "stale but uniquely informative" guard: an old statement that
                # still carries numbers/dates none of the live hits has (e.g.
                # the cat's age lives only in the superseded sentence) gets a
                # much softer penalty instead of none - fully unpenalized old
                # values read as "新旧并存无法辨最新" (battle round-3 seed1
                # regression: rent 2300 vs 2600 both at full score), while the
                # soft penalty keeps it visible but clearly below the latest.
                live_digits: set[str] = set()
                for hit in hits:
                    if not hit.memory.metadata.get("superseded_by"):
                        live_digits |= _digits(hit.memory.text)
                for hit in hits:
                    if hit.memory.metadata.get("superseded_by"):
                        if _digits(hit.memory.text) - live_digits:
                            hit.score *= 1.0 - penalty * STALE_UNIQUE_SHARE
                        else:
                            hit.score *= 1.0 - penalty
                hits.sort(key=lambda h: h.score, reverse=True)
            if e.extraction.enabled:
                # keep_raw completeness bonus: raws only exist when the
                # opt-in dual write is on, so this branch is inert by default
                for hit in hits:
                    if hit.memory.metadata.get("kind") == "raw":
                        hit.score *= RAW_COMPLETENESS_BONUS
                hits.sort(key=lambda h: h.score, reverse=True)
            if e.qapair.enabled:
                # bare question memories only surface through direct replay
                for hit in hits:
                    if (
                        "question" in hit.memory.tags
                        and hit.memory.metadata.get("kind") != "qapair_replay"
                    ):
                        hit.score *= QUESTION_RANK_PENALTY
                hits.sort(key=lambda h: h.score, reverse=True)
            if e.noise.enabled:
                hits = self._collapse_redundant(hits)
            return hits[:orig_k]
        finally:
            # restore the caller-owned query whatever happens (pool widening
            # was an internal device of search_pre/search_post)
            if getattr(query, "_v2_pooled", False):
                query.top_k = orig_k
                if hasattr(query, "_v2_orig_top_regions"):
                    query.top_regions = query._v2_orig_top_regions
                    del query._v2_orig_top_regions
                query._v2_pooled = False
                query._v2_orig_top_k = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _collapse_redundant(hits: list[SearchHit]) -> list[SearchHit]:
        """Module 06: demote near-duplicate restatements of a kept hit.

        Paraphrase variants of one fact (write-time paraphrase dedup only
        catches cosine >= 0.92, measured variants sit at 0.82-0.90) fill
        every top-k slot and crowd out the complete fact and the superseded
        old statement. Redundant hits are demoted below the diverse tail,
        never dropped. Digit rules:

        * candidate adds no digit the keeper lacks  -> demote (pure variant)
        * candidate carries a digit superset        -> SWAP: the more
          informative member of the cluster (usually the raw sentence that
          subsumes its extracted fragment) takes the slot
        * digit sets incomparable (e.g. 月租2300 vs 月租1900) -> keep both
        """
        kept: list[SearchHit] = []
        dropped: list[SearchHit] = []
        for hit in hits:  # input is sorted by score desc
            action = "keep"
            emb = hit.memory.embedding
            if emb is not None:
                digits = _digits(hit.memory.text)
                for i, k in enumerate(kept):
                    kemb = k.memory.embedding
                    if kemb is None:
                        continue
                    cos = cosine_similarity(emb, kemb)
                    if cos < REDUNDANCY_COS:
                        continue
                    kdigits = _digits(k.memory.text)
                    if digits <= kdigits:
                        action = "drop"            # pure variant
                        break
                    if kdigits < digits:
                        # superset of the keeper: swap in the richer text
                        # (usually the raw sentence subsuming its fragment);
                        # the displaced keeper becomes the demoted one
                        kept[i] = hit
                        hit = k
                        action = "drop"
                        break
                    action = "keep"                # incomparable: keep both
            (dropped if action == "drop" else kept).append(hit)
        return kept + dropped

    # ------------------------------------------------------------------ #
    def _factgraph_expand(self, query: Any, hits: list[SearchHit]) -> list[SearchHit]:
        """Module 03: temporal multi-hop graph expansion of the hit list."""
        e = self.engine
        fg = e.factgraph
        if not query.text:
            return hits
        entities = fg.find_entities(query.text)
        if not entities:
            return hits
        mem_hops = fg.memories_for([ent.id for ent in entities])
        if not mem_hops:
            return hits
        existing = {h.memory.id for h in hits}
        # hop >= 1 only: hop-0 memories belong to entities the query already
        # names (they surface via the vector channel); the graph channel adds
        # the *deeper* hops, and invalidated relations never reach them
        new_mems = [(mid, depth) for mid, depth in mem_hops
                    if mid not in existing and depth >= 1]
        if not new_mems:
            return hits
        top_score = max(h.score for h in hits) if hits else 0.5
        decay = fg.config.hop_decay
        extra: list[SearchHit] = []
        from sme.retrieval.retriever import TwoStageRetriever

        for mid, depth in new_mems:
            mem = e.memories.get(mid)
            if mem is None or mem.archived:
                continue
            # module 12 isolation + general metadata filters: graph-recovered
            # memories must satisfy the query filters exactly like vector hits
            if not TwoStageRetriever._metadata_matches(mem, query.metadata_filters):
                continue
            extra.append(
                SearchHit(
                    memory=mem,
                    score=top_score * (decay ** depth) * 0.9,
                    region_id=mem.region_id,
                    region_score=0.0,
                    vector_score=round(top_score * (decay ** depth), 4),
                    metadata_match=True,
                )
            )
        if not extra:
            return hits
        extra.sort(key=lambda h: h.score, reverse=True)
        reserved = max(1, query.top_k // 5)
        merged = hits[: max(0, query.top_k - reserved)] + extra[:reserved]
        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[: query.top_k]
