"""Module 02 - QAPair: question/answer pairs with direct replay.

Fixes the v1 failure mode "你叫什么名字" -> the *question* was stored but
the *answer* was not, so the model re-answered wrongly. QAPair locks
question -> answer so retrieval can replay the stored answer directly.

Flow::

    write:  user question -> question memory (tag "question") + pair stored
            assistant answer -> pair.answer_text (+ answer memory when
            store_assistant=True, linked via an "answer" graph edge)
    read:   engine.search(q) -> qapair.lookup(q) -> direct answer replay

Disabled => no-op; questions are simply dropped by extraction (v1+module 01).
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np

from sme.config import QAPairConfig
from sme.models import QAPair, SearchHit, ScoreBreakdown
from sme.utils import now

QUESTION_TAG = "question"


def looks_like_question(text: str) -> bool:
    """Cheap local question check (kept separate from extraction module)."""
    from sme.extraction import is_question

    return is_question(text)


class QAPairStore:
    def __init__(self, config: QAPairConfig) -> None:
        self.config = config
        self.pairs: list[QAPair] = []
        # question vector cache (iteration 1.1): vectors are computed once
        # per question (at put time) so lookup becomes a single matmul
        # instead of re-embedding every question on every query.
        self._vec_cache: dict[str, np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------ #
    def put(self, question: str, answer_text: str,
            question_memory_id: str | None = None,
            answer_memory_id: str | None = None,
            ns: str | None = None,
            engine: Any = None) -> QAPair:
        """Store or refresh one question/answer pair."""
        q = question.strip()
        a = (answer_text or "").strip()
        if not q:
            raise ValueError("question must not be empty")
        if engine is not None and q not in self._vec_cache:
            try:
                self._vec_cache[q] = engine.embeddings.embed_one(q)
            except Exception:  # noqa: BLE001 - caching is best-effort
                pass
        for pair in self.pairs:
            if pair.question == q:
                pair.answer_text = a or pair.answer_text
                pair.answer_memory_id = answer_memory_id or pair.answer_memory_id
                pair.question_memory_id = question_memory_id or pair.question_memory_id
                pair.ns = ns or pair.ns
                pair.created_at = now()
                return pair
        pair = QAPair(
            question=q,
            answer_text=a,
            question_memory_id=question_memory_id,
            answer_memory_id=answer_memory_id,
            created_at=now(),
            ns=ns,
        )
        self.pairs.append(pair)
        return pair

    # ------------------------------------------------------------------ #
    def _fill_cache(self, engine: Any) -> None:
        """Batch-embed every uncached question (once, ever)."""
        missing = [p.question for p in self.pairs if p.question not in self._vec_cache]
        if not missing:
            return
        try:
            for q, v in zip(missing, engine.embeddings.embed(missing)):
                self._vec_cache[q] = v
        except Exception:  # noqa: BLE001 - degrade to per-pair embedding
            pass

    def lookup(self, question: str, engine: Any, top_k: int | None = None,
               ns: str | None = None) -> list[QAPair]:
        """Semantic lookup of stored questions; sorted by cosine desc.

        ``ns`` (module 12): when a namespace is given, only pairs stored in
        that namespace are eligible (unlabeled legacy pairs are excluded).

        Uses the cached question vectors (iteration 1.1): the per-pair
        embed loop is replaced by one batched fill + one vectorized matmul,
        turning ~1.1s (187 pairs x BGE) into single-digit milliseconds.
        """
        if not self.enabled or not self.pairs or not question.strip():
            return []
        top_k = top_k or self.config.answer_top_k
        if engine.embeddings is None:
            return []
        # provider-adaptive threshold: real embeddings score paraphrases
        # softer than the hashing signature (0.74-0.79 vs 0.85+)
        threshold = self.config.similarity_threshold
        if engine.embeddings.name != "hashing":
            threshold = min(threshold, 0.72)
        qvec = engine.embeddings.embed_one(question)
        self._fill_cache(engine)
        scored: list[tuple[float, QAPair]] = []
        cutoff = now() - self.config.max_age_days * 86400.0
        qn = float(np.linalg.norm(qvec))
        for pair in self.pairs:
            if pair.created_at < cutoff:
                continue
            if ns is not None and pair.ns != ns:
                continue
            pvec = self._vec_cache.get(pair.question)
            if pvec is None:
                continue
            cos = float(
                np.dot(qvec, pvec)
                / max(qn * float(np.linalg.norm(pvec)), 1e-12)
            )
            scored.append((cos, pair))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for c, p in scored if c >= threshold][:top_k]

    # ------------------------------------------------------------------ #
    def answer_hits(self, pairs: list[QAPair], engine: Any) -> list[SearchHit]:
        """Turn matched pairs into replay hits (synthesized memories)."""
        hits: list[SearchHit] = []
        for pair in pairs:
            answer_memory = None
            if pair.answer_memory_id:
                answer_memory = engine.memories.get(pair.answer_memory_id)
            text = answer_memory.text if answer_memory is not None else pair.answer_text
            if not text:
                continue
            mid = (
                answer_memory.id if answer_memory is not None
                else "qa_" + hashlib.sha1(pair.question.encode("utf-8")).hexdigest()[:16]
            )
            mem = _synthetic_memory(mid, text, answer_memory)
            hits.append(
                SearchHit(
                    memory=mem,
                    score=0.99,
                    breakdown=ScoreBreakdown(final=0.99),
                    region_id=answer_memory.region_id if answer_memory else None,
                    vector_score=0.99,
                    metadata_match=True,
                )
            )
        return hits

    # ------------------------------------------------------------------ #
    def count(self) -> int:
        return len(self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {"pairs": [p.to_dict() for p in self.pairs]}

    def load_dict(self, data: dict[str, Any]) -> None:
        self.pairs = [QAPair.from_dict(p) for p in data.get("pairs", [])]
        # vectors are re-embedded lazily in one batch on the first lookup
        self._vec_cache = {}


def _synthetic_memory(mid: str, text: str, original: Any = None):
    """A lightweight memory used for direct answer replay.

    Not registered in the engine store: reinforcing it is a safe no-op
    (the memory manager rejects unknown ids).
    """
    from sme.models import Memory

    return Memory(
        id=mid,
        text=text,
        metadata={"kind": "qapair_replay", "answer_for": original.metadata.get("fact_subject") if original else None},
        tags=[QUESTION_TAG, "replay"],
        source="user",
        importance=0.8,
        embedding=original.embedding if original is not None else None,
    )
