"""Module 05 - FactVersion: fact versioning and correction handling.

Same fact repeated -> one canonical memory. A *correction* statement
("其实更喜欢深紫色") supersedes the old version: the old memory keeps its
history but loses retrieval weight (metadata ``superseded_by``), the new
memory becomes canonical (metadata ``supersedes``). Latest statement wins.

Disabled => no-op; the engine keeps storing every statement like v1.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from sme.config import FactVersionConfig
from sme.models import Fact

STALE_TAG = "superseded_by"
SUPERSEDE_TAG = "supersedes"


class FactVersion:
    def __init__(self, config: FactVersionConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------ #
    @staticmethod
    def is_correction(fact: Fact) -> bool:
        return fact.kind == "correction"

    def has_correction_marker(self, text: str) -> bool:
        return any(m in text for m in self.config.correct_markers)

    # ------------------------------------------------------------------ #
    def resolve(self, fact: Fact, engine: Any) -> tuple[Optional[Any], bool]:
        """Resolve a fact against existing memories.

        Returns (memory_or_None, is_new):

        * correction matched to a similar old fact -> old memory marked
          ``superseded_by``, a *new* canonical memory is created.
        * duplicate (cos >= dedup_threshold, not a correction) -> the
          existing memory is returned and nothing new is stored.
        * no match -> a new memory is created (delegated by the caller).
        """
        if not self.enabled:
            return None, True
        if self.is_correction(fact) or self.has_correction_marker(fact.text):
            return self._apply_correction(fact, engine)
        return self._dedup(fact, engine)

    # ------------------------------------------------------------------ #
    def _find_matches(self, engine: Any, fact: Fact):
        """Return (best_memory, cosine) or (None, 0.0)."""
        vec = engine.embeddings.embed_one(fact.text)
        try:
            regions = engine.space.query_regions(vec, 2)
        except Exception:  # noqa: BLE001
            regions = []
        ids: set[str] = set()
        for rhit in regions:
            ids.update(engine.space.candidates_in_region(rhit.region.id))
            if len(ids) >= 400:
                break
        best_mem, best_cos = None, 0.0
        for mid in ids:
            mem = engine.memories.get(mid)
            if mem is None or mem.archived:
                continue
            if mem.metadata.get("fact_kind") != "fact":
                continue
            if mem.metadata.get(STALE_TAG):
                continue
            cos = float(
                np.dot(vec, mem.embedding)
                / max(
                    float(np.linalg.norm(vec)) * float(np.linalg.norm(mem.embedding)),
                    1e-12,
                )
            )
            if cos > best_cos:
                best_mem, best_cos = mem, cos
        return best_mem, best_cos

    # ------------------------------------------------------------------ #
    def _threshold(self, engine: Any, name: str) -> float:
        """Provider-adaptive thresholds (hashing cosine is softer)."""
        value = getattr(self.config, name)
        if engine.embeddings.name == "hashing":
            if name == "correction_threshold":
                return min(value, 0.60)
            if name == "dedup_threshold":
                return min(value, 0.90)
        else:
            if name == "correction_threshold":
                # real embeddings: corrected statements rephrase the old one
                # (e.g. 四季春奶茶 -> 乌龙奶茶) and cosine drops to ~0.66-0.71
                return min(value, 0.66)
        return value

    def _apply_correction(self, fact: Fact, engine: Any):
        """'其实/不对/更正' -> the newest statement is canonical."""
        old, cos = self._find_matches(engine, fact)
        if old is not None and cos >= self._threshold(engine, "correction_threshold"):
            new = engine.memory_manager.add_memory(
                text=fact.text,
                metadata={
                    "fact_kind": "fact",
                    "fact_subject": fact.subject,
                    "confidence": fact.confidence,
                    SUPERSEDE_TAG: old.id,
                    "corrects": old.id,
                },
                tags=["fact", "extracted", "corrected"],
                importance=0.65,  # corrected facts are deliberately fresh
                source="user",
            )
            old.metadata[STALE_TAG] = new.id
            return new, True
        return None, True  # correction without a clear target => plain fact

    def _dedup(self, fact: Fact, engine: Any):
        """Repeat statements collapse onto the existing canonical memory."""
        old, cos = self._find_matches(engine, fact)
        if old is not None and cos >= self._threshold(engine, "dedup_threshold"):
            return old, False
        return None, True
