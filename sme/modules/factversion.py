"""Module 05 - FactVersion: fact versioning and correction handling.

Same fact repeated -> one canonical memory. A *correction* statement
("其实更喜欢深紫色") supersedes the old version: the old memory keeps its
history but loses retrieval weight (metadata ``superseded_by``), the new
memory becomes canonical (metadata ``supersedes``). Latest statement wins.

Bi-temporal timeline (Zep/Graphiti-style): every version carries two
temporal stamps in its metadata — ``valid_at`` (the claim became true /
entered the chain) and ``invalid_at`` (the claim was falsified, None =
current). Both are stamped with the *same* instant at the supersede
moment, so consecutive versions chain exactly:
``v1.invalid_at == v2.valid_at``. Legacy records written before the
stamps existed read back compatibly (``valid_at=None``, ``invalid_at``
inferred from the superseded state — see :func:`invalid_at_of`).

Disabled => no-op; the engine keeps storing every statement like v1.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from sme.config import FactVersionConfig
from sme.models import Fact
from sme.utils import now

STALE_TAG = "superseded_by"
SUPERSEDE_TAG = "supersedes"
ADD_BATCH_KEY = "add_batch"   # write-batch id: all memories from one add() call
VALID_AT_KEY = "valid_at"     # bi-temporal start: this version became true
INVALID_AT_KEY = "invalid_at"  # bi-temporal end: superseded moment (None = current)

# timeline lookup floor: below this cosine the query text is treated as
# "no such fact" (unrelated memories score <= 0.40 on real embeddings and
# <= 0.345 on hashing; a restatement of the claim itself scores far higher)
TIMELINE_LOOKUP_COS = 0.48
TIMELINE_LOOKUP_COS_HASHING = 0.40

# cycle guard for the timeline chain walk (corrupt/dense metadata)
MAX_CHAIN = 64


def valid_at_of(memory: Any) -> Optional[float]:
    """Bi-temporal start of one version.

    Legacy records (stamp absent) read as ``None`` - the moment the old
    claim became true is simply unknown, it must not be invented.
    """
    v = memory.metadata.get(VALID_AT_KEY)
    return float(v) if isinstance(v, (int, float)) else None


def invalid_at_of(memory: Any, memories: Any = None) -> Optional[float]:
    """Bi-temporal end of one version (``None`` = currently valid).

    Legacy compatibility: records written before the stamps existed carry
    no ``invalid_at``; a superseded legacy record inherits its successor's
    ``valid_at`` (fallback: ``created_at``) as the estimated falsification
    moment, so old snapshots still yield a closed timeline.
    """
    v = memory.metadata.get(INVALID_AT_KEY)
    if isinstance(v, (int, float)):
        return float(v)
    sid = memory.metadata.get(STALE_TAG)
    if memories is not None and sid:
        succ = memories.get(sid)
        if succ is not None:
            return valid_at_of(succ) or float(succ.created_at)
    return None


# stale propagation radii (iteration 2.6 - battle misled attribution):
# a correction retires the *whole old statement*, not just the single best
# matching fragment. Measured on Qwen3-Embedding (battle seeds):
#   correction-raw vs old-raw           0.64-0.79
#   old-round sibling fragments         >= 0.55 (same add batch)
#   cross-batch restatement variants    >= 0.80 (0.85 missed "养的猫叫团子,
#   2岁"-style variants carrying the same old claim, battle round 2)
#   different claims, same entities     <= 0.71
#   unrelated memories                  0.34-0.40
# The hashing signature scores softer (siblings 0.33-0.59, verbatim variants
# 0.73-0.90, unrelated <= 0.345), so the radii adapt like _threshold() does.
# Over-marking is safe-ish: search_post exempts stale memories carrying
# digits no live hit has (unique values stay findable).
BATCH_SIBLING_COS = 0.55
VARIANT_COS = 0.80
BATCH_SIBLING_COS_HASHING = 0.30
VARIANT_COS_HASHING = 0.70


def _current_batch(engine: Any) -> str | None:
    """Write-batch id set by V2Bridge.add for the running add() call."""
    bridge = getattr(engine, "_v2", None)
    return getattr(bridge, "_current_batch", None) if bridge is not None else None


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
            # the supersede moment: one instant closes the old validity
            # window and opens the new one (v_old.invalid_at == v_new.valid_at)
            ts = now()
            new = engine.memory_manager.add_memory(
                text=fact.text,
                metadata={
                    "fact_kind": "fact",
                    "fact_subject": fact.subject,
                    "confidence": fact.confidence,
                    SUPERSEDE_TAG: old.id,
                    "corrects": old.id,
                    ADD_BATCH_KEY: _current_batch(engine),
                    VALID_AT_KEY: ts,
                },
                tags=["fact", "extracted", "corrected"],
                importance=0.65,  # corrected facts are deliberately fresh
                source="user",
            )
            self._mark_stale_siblings(engine, old, new, ts)
            old.metadata[STALE_TAG] = new.id
            old.metadata[INVALID_AT_KEY] = ts
            return new, True
        return None, True  # correction without a clear target => plain fact

    # ------------------------------------------------------------------ #
    def _mark_stale_siblings(
        self, engine: Any, old: Any, new: Any, ts: Optional[float] = None
    ) -> int:
        """Propagate the stale mark across the whole old statement.

        A correction supersedes not just the best-matching fragment but every
        memory that restates the same old claim: the sibling fragments of the
        same write batch (one add() call often yields 3-4 fragments plus the
        raw text) and close cross-batch paraphrase variants. Without this,
        the old statement survives piecemeal at full score and misleads
        retrieval ("新旧并存但辨认不出最新" -> battle misled).

        ``ts`` is the shared supersede instant: every retired sibling also
        gets its ``invalid_at`` stamped with it (bi-temporal closure).
        """
        if ts is None:
            ts = now()
        old_vec = old.embedding
        if old_vec is None:
            return 0
        n = float(np.linalg.norm(old_vec))
        if n < 1e-12:
            return 0
        old_vec = old_vec / n
        old_batch = old.metadata.get(ADD_BATCH_KEY)
        hashing = getattr(engine.embeddings, "name", "") == "hashing"
        sibling_cos = BATCH_SIBLING_COS_HASHING if hashing else BATCH_SIBLING_COS
        variant_cos = VARIANT_COS_HASHING if hashing else VARIANT_COS
        marked = 0
        for mid, mem in list(engine.memories.items()):
            if mid in (old.id, new.id) or mem is None or mem.archived:
                continue
            if mem.metadata.get(STALE_TAG):
                continue
            emb = mem.embedding
            if emb is None:
                continue
            denom = float(np.linalg.norm(emb))
            if denom < 1e-12:
                continue
            cos = float(np.dot(old_vec, emb / denom))
            same_batch = (
                old_batch is not None
                and mem.metadata.get(ADD_BATCH_KEY) == old_batch
            )
            if cos >= variant_cos or (same_batch and cos >= sibling_cos):
                mem.metadata[STALE_TAG] = new.id
                if not isinstance(mem.metadata.get(INVALID_AT_KEY), (int, float)):
                    mem.metadata[INVALID_AT_KEY] = ts
                marked += 1
        return marked

    def _dedup(self, fact: Fact, engine: Any):
        """Repeat statements collapse onto the existing canonical memory."""
        old, cos = self._find_matches(engine, fact)
        if old is not None and cos >= self._threshold(engine, "dedup_threshold"):
            return old, False
        return None, True

    # ------------------------------------------------------------------ #
    # bi-temporal timeline (read side)
    # ------------------------------------------------------------------ #
    def timeline(self, memory_or_key: Any, engine: Any) -> list[dict[str, Any]]:
        """Full bi-temporal version chain of one fact, oldest -> newest.

        ``memory_or_key``: a Memory object, a memory id, or free text
        (matched by embedding; superseded versions are searchable too -
        unlike the write path, history lookup must find stale memories).
        Returns one :meth:`fact_summary` entry per backbone version; an
        empty list means "no matching fact". Module off => empty list (the
        timeline is part of this module's contract, REST reports the same).
        """
        if memory_or_key is None or not self.enabled:
            return []
        mem = memory_or_key
        if isinstance(mem, str):
            mem = self._lookup_fact(engine, mem)
            if mem is None:
                return []
        return [self.fact_summary(m, engine) for m in self._chain_backbone(engine, mem)]

    def fact_summary(self, memory: Any, engine: Any) -> dict[str, Any]:
        """REST view of one fact version (bi-temporal stamps included)."""
        invalid = invalid_at_of(memory, engine.memories)
        return {
            "id": memory.id,
            "text": memory.text,
            "created_at": memory.created_at,
            "valid_at": valid_at_of(memory),
            "invalid_at": invalid,
            "is_current": invalid is None,
            "supersedes": memory.metadata.get(SUPERSEDE_TAG),
            "superseded_by": memory.metadata.get(STALE_TAG),
        }

    def _chain_backbone(self, engine: Any, start: Any) -> list[Any]:
        """Backbone version chain (oldest -> newest) containing ``start``.

        Walk ``superseded_by`` forward to the newest version first: entering
        from any stale *sibling* of an old statement then still lands on the
        canonical chain (siblings share the stale mark but nobody supersedes
        them directly). Then collect the backbone by following ``supersedes``
        backward. Corrupt/cyclic metadata is contained by ``MAX_CHAIN``.
        """
        memories = engine.memories
        cur = start
        seen = {cur.id}
        for _ in range(MAX_CHAIN):
            nxt = memories.get(cur.metadata.get(STALE_TAG) or "")
            if nxt is None or nxt.id in seen:
                break
            seen.add(nxt.id)
            cur = nxt
        chain = [cur]
        for _ in range(MAX_CHAIN):
            prev = memories.get(chain[-1].metadata.get(SUPERSEDE_TAG) or "")
            if prev is None or prev.id in {m.id for m in chain}:
                break
            chain.append(prev)
        chain.reverse()
        return chain

    def _lookup_fact(self, engine: Any, text: str) -> Optional[Any]:
        """Best-matching fact memory for a timeline query (stale included).

        Accepts a memory id verbatim; otherwise embeds the text and scans all
        fact memories (the write path's region/candidate shortcut would be
        fine, but it excludes superseded versions - exactly what a history
        query is after). A similarity floor rejects unrelated text.
        """
        text = (text or "").strip()
        if not text:
            return None
        mem = engine.memories.get(text)
        if mem is not None:
            return mem
        vec = engine.embeddings.embed_one(text)
        floor = (
            TIMELINE_LOOKUP_COS_HASHING
            if getattr(engine.embeddings, "name", "") == "hashing"
            else TIMELINE_LOOKUP_COS
        )
        best_mem, best_cos = None, 0.0
        for m in engine.memories.values():
            if m is None or m.archived:
                continue
            if m.metadata.get("fact_kind") != "fact":
                continue
            emb = m.embedding
            if emb is None:
                continue
            cos = float(
                np.dot(vec, emb)
                / max(
                    float(np.linalg.norm(vec)) * float(np.linalg.norm(emb)),
                    1e-12,
                )
            )
            if cos > best_cos:
                best_mem, best_cos = m, cos
        return best_mem if best_cos >= floor else None
