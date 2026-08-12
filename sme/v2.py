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

from typing import Any, Optional

from sme.models import Fact, SearchHit
from sme.namespaces import NS_KEY
from sme.pipeline import WriteContext, WritePipeline

# score penalty applied to bare question memories outside qapair replay
QUESTION_RANK_PENALTY = 0.05


def _is_question(text: str) -> bool:
    from sme.qapair import looks_like_question

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
        for item in ctx.extra.get("canonical", []):
            if item == "__question__":
                if not engine.qapair.enabled:
                    continue
                mem = self._store_question(engine, ctx)
            elif isinstance(item, str):
                mem = engine.memory_manager.add_memory(
                    text=item, metadata=ctx.metadata, tags=ctx.tags,
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
                        },
                        tags=list(set(ctx.tags + ["fact", "extracted"])),
                        importance=0.55,
                        source=ctx.source,
                    )
            else:
                mem = item  # already stored by factversion
            ctx.primary = ctx.primary or mem
            self._after_store(engine, ctx, mem)
        return ctx

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
        self._pipeline.run(e, ctx)
        return ctx.primary

    # ------------------------------------------------------------------ #
    # search path
    # ------------------------------------------------------------------ #
    def search_pre(self, query: Any) -> list[SearchHit]:
        """Module 02: direct question/answer replay (before v1 retrieval)."""
        e = self.engine
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
            for hit in hits:
                if hit.memory.metadata.get("superseded_by"):
                    hit.score *= 1.0 - penalty
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
        return hits[: query.top_k]

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
