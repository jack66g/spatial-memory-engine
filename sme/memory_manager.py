"""Memory Manager.

Owns the lifecycle of every Memory record:

    add / update / delete / archive / restore / reinforce / tag / summary
    parent-children management / versioning / statistics

Coordinates with the SpatialMemorySpace (embedding + region membership),
the MemoryGraph, the embedding provider, and the policy/reinforcement/decay
subsystems.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from sme.archive import ArchiveManager
from sme.config import SMEConfig
from sme.decay import MemoryDecay
from sme.embedding.base import EmbeddingProvider
from sme.graph import (
    KIND_CAUSE,
    KIND_CONVERSATION,
    KIND_PARENT,
    KIND_REFERENCE,
    KIND_SUMMARY,
    MemoryGraph,
)
from sme.models import Memory, MemoryStats
from sme.policy import MemoryPolicy
from sme.reinforcement import EbbinghausReinforcement
from sme.space.space import SpatialMemorySpace
from sme.utils import now


class MemoryManager:
    def __init__(
        self,
        config: SMEConfig,
        space: SpatialMemorySpace,
        embeddings: EmbeddingProvider,
        graph: MemoryGraph,
        policy: MemoryPolicy,
        reinforcement: EbbinghausReinforcement,
        decay: MemoryDecay,
        archive: ArchiveManager,
        on_upsert: Optional[Any] = None,
        on_delete: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.space = space
        self.embeddings = embeddings
        self.graph = graph
        self.policy = policy
        self.reinforcement = reinforcement
        self.decay = decay
        self.archive = archive
        self.on_upsert = on_upsert
        self.on_delete = on_delete
        self.memories: dict[str, Memory] = {}

    # ------------------------------------------------------------------ #
    # add
    # ------------------------------------------------------------------ #
    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        tags: Iterable[str] | None = None,
        importance: float = 0.5,
        embedding: Optional[Any] = None,
        source: str = "user",
        link_to: Optional[str] = None,
        link_kind: str = KIND_REFERENCE,
        memory_id: Optional[str] = None,
    ) -> Memory:
        if not text or not text.strip():
            raise ValueError("memory text must not be empty")
        vector = embedding
        if vector is None:
            vector = self.embeddings.embed_one(text)
        memory = Memory(
            **({"id": memory_id} if memory_id else {}),
            text=text,
            metadata=metadata or {},
            tags=list(tags or []),
            importance=min(max(importance, 0.0), 1.0),
            embedding=vector,
            source=source,
        )
        self._register(memory)
        if link_to is not None:
            self.graph.add_edge(memory.id, link_to, link_kind)
        return memory

    def add_many(self, texts: list[str], **kwargs) -> list[Memory]:
        vectors = self.embeddings.embed(texts)
        out = []
        for text, vector in zip(texts, vectors):
            memory = Memory(
                text=text,
                metadata=kwargs.get("metadata", {}) or {},
                tags=list(kwargs.get("tags", []) or []),
                importance=kwargs.get("importance", 0.5),
                embedding=vector,
                source=kwargs.get("source", "user"),
            )
            self._register(memory)
            out.append(memory)
        return out

    def _register(self, memory: Memory) -> None:
        assert memory.embedding is not None
        memory.region_id = self.space.insert(memory.id, memory.embedding)
        self.memories[memory.id] = memory
        if self.on_upsert is not None:
            self.on_upsert(memory)

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #
    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        weight: Optional[float] = None,
        summary: Optional[str] = None,
    ) -> Memory:
        memory = self.require(memory_id)
        changed = False
        if text is not None and text.strip() != memory.text:
            memory.text = text
            memory.embedding = self.embeddings.embed_one(text)
            # re-insert into the spatial space
            self.space.remove(memory.id)
            memory.region_id = self.space.insert(memory.id, memory.embedding)
            changed = True
        if metadata is not None and metadata != memory.metadata:
            memory.metadata = metadata
            changed = True
        if tags is not None and tags != memory.tags:
            memory.tags = tags
            changed = True
        if importance is not None and importance != memory.importance:
            memory.importance = min(max(importance, 0.0), 1.0)
            changed = True
        if weight is not None and weight != memory.weight:
            memory.weight = max(0.0, weight)
            changed = True
        if summary is not None and summary != memory.summary:
            memory.summary = summary
            changed = True
        if changed:
            # only a substantive change is a "touch": a no-op update (or a
            # metadata-only patch) must not reset last_hit/freshness/version
            memory.touched()
            if self.on_upsert is not None:
                self.on_upsert(memory)
        return memory

    # ------------------------------------------------------------------ #
    # delete / archive
    # ------------------------------------------------------------------ #
    def delete_memory(self, memory_id: str) -> bool:
        memory = self.memories.pop(memory_id, None)
        if memory is None:
            return False
        self.space.remove(memory_id)
        self.graph.remove_edges_for(memory_id)
        for other in self.memories.values():
            other.neighbors.discard(memory_id)
            if memory_id in other.children:
                other.children.remove(memory_id)
            if other.parent_id == memory_id:
                other.parent_id = None
        if self.on_delete is not None:
            self.on_delete(memory_id)
        return True

    def archive_memory(self, memory_id: str) -> bool:
        memory = self.memories.get(memory_id)
        if memory is None or memory.archived:
            return False
        # NOTE: graph edges and neighbors are intentionally KEPT for archived
        # memories - restore() brings them back intact, and graph traversal
        # already skips archived nodes (retriever._graph_expand).
        self.space.remove(memory_id)
        self.archive.archive(memory)
        if self.on_delete is not None:
            self.on_delete(memory_id)
        return True

    def restore_memory(self, memory_id: str) -> bool:
        memory = self.memories.get(memory_id)
        if memory is None or not memory.archived:
            return False
        assert memory.embedding is not None
        memory.region_id = self.space.insert(memory.id, memory.embedding)
        self.archive.restore(memory)
        if self.on_upsert is not None:
            self.on_upsert(memory)
        return True

    # ------------------------------------------------------------------ #
    # reinforcement & decay
    # ------------------------------------------------------------------ #
    def reinforce(self, memory_id: str, reference: float | None = None) -> Optional[dict]:
        memory = self.memories.get(memory_id)
        if memory is None or memory.archived:
            return None
        return self.reinforcement.on_hit(memory, reference)

    def apply_decay_all(self, reference: float | None = None) -> int:
        count = 0
        for memory in self.memories.values():
            if memory.archived:
                continue
            self.decay.apply(memory, reference)
            count += 1
        return count

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def require(self, memory_id: str) -> Memory:
        if memory_id not in self.memories:
            raise KeyError(f"memory {memory_id} not found")
        return self.memories[memory_id]

    def get(self, memory_id: str) -> Optional[Memory]:
        return self.memories.get(memory_id)

    def list_memories(
        self,
        tag: Optional[str] = None,
        source: Optional[str] = None,
        archived: Optional[bool] = None,
        include_archived: bool = False,
    ) -> list[Memory]:
        out: list[Memory] = []
        for memory in self.memories.values():
            if not include_archived and memory.archived:
                continue
            if archived is not None and memory.archived != archived:
                continue
            if tag and tag not in memory.tags:
                continue
            if source and memory.source != source:
                continue
            out.append(memory)
        return out

    def stats(self) -> MemoryStats:
        active = [m for m in self.memories.values() if not m.archived]
        s = MemoryStats(
            total=len(self.memories),
            active=len(active),
            archived=self.archive.cold_count(),
            summarized=sum(1 for m in self.memories.values() if m.summary),
            total_hits=sum(m.hit_count for m in self.memories.values()),
        )
        if active:
            s.avg_weight = sum(m.weight for m in active) / len(active)
            s.avg_importance = sum(m.importance for m in active) / len(active)
            s.avg_hit_count = sum(m.hit_count for m in active) / len(active)
            s.oldest_age_days = (now() - min(m.created_at for m in active)) / 86400.0
        return s

    # ------------------------------------------------------------------ #
    # graph helpers
    # ------------------------------------------------------------------ #
    def link(self, a: str, b: str, kind: str = KIND_REFERENCE, weight: float = 1.0, note: str = "") -> bool:
        if a not in self.memories or b not in self.memories:
            return False
        self.graph.add_edge(a, b, kind, weight=weight, note=note)
        if kind == KIND_CAUSE:
            self.graph.add_edge(b, a, KIND_CONVERSATION, weight=1.0, note="timeline")
        return True

    def set_parent(self, child_id: str, parent_id: str) -> bool:
        if child_id not in self.memories or parent_id not in self.memories:
            return False
        child = self.memories[child_id]
        parent = self.memories[parent_id]
        child.parent_id = parent_id
        if child_id not in parent.children:
            parent.children.append(child_id)
        self.graph.add_edge(child_id, parent_id, KIND_PARENT, weight=1.0, note="child")
        return True

    def summary_memory(
        self,
        text: str,
        member_ids: Iterable[str],
        importance: float = 0.7,
    ) -> Memory:
        """Create a summary memory and wire it as parent of `member_ids`."""
        summary = self.add_memory(
            text=text,
            importance=importance,
            source="summary",
            tags=["summary"],
        )
        summary.summary = text
        for mid in member_ids:
            self.set_parent(mid, summary.id)
            self.graph.add_edge(summary.id, mid, KIND_SUMMARY, weight=1.0, note="covers")
        return summary

    def load_state(self, memories: list[Memory], graph: dict) -> None:
        self.memories = {m.id: m for m in memories}
        self.graph.load_dict(graph)
