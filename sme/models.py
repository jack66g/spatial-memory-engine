"""Core data models for the Spatial Memory Engine.

Runtime models keep numpy arrays in memory; serialization (to_dict/from_dict)
converts them to plain JSON-compatible structures for persistence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from sme.utils import bbox_of, now, vector_mean


def new_id(prefix: str = "m") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
@dataclass
class Memory:
    """A single memory record.

    Attributes:
        id: unique identifier
        text: the memory content
        metadata: free-form key/value metadata
        created_at / last_hit: unix timestamps
        hit_count: how many times this memory has been reinforced
        weight: reinforcement weight (grows with hits, shrinks with decay)
        importance: semantic importance in [0, 1]
        freshness: recency indicator in [0, 1]
        decay_factor: current decay multiplier in [0, 1]
        region_id: id of the spatial region the memory lives in
        neighbors: ids of spatially close memories
        tags: user tags
        summary: optional generated summary text
        version: incremented on every update
        parent_id / children: memory graph (consolidation / summaries)
        archived: whether the memory is in cold storage
        embedding: the embedding vector (runtime: np.ndarray, storage: list)
        source: provenance - "user" | "llm" | "summary" | "system"
    """

    id: str = field(default_factory=lambda: new_id("m"))
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now)
    last_hit: float = field(default_factory=now)
    hit_count: int = 0
    weight: float = 1.0
    importance: float = 0.5
    freshness: float = 1.0
    decay_factor: float = 1.0
    region_id: Optional[str] = None
    neighbors: set[str] = field(default_factory=set)
    tags: list[str] = field(default_factory=list)
    summary: Optional[str] = None
    version: int = 1
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    archived: bool = False
    embedding: Optional[np.ndarray] = None
    source: str = "user"

    # ------------------------------------------------------------------ #
    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_hit": self.last_hit,
            "hit_count": self.hit_count,
            "weight": self.weight,
            "importance": self.importance,
            "freshness": self.freshness,
            "decay_factor": self.decay_factor,
            "region_id": self.region_id,
            "neighbors": sorted(self.neighbors),
            "tags": self.tags,
            "summary": self.summary,
            "version": self.version,
            "parent_id": self.parent_id,
            "children": self.children,
            "archived": self.archived,
            "source": self.source,
        }
        if include_embedding and self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        emb = data.get("embedding")
        mem = cls(
            id=data.get("id", new_id("m")),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", now()),
            last_hit=data.get("last_hit", now()),
            hit_count=data.get("hit_count", 0),
            weight=data.get("weight", 1.0),
            importance=data.get("importance", 0.5),
            freshness=data.get("freshness", 1.0),
            decay_factor=data.get("decay_factor", 1.0),
            region_id=data.get("region_id"),
            neighbors=set(data.get("neighbors", [])),
            tags=data.get("tags", []),
            summary=data.get("summary"),
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
            archived=data.get("archived", False),
            source=data.get("source", "user"),
        )
        if emb is not None:
            mem.embedding = np.asarray(emb, dtype=np.float64)
        return mem

    def touched(self) -> None:
        """Mark the memory as freshly created/updated."""
        self.last_hit = now()
        self.freshness = 1.0
        self.version += 1


# --------------------------------------------------------------------------- #
# Spatial structures
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    """A density-based region of the spatial memory space.

    Regions are irregular, grow dynamically, and support split / merge /
    neighbor edges. A region is defined by its centroid, member set, radius
    and axis-aligned bounding box (min/max).
    """

    id: str = field(default_factory=lambda: new_id("r"))
    centroid: Optional[np.ndarray] = None
    member_ids: set[str] = field(default_factory=set)
    radius: float = 0.0
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None
    density: float = 0.0
    created_at: float = field(default_factory=now)
    last_updated: float = field(default_factory=now)
    parent_region: Optional[str] = None
    generation: int = 1
    # runtime-only flag: the centroid is kept exact incrementally, while
    # radius/density/bbox are deferred until a consumer actually needs them
    _geometry_stale: bool = False

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def mark_stale(self) -> None:
        """Centroid stays exact; radius/density/bbox are now out of date."""
        self._geometry_stale = True
        self.last_updated = now()

    def refresh_if_stale(
        self,
        vectors: dict[str, np.ndarray],
        dim: int,
    ) -> None:
        """Exact recompute of the deferred geometry when it is stale."""
        if self._geometry_stale:
            self.update_geometry(vectors, dim)
            self._geometry_stale = False

    def update_geometry(
        self,
        vectors: dict[str, np.ndarray],
        dim: int,
    ) -> None:
        """Recompute centroid, radius, bbox and density from member vectors."""
        members = [vectors[mid] for mid in self.member_ids if mid in vectors]
        if not members:
            self.centroid = np.zeros(dim, dtype=np.float64)
            self.radius = 0.0
            self.density = 0.0
            return
        self.centroid = vector_mean(members)
        dists = np.linalg.norm(np.stack(members) - self.centroid, axis=1)
        self.radius = float(np.max(dists)) if len(dists) else 0.0
        self.bbox_min, self.bbox_max = bbox_of(members, dim)
        # density: members per unit of "size" (mean distance to centroid)
        self.density = len(members) / (1.0 + float(np.mean(dists)))
        self.last_updated = now()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "member_ids": sorted(self.member_ids),
            "radius": self.radius,
            "density": self.density,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "parent_region": self.parent_region,
            "generation": self.generation,
        }
        if self.centroid is not None:
            d["centroid"] = self.centroid.tolist()
        if self.bbox_min is not None:
            d["bbox_min"] = self.bbox_min.tolist()
        if self.bbox_max is not None:
            d["bbox_max"] = self.bbox_max.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Region":
        def arr(key: str):
            v = data.get(key)
            return None if v is None else np.asarray(v, dtype=np.float64)

        return cls(
            id=data.get("id", new_id("r")),
            centroid=arr("centroid"),
            member_ids=set(data.get("member_ids", [])),
            radius=data.get("radius", 0.0),
            bbox_min=arr("bbox_min"),
            bbox_max=arr("bbox_max"),
            density=data.get("density", 0.0),
            created_at=data.get("created_at", now()),
            last_updated=data.get("last_updated", now()),
            parent_region=data.get("parent_region"),
            generation=data.get("generation", 1),
        )


@dataclass
class RegionEdge:
    """Undirected edge between two regions."""

    source: str
    target: str
    distance: float
    kind: str = "neighbor"  # neighbor | overlap | sibling

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "distance": self.distance,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            distance=data.get("distance", 0.0),
            kind=data.get("kind", "neighbor"),
        )


@dataclass
class MemoryEdge:
    """Directed or undirected edge in the memory graph."""

    source: str
    target: str
    kind: str  # reference | cause | conversation | summary | parent | child | neighbor
    weight: float = 1.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "weight": self.weight,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            kind=data.get("kind", "reference"),
            weight=data.get("weight", 1.0),
            note=data.get("note", ""),
        )


# --------------------------------------------------------------------------- #
# Retrieval results
# --------------------------------------------------------------------------- #
@dataclass
class ScoreBreakdown:
    """Transparent decomposition of a final memory score."""

    semantic: float = 0.0
    importance: float = 0.0
    freshness: float = 0.0
    weight: float = 0.0
    decay: float = 0.0
    hit_count: float = 0.0
    recency: float = 0.0
    region: float = 0.0
    final: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "importance": self.importance,
            "freshness": self.freshness,
            "weight": self.weight,
            "decay": self.decay,
            "hit_count": self.hit_count,
            "recency": self.recency,
            "region": self.region,
            "final": self.final,
        }


@dataclass
class SearchHit:
    memory: Memory
    score: float
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    region_id: Optional[str] = None
    region_score: float = 0.0
    keyword_score: float = 0.0
    vector_score: float = 0.0
    metadata_match: bool = True

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        d = self.memory.to_dict(include_embedding=include_embedding)
        d["score"] = self.score
        d["breakdown"] = self.breakdown.to_dict()
        d["region_id"] = self.region_id
        d["region_score"] = self.region_score
        d["keyword_score"] = self.keyword_score
        d["vector_score"] = self.vector_score
        d["metadata_match"] = self.metadata_match
        return d


@dataclass
class RegionHit:
    region: Region
    score: float
    distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region.id,
            "score": self.score,
            "distance": self.distance,
            "size": self.region.size,
            "density": self.region.density,
        }


# --------------------------------------------------------------------------- #
# Engine stats
# --------------------------------------------------------------------------- #
@dataclass
class MemoryStats:
    total: int = 0
    active: int = 0
    archived: int = 0
    summarized: int = 0
    total_hits: int = 0
    avg_weight: float = 0.0
    avg_importance: float = 0.0
    avg_hit_count: float = 0.0
    oldest_age_days: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# v2 modules (v2 模块设计) data structures. These never touch the v1
# serialized snapshot keys - module state lives in separate sidecar files,
# so the v1 snapshot schema (v2) stays backward compatible.
# --------------------------------------------------------------------------- #
@dataclass
class Fact:
    """One extracted fact (module 01)."""

    text: str
    kind: str = "fact"          # fact | question | correction | chat | qa
    confidence: float = 1.0
    subject: str = "用户"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "confidence": self.confidence,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fact":
        return cls(
            text=data.get("text", ""),
            kind=data.get("kind", "fact"),
            confidence=float(data.get("confidence", 1.0)),
            subject=data.get("subject", "用户"),
        )


@dataclass
class QAPair:
    """A question/answer pair (module 02)."""

    question: str
    answer_text: str
    question_memory_id: str | None = None
    answer_memory_id: str | None = None
    created_at: float = field(default_factory=now)
    ns: str | None = None  # optional namespace tag (module 12 isolation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer_text": self.answer_text,
            "question_memory_id": self.question_memory_id,
            "answer_memory_id": self.answer_memory_id,
            "created_at": self.created_at,
            "ns": self.ns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAPair":
        return cls(
            question=data.get("question", ""),
            answer_text=data.get("answer_text", ""),
            question_memory_id=data.get("question_memory_id"),
            answer_memory_id=data.get("answer_memory_id"),
            created_at=float(data.get("created_at", now())),
            ns=data.get("ns"),
        )


@dataclass
class Entity:
    """A named entity with a temporal validity window (module 03)."""

    id: str = field(default_factory=lambda: new_id("e"))
    name: str = ""
    kind: str = "person"        # person | place | item | org | other
    valid_at: float = field(default_factory=now)
    invalid_at: float | None = None
    memory_ids: list[str] = field(default_factory=list)

    def is_valid(self, reference: float | None = None) -> bool:
        ref = now() if reference is None else reference
        return self.valid_at <= ref and (self.invalid_at is None or ref <= self.invalid_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "memory_ids": list(self.memory_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        return cls(
            id=data.get("id", new_id("e")),
            name=data.get("name", ""),
            kind=data.get("kind", "person"),
            valid_at=float(data.get("valid_at", now())),
            invalid_at=data.get("invalid_at"),
            memory_ids=list(data.get("memory_ids", [])),
        )


@dataclass
class Relation:
    """A temporal relation between two entities (module 03)."""

    id: str = field(default_factory=lambda: new_id("r"))
    source: str = ""            # entity id
    target: str = ""            # entity id
    predicate: str = ""         # e.g. "friend", "works_at", "likes"
    valid_at: float = field(default_factory=now)
    invalid_at: float | None = None
    memory_ids: list[str] = field(default_factory=list)

    def is_valid(self, reference: float | None = None) -> bool:
        ref = now() if reference is None else reference
        return self.valid_at <= ref and (self.invalid_at is None or ref <= self.invalid_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "predicate": self.predicate,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "memory_ids": list(self.memory_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        return cls(
            id=data.get("id", new_id("r")),
            source=data.get("source", ""),
            target=data.get("target", ""),
            predicate=data.get("predicate", ""),
            valid_at=float(data.get("valid_at", now())),
            invalid_at=data.get("invalid_at"),
            memory_ids=list(data.get("memory_ids", [])),
        )


@dataclass
class RegionStats:
    count: int = 0
    avg_size: float = 0.0
    avg_density: float = 0.0
    max_size: int = 0
    min_size: int = 0
    avg_radius: float = 0.0
    edge_count: int = 0
    avg_neighbors: float = 0.0
    split_count: int = 0
    merge_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()
