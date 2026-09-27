"""Persistence layer: JSON snapshot + numpy embedding store.

The engine state is written as TWO files:

    state.json[.gz]      - memories (without embeddings), regions, graph
                           edges, counters, configuration
    state.embeddings.npz - {memory_id -> vector} as a numpy array store

Splitting embeddings out of the JSON document keeps the text snapshot small
and fast to serialize, while the vectors live in a compact binary format
(~4x smaller than JSON lists). The loader also accepts the legacy format
(schema_version 1) where embeddings were embedded inline in the JSON.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any

import numpy as np

from sme.config import SMEConfig
from sme.models import Memory, MemoryEdge, Region, RegionEdge


class EngineSnapshot:
    """Serializable engine state."""

    def __init__(
        self,
        memories: list[Memory],
        regions: list[Region],
        region_edges: list[RegionEdge],
        memory_edges: list[MemoryEdge],
        counters: dict[str, int],
        config: SMEConfig,
    ) -> None:
        self.memories = memories
        self.regions = regions
        self.region_edges = region_edges
        self.memory_edges = memory_edges
        self.counters = counters
        self.config = config

    def to_dict(self, include_embeddings: bool = True) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "counters": self.counters,
            "config": self.config.to_dict(),
            "memories": [
                m.to_dict(include_embedding=include_embeddings) for m in self.memories
            ],
            "regions": [r.to_dict() for r in self.regions],
            "region_edges": [e.to_dict() for e in self.region_edges],
            "memory_edges": [e.to_dict() for e in self.memory_edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineSnapshot":
        return cls(
            memories=[Memory.from_dict(m) for m in data.get("memories", [])],
            regions=[Region.from_dict(r) for r in data.get("regions", [])],
            region_edges=[
                RegionEdge.from_dict(e) for e in data.get("region_edges", [])
            ],
            memory_edges=[
                MemoryEdge.from_dict(e) for e in data.get("memory_edges", [])
            ],
            counters=data.get("counters", {}),
            config=SMEConfig.from_dict(data.get("config", {})),
        )


# --------------------------------------------------------------------------- #
# embedding sidecar
# --------------------------------------------------------------------------- #
def embedding_path(path: str) -> str:
    """Sidecar npz path for a snapshot path (e.g. state.json.gz)."""
    base = path[:-3] if path.lower().endswith(".gz") else path
    if base.lower().endswith(".json"):
        base = base[:-5]
    return base + ".embeddings.npz"


def _save_embeddings(path: str, memories: list[Memory]) -> bool:
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for m in memories:
        if m.embedding is not None:
            ids.append(m.id)
            vectors.append(np.asarray(m.embedding, dtype=np.float64))
    if not ids:
        return False
    mat = np.stack(vectors)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # write to a temp file and atomically replace: an interrupted save must
    # never leave a truncated npz (a torn sidecar would brick the snapshot)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, ids=np.array(ids), matrix=mat)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return True


def _load_embeddings(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        ids = [str(i) for i in data["ids"]]
        mat = data["matrix"]
    return {mid: np.array(mat[i], dtype=np.float64) for i, mid in enumerate(ids)}


# --------------------------------------------------------------------------- #
# atomic JSON write (gzip or plain)
# --------------------------------------------------------------------------- #
def _write_json_atomic(path: str, payload: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".tmp_{os.getpid()}_{os.path.basename(path)}")
    try:
        if path.lower().endswith(".gz"):
            with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
                fh.write(payload)
        else:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(payload)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _read_json(path: str) -> str:
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save_snapshot(path: str, snapshot: EngineSnapshot, compress: bool = True) -> str:
    """Write the snapshot to disk: JSON state + npz embeddings sidecar."""
    if compress and path.lower().endswith(".json"):
        path = path + ".gz"
    payload = json.dumps(snapshot.to_dict(include_embeddings=False), ensure_ascii=False)
    data = json.loads(payload)
    emb_path = embedding_path(path)
    if _save_embeddings(emb_path, snapshot.memories):
        data["embedding_file"] = os.path.basename(emb_path)
    _write_json_atomic(path, json.dumps(data, ensure_ascii=False))
    return path


def load_snapshot(path: str) -> EngineSnapshot | None:
    """Load a snapshot from disk. Returns None if the file does not exist."""
    candidates = [path]
    if path.lower().endswith(".json"):
        candidates.append(path + ".gz")
    for cand in candidates:
        if not os.path.exists(cand):
            continue
        try:
            data = json.loads(_read_json(cand))
            snapshot = EngineSnapshot.from_dict(data)
            # attach the embedding sidecar (new format) or keep the inline
            # embeddings (legacy format); missing vectors stay None
            emb_file = data.get("embedding_file")
            if emb_file:
                emb_path = os.path.join(os.path.dirname(os.path.abspath(cand)), emb_file)
                if os.path.exists(emb_path):
                    vectors = _load_embeddings(emb_path)
                    by_id = {m.id: m for m in snapshot.memories}
                    for mid, vec in vectors.items():
                        memory = by_id.get(mid)
                        if memory is not None:
                            memory.embedding = vec
            return snapshot
        except Exception as exc:  # noqa: BLE001 - surface as load error
            raise ValueError(f"Failed to load snapshot from {cand}: {exc}") from exc
    return None
