"""Persistence layer: JSON snapshot + numpy embedding store.

The engine state is written as TWO files:

    state.json[.gz]      - memories (without embeddings), regions, graph
                           edges, counters, configuration
    state.embeddings.<generation>.npz - vectors for the committed generation

Splitting embeddings out of the JSON document keeps the text snapshot small
and fast to serialize, while the vectors live in a compact binary format
(~4x smaller than JSON lists). The loader also accepts the legacy format
(schema_version 1) where embeddings were embedded inline in the JSON.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import uuid
from pathlib import Path
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
        module_state: dict | None = None,
        wal_sequence: int = 0,
    ) -> None:
        self.memories = memories
        self.regions = regions
        self.region_edges = region_edges
        self.memory_edges = memory_edges
        self.counters = counters
        self.config = config
        self.module_state = module_state
        self.wal_sequence = wal_sequence

    def to_dict(self, include_embeddings: bool = True) -> dict[str, Any]:
        data = {
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
        if self.module_state is not None:
            data["module_state"] = self.module_state
        data["wal_sequence"] = self.wal_sequence
        return data

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
            module_state=data.get("module_state"),
            wal_sequence=int(data.get("wal_sequence", 0)),
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


def _save_embeddings(path: str, memories: list[Memory], generation: str = "") -> bool:
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
            np.savez_compressed(fh, ids=np.array(ids), matrix=mat, generation=np.array(generation))
            fh.flush()
            os.fsync(fh.fileno())
        _replace_durable(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return True


def _load_embeddings(path: str, generation: str = "") -> dict[str, np.ndarray]:
    with np.load(path) as data:
        if generation and ("generation" not in data or str(data["generation"]) != generation):
            raise ValueError("embedding generation does not match snapshot")
        ids = [str(i) for i in data["ids"]]
        mat = data["matrix"]
        if mat.ndim != 2 or len(ids) != len(mat) or len(set(ids)) != len(ids):
            raise ValueError("invalid embedding matrix or duplicate ids")
    return {mid: np.array(mat[i], dtype=np.float64) for i, mid in enumerate(ids)}


def _replace_durable(source: str, destination: str) -> None:
    """Atomic publication after file fsync, including rename persistence.

    Windows MoveFileEx WRITE_THROUGH waits for the move to reach disk;
    POSIX persists the containing directory after rename.
    """
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        if not move(os.path.abspath(source), os.path.abspath(destination), 0x1 | 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.replace(source, destination)
        directory = os.open(os.path.dirname(os.path.abspath(destination)), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


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
        # gzip's footer is only complete after the writer closes.
        with open(tmp_path, "rb+") as fh:
            os.fsync(fh.fileno())
        _replace_durable(tmp_path, path)
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
    generation = uuid.uuid4().hex
    emb_base = embedding_path(path)
    emb_path = emb_base[:-4] + "." + generation + ".npz"
    if _save_embeddings(emb_path, snapshot.memories, generation):
        data["embedding_file"] = os.path.basename(emb_path)
        data["embedding_generation"] = generation
    _write_json_atomic(path, json.dumps(data, ensure_ascii=False))
    _cleanup_embeddings(path, emb_base)
    return path


def _cleanup_embeddings(path: str, emb_base: str) -> None:
    """Remove only our unreferenced generations after successful publication.

    Preserve references from both plain/gzip manifests, plus legacy sidecars.
    Cleanup failure never changes the success of an already committed save.
    """
    base = path[:-3] if path.lower().endswith(".gz") else path
    references = set()
    try:
        for manifest in (base, base + ".gz"):
            if os.path.exists(manifest):
                references.add(json.loads(_read_json(manifest)).get("embedding_file"))
        prefix = Path(emb_base).name[:-4]
        pattern = re.compile(re.escape(prefix) + r"\.[0-9a-f]{32}\.npz$")
        for candidate in Path(emb_base).parent.iterdir():
            if pattern.fullmatch(candidate.name) and candidate.name not in references:
                candidate.unlink()
    except (OSError, ValueError):
        pass


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
                if os.path.basename(emb_file) != emb_file:
                    raise ValueError("embedding_file must be a local filename")
                emb_path = os.path.join(os.path.dirname(os.path.abspath(cand)), emb_file)
                if os.path.exists(emb_path):
                    vectors = _load_embeddings(emb_path, data.get("embedding_generation", ""))
                    by_id = {m.id: m for m in snapshot.memories}
                    for mid, vec in vectors.items():
                        memory = by_id.get(mid)
                        if memory is not None:
                            memory.embedding = vec
                        elif data.get("embedding_generation"):
                            raise ValueError("embedding id absent from snapshot")
                elif data.get("embedding_generation"):
                    raise ValueError("snapshot embedding file is missing")
            return snapshot
        except Exception as exc:  # noqa: BLE001 - surface as load error
            raise ValueError(f"Failed to load snapshot from {cand}: {exc}") from exc
    return None
