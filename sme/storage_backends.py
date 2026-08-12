"""Module 08 - StorageBackends: pluggable persistence providers.

* ``LocalJsonBackend`` - the v1 JSON + NPZ snapshot (unchanged behavior,
  the default; factory returns it whenever backend == "json").
* ``SqliteBackend``    - optional SQLite backend: snapshot payload + vector
  matrix in one transactional DB file.

Factory::

    backend = build_storage_backend(config.storage, snapshot)

``backend`` defaults to "json" => the engine's behavior is byte-for-byte v1.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Protocol

import numpy as np

from sme.storage import (
    EngineSnapshot,
    load_snapshot,
    save_snapshot,
)


class StorageBackend(Protocol):
    def save(self, path: str, snapshot: EngineSnapshot, compress: bool = True) -> str: ...
    def load(self, path: str) -> EngineSnapshot | None: ...
    def query_vectors(self, path: str, vector: np.ndarray, top_k: int) -> list: ...


class LocalJsonBackend:
    """v1 behavior: JSON snapshot + npz sidecar, atomic writes."""

    name = "json"

    def save(self, path, snapshot, compress=True) -> str:
        return save_snapshot(path, snapshot, compress=compress)

    def load(self, path) -> EngineSnapshot | None:
        return load_snapshot(path)

    def query_vectors(self, path: str, vector, top_k):
        raise NotImplementedError("use the engine's in-memory space for queries")


class SqliteBackend:
    """SQLite storage: one DB file holding the snapshot + vectors."""

    name = "sqlite"

    # ------------------------------------------------------------------ #
    @staticmethod
    def _connect(path: str):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS snapshots ("
            " id INTEGER PRIMARY KEY, payload TEXT, saved_at REAL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            " memory_id TEXT PRIMARY KEY, blob BLOB NOT NULL)"
        )
        conn.commit()
        return conn

    def save(self, path: str, snapshot: EngineSnapshot, compress: bool = True) -> str:
        data = snapshot.to_dict(include_embeddings=False)
        payload = json.dumps(data, ensure_ascii=False)
        conn = self._connect(path)
        try:
            with conn:  # transaction: snapshot + vectors commit atomically
                conn.execute("DELETE FROM snapshots")
                conn.execute(
                    "INSERT INTO snapshots (payload, saved_at) VALUES (?, ?)",
                    (payload, __import__("time").time()),
                )
                conn.execute("DELETE FROM vectors")
                ids, blobs = [], []
                for m in snapshot.memories:
                    if m.embedding is not None:
                        ids.append(m.id)
                        blobs.append(
                            np.asarray(m.embedding, dtype=np.float64).tobytes()
                        )
                if ids:
                    conn.executemany(
                        "INSERT OR REPLACE INTO vectors (memory_id, blob) VALUES (?, ?)",
                        list(zip(ids, blobs)),
                    )
        finally:
            conn.close()
        return path

    def load(self, path: str) -> EngineSnapshot | None:
        if not os.path.exists(path):
            return None
        conn = self._connect(path)
        try:
            row = conn.execute(
                "SELECT payload FROM snapshots ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            data = json.loads(row[0])
            snapshot = EngineSnapshot.from_dict(data)
            rows = conn.execute("SELECT memory_id, blob FROM vectors").fetchall()
            by_id = {m.id: m for m in snapshot.memories}
            for mid, blob in rows:
                mem = by_id.get(mid)
                if mem is not None:
                    mem.embedding = np.frombuffer(blob, dtype=np.float64)
            return snapshot
        finally:
            conn.close()

    def query_vectors(self, path: str, vector, top_k: int = 10) -> list:
        """Brute-force cosine search over the stored vectors (iteration 3.3).

        Returns [(memory_id, cosine)] sorted descending. Useful for the
        REST/embedding-style query path without loading the whole engine.
        """
        if not os.path.exists(path):
            return []
        conn = self._connect(path)
        try:
            rows = conn.execute("SELECT memory_id, blob FROM vectors").fetchall()
        finally:
            conn.close()
        if not rows:
            return []
        q = np.asarray(vector, dtype=np.float64)
        q = q / np.clip(np.linalg.norm(q), 1e-12, None)
        scored: list[tuple[float, str]] = []
        for mid, blob in rows:
            v = np.frombuffer(blob, dtype=np.float64)
            n = np.linalg.norm(v)
            if n < 1e-12:
                continue
            scored.append((float(v @ q / n), mid))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]


def build_storage_backend(backend: str) -> StorageBackend:
    """Factory: json (v1, default) | sqlite."""
    if backend == "sqlite":
        return SqliteBackend()
    return LocalJsonBackend()
