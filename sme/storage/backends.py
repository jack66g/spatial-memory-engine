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
import threading
from typing import Protocol

import numpy as np

from sme.storage.snapshot import (
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
    """SQLite storage: one DB file holding the snapshot + vectors.

    线程安全（uvicorn 线程池多线程访问）：连接每次操作新建、即用即关，
    ``check_same_thread=False`` 消除 sqlite 的线程亲和限制，类级锁串行化
    同进程内的并发读写（锁必须挂类上——``build_storage_backend`` 和
    ``engine.save`` 每次调用都会 new 一个 SqliteBackend 实例，实例锁挡
    不住跨实例并发）；``busy_timeout`` 兜住跨进程的文件锁竞争。
    """

    name = "sqlite"

    _class_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _connect(path: str):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout = 5000")
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
        with self._class_lock:
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
        with self._class_lock:
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
            finally:
                conn.close()
        by_id = {m.id: m for m in snapshot.memories}
        for mid, blob in rows:
            mem = by_id.get(mid)
            if mem is not None:
                mem.embedding = np.frombuffer(blob, dtype=np.float64)
        return snapshot

    def query_vectors(self, path: str, vector, top_k: int = 10) -> list:
        """Brute-force cosine search over the stored vectors (iteration 3.3).

        Returns [(memory_id, cosine)] sorted descending. Useful for the
        REST/embedding-style query path without loading the whole engine.
        """
        if not os.path.exists(path):
            return []
        with self._class_lock:
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
