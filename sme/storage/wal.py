"""Module 07 - IncrementalPersistence: write-ahead log + periodic checkpoints.

Solves the v1 bottleneck where every round re-saved the whole 100k snapshot
(15-30s). With this module enabled, every write op appends ONE JSON line to
a WAL (or one row to a sqlite ``wal_ops`` table when the storage backend is
sqlite - iteration 3.3), and a full snapshot is written only every
``checkpoint_every`` ops. Loading = snapshot + WAL replay, so a crash loses
at most the ops between the last checkpoint and the crash (auto-recovered).

Disabled => the engine keeps the original full-save path (v1 behavior).

The WAL is an append-only log::

    {"op": "add",     "mid": ..., "text": ..., "metadata": ..., "tags": ...}
    {"op": "update",  "mid": ...}
    {"op": "delete",  "mid": ...}
    {"op": "archive", "mid": ...}
    {"op": "restore", "mid": ...}
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Union

from sme.config import PersistenceConfig, StorageConfig


def default_wal_path(storage_path: str) -> str:
    return storage_path + ".wal"


class WriteAheadLog:
    def __init__(self, config: PersistenceConfig,
                 storage: Union[StorageConfig, str] = "",
                 sqlite: bool = False) -> None:
        self.config = config
        # a live StorageConfig reference keeps the WAL path in sync when the
        # engine's storage path is changed after construction; a plain string
        # keeps the old static-path behavior for direct instantiations
        self._storage = storage if isinstance(storage, StorageConfig) else None
        self._static_path = "" if isinstance(storage, StorageConfig) else str(storage)
        self._sqlite_flag = sqlite  # static fallback for direct instantiations
        self._fh = None
        self._conn: sqlite3.Connection | None = None
        self.ops = 0
        self.replayed = 0
        self.checkpointed = 0

    @property
    def sqlite(self) -> bool:
        """sqlite mode is derived live from the storage backend config, so
        changing ``engine.config.storage.backend`` after construction works."""
        if self._storage is not None:
            return self._storage.backend == "sqlite"
        return self._sqlite_flag

    @property
    def path(self) -> str:
        if self._storage is not None:
            base = self._storage.path
        else:
            base = self._static_path
        if self.sqlite:
            # sqlite mode reuses the snapshot db itself (transactional)
            return base
        if self.config.wal_path:
            return self.config.wal_path
        return default_wal_path(base)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------ #
    def _open(self):
        if self.sqlite:
            if self._conn is not None:
                return
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            if self.config.sync_mode == "fsync":
                self._conn.execute("PRAGMA synchronous = FULL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS wal_ops ("
                " seq INTEGER PRIMARY KEY AUTOINCREMENT, op TEXT NOT NULL)"
            )
            self._conn.commit()
            return
        if self._fh is not None:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, op: dict[str, Any]) -> None:
        """Append one op (single write + fsync in fsync mode)."""
        if not self.enabled:
            return
        self._open()
        try:
            line = json.dumps(op, ensure_ascii=False)
        except (TypeError, ValueError):
            # metadata may hold non-JSON-safe values (numpy scalars, sets...);
            # never let that break the write path - drop the unsafe fields
            safe = {k: v for k, v in op.items() if k != "metadata"}
            line = json.dumps(safe, ensure_ascii=False)
        if self.sqlite:
            assert self._conn is not None
            self._conn.execute("INSERT INTO wal_ops (op) VALUES (?)", (line,))
            self._conn.commit()
        else:
            assert self._fh is not None
            self._fh.write(line + "\n")
            self._fh.flush()
            if self.config.sync_mode == "fsync":
                os.fsync(self._fh.fileno())
        self.ops += 1

    def ops_since_checkpoint(self) -> int:
        return self.ops

    def reset(self) -> None:
        """Truncate the WAL after a successful checkpoint."""
        self.close()
        if self.sqlite:
            if not os.path.exists(self.path):
                self.ops = 0
                return
            self._open()
            assert self._conn is not None
            self._conn.execute("DELETE FROM wal_ops")
            self._conn.commit()
            self.close()
        elif os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        self.ops = 0

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    # ------------------------------------------------------------------ #
    def replay(self, engine: Any) -> int:
        """Replay pending ops onto an already-loaded engine."""
        if not os.path.exists(self.path):
            return 0
        if self.sqlite:
            self._open()
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT op FROM wal_ops ORDER BY seq"
            ).fetchall()
            lines = [row[0] for row in rows]
        else:
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                op = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._apply(engine, op)
            count += 1
        self.replayed += count
        if count:
            self.reset()
        return count

    @staticmethod
    def _apply(engine: Any, op: dict[str, Any]) -> None:
        kind = op.get("op")
        mid = op.get("mid")
        mm = engine.memory_manager
        try:
            if kind == "add":
                mm.add_memory(
                    text=op.get("text", ""),
                    metadata=op.get("metadata", {}) or {},
                    tags=op.get("tags", []) or [],
                    importance=float(op.get("importance", 0.5)),
                    source=op.get("source", "user"),
                    memory_id=mid,
                )
            elif kind == "update" and mid:
                mm.update_memory(
                    mid,
                    text=op.get("text"),
                    metadata=op.get("metadata"),
                    tags=op.get("tags"),
                    importance=op.get("importance"),
                    weight=op.get("weight"),
                    summary=op.get("summary"),
                )
            elif kind == "delete" and mid:
                mm.delete_memory(mid)
            elif kind == "archive" and mid:
                mm.archive_memory(mid)
            elif kind == "restore" and mid:
                mm.restore_memory(mid)
        except Exception:  # noqa: BLE001 - tolerate partial/corrupt WALs
            pass

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": self.path,
            "backend": "sqlite" if self.sqlite else "file",
            "pending_ops": self.ops,
            "replayed": self.replayed,
            "checkpointed": self.checkpointed,
        }
