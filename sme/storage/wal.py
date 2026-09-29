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
import threading
from typing import Any, Union

from sme.config import PersistenceConfig, StorageConfig
from sme.utils import logger


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
        # sqlite 模式的连接串行化锁（RLock：reset/has_pending 会在持锁时
        # 调 _open，不可重入会自锁死）
        self._db_lock = threading.RLock()
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
            with self._db_lock:
                if self._conn is not None:
                    return
                directory = os.path.dirname(os.path.abspath(self.path))
                os.makedirs(directory, exist_ok=True)
                # check_same_thread=False + _db_lock：uvicorn 线程池里不同的
                # 线程会先后使用这条持久连接，sqlite 默认的线程亲和检查会
                # 直接抛 "SQLite objects created in a thread can only be
                # used in that same thread"
                self._conn = sqlite3.connect(
                    self.path, check_same_thread=False
                )
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
            with self._db_lock:
                # 拿到锁后连接可能已被并发的 reset() 关掉：重开再写
                if self._conn is None:
                    self._open()  # RLock 可重入，持锁重开安全
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
        """Truncate the WAL after a successful checkpoint.

        删除失败不允许静默：WAL 文件还在就说明 checkpoint 实际没有完成，
        重启后会重放已入快照的 ops。失败时记 warning 并把 OSError 抛给
        调用方（engine.save 路径因此显式失败被上层看到；replay 路径在
        replay() 内部兜住，不会阻断 load）。失败时 ``self.ops`` 保持
        pending，不谎报已 checkpoint。
        """
        self.close()
        if self.sqlite:
            with self._db_lock:
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
            except OSError as exc:
                logger.warning(
                    "WAL checkpoint 未完成：无法删除 %s（%s），"
                    "重启后将重放其中的 ops", self.path, exc,
                )
                raise
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
        # 只有文件模式才做"文件存在"探测：sqlite 模式下 path 就是快照库
        # 本身，按文件存在性判断恒真（库在即"WAL 在"）；sqlite 分支直接
        # 读 wal_ops 表，空表自然重放 0 条
        if not self.sqlite and not os.path.exists(self.path):
            return 0
        if self.sqlite:
            with self._db_lock:
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
            try:
                self.reset()
            except OSError:
                # 重放本身已成功，引擎状态正确；WAL 文件删不掉只意味着
                # 下次启动会把这些 ops 再（幂等地）重放一遍，不能因此
                # 让整个 load 失败。reset() 内部已记 warning，不静默。
                pass
        return count

    @staticmethod
    def _apply(engine: Any, op: dict[str, Any]) -> None:
        kind = op.get("op")
        mid = op.get("mid")
        mm = engine.memory_manager
        try:
            if kind == "add":
                if mid and mid in engine.memories:
                    # 重放幂等：该 add 已在快照/先前重放中落库（archived
                    # 记忆仍留在 memories 里，同样被覆盖）。重复执行会用
                    # op 里的少数字段重建 Memory，重置 hit_count/weight/
                    # version/created_at/last_hit，并对 Region 质心二次
                    # 累加——直接跳过。update/delete/archive/restore 天然
                    # 幂等，不需要 guard。
                    return
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

    def has_pending(self) -> bool:
        """True when the WAL still holds ops not checkpointed into a snapshot.

        文件模式探测 ``.wal`` 文件是否存在且非空；sqlite 模式查
        ``wal_ops`` 行数——sqlite 模式下 ``path`` 就是快照库本身，调用方
        若按 ``os.path.getsize(wal.path) > 0`` 探测会恒真（库存在即
        "有积压"），导致每次重启都白做一次全量 save。探测失败按无积压
        处理，不阻断恢复路径。
        """
        if not self.enabled:
            return False
        try:
            if self.sqlite:
                if not os.path.exists(self.path):
                    return False
                with self._db_lock:
                    self._open()
                    assert self._conn is not None
                    row = self._conn.execute(
                        "SELECT COUNT(*) FROM wal_ops"
                    ).fetchone()
                    self.close()
                return bool(row and row[0])
            return os.path.exists(self.path) and os.path.getsize(self.path) > 0
        except Exception:  # noqa: BLE001 - 探测失败视为无积压
            return False

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": self.path,
            "backend": "sqlite" if self.sqlite else "file",
            "pending_ops": self.ops,
            "replayed": self.replayed,
            "checkpointed": self.checkpointed,
        }
