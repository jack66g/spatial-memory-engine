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

Durability modes (``persistence.sync_mode``) - iteration 3.4:

- ``fsync`` (default, unchanged legacy behavior): every append does one
  write + flush + fsync. Safest, but SSD fsync costs 0.5-5ms per op.
- ``grouped`` (group commit): append only buffers the line in memory and
  wakes a daemon flusher thread; the flusher merges everything buffered
  into ONE write + flush + fsync every ``group_window_ms`` (or when 256
  ops pile up). sqlite mode batches the inserts into one transaction with
  ``PRAGMA synchronous = NORMAL``. Crash window = one batching window.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import weakref
from typing import Any, Union

from sme.config import PersistenceConfig, StorageConfig
from sme.utils import logger

# grouped 模式攒批上限：buffer 达到该条数立即唤醒 flusher（不等窗口）
_GROUP_FLUSH_CAP = 256


def default_wal_path(storage_path: str) -> str:
    return storage_path + ".wal"


def _wal_path_of(persist: PersistenceConfig,
                 storage: Union[StorageConfig, None],
                 static_path: str,
                 sqlite: bool) -> str:
    """``WriteAheadLog.path`` 的纯函数版：GC 收尾时对象已不可达，需要用
    相同的输入单独计算落盘路径（property 调不了）。"""
    base = storage.path if storage is not None else static_path
    if sqlite:
        return base
    if persist.wal_path:
        return persist.wal_path
    return default_wal_path(base)


def _gc_finalize(buf: list[str],
                 buf_lock: threading.Lock,
                 wake: threading.Event,
                 stop: threading.Event,
                 persist: PersistenceConfig,
                 storage: Union[StorageConfig, None],
                 static_path: str,
                 sqlite_flag: bool) -> None:
    """WAL 对象被 GC（引擎热重建 / 整体丢弃）时的收尾。

    引擎没有 close()，重建路径只是丢掉旧引擎引用：后台 flusher 线程若不
    停掉，会一直持有旧 WAL 的文件句柄（Windows 下导致新引擎 reset 时
    os.remove 报 PermissionError）。线程只持 weakref，WAL 不可达时由此
    finalizer 停线程，并把仍在内存攒批、尚未落盘的 ops 按路径补写进
    WAL（句柄重开追加；``_fh`` 本体随引用计数回收自动关闭）。
    """
    stop.set()
    wake.set()
    with buf_lock:
        lines = buf[:]
        del buf[:]
    if not lines:
        return
    sqlite = (storage.backend == "sqlite") if storage is not None else sqlite_flag
    path = _wal_path_of(persist, storage, static_path, sqlite)
    try:
        if sqlite:
            conn = sqlite3.connect(path, check_same_thread=False)
            try:
                conn.executemany(
                    "INSERT INTO wal_ops (op) VALUES (?)",
                    [(line,) for line in lines],
                )
                conn.commit()
            finally:
                conn.close()
        else:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("".join(line + "\n" for line in lines))
                fh.flush()
                os.fsync(fh.fileno())
    except Exception:  # noqa: BLE001 - GC 上下文绝不能抛
        logger.warning(
            "WAL grouped 模式 GC 收尾：%d 条攒批 ops 补写 %s 失败",
            len(lines), path, exc_info=True,
        )


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
        # ---- grouped 组提交状态（fsync/off 模式不碰，惰性启动） ---- #
        # buffer 永不重新绑定：flusher 线程与 GC finalizer 持同一个 list
        self._buf: list[str] = []
        self._buf_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._flusher: threading.Thread | None = None
        self._finalizer = None
        self._refresh_finalizer()
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
        return _wal_path_of(self.config, self._storage, self._static_path,
                            self.sqlite)

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
                # used by that same thread"
                self._conn = sqlite3.connect(
                    self.path, check_same_thread=False
                )
                if self.config.sync_mode == "fsync":
                    self._conn.execute("PRAGMA synchronous = FULL")
                elif self.config.sync_mode == "grouped":
                    # 组提交：攒批事务 + NORMAL（事务提交仍持久化，但不为
                    # 每次提交等完整 fsync 序列，批量摊薄后足够安全）
                    self._conn.execute("PRAGMA synchronous = NORMAL")
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

    # ------------------------------------------------------------------ #
    # grouped 组提交（group commit）
    # ------------------------------------------------------------------ #
    def _refresh_finalizer(self) -> None:
        """（重新）注册 GC 收尾 finalizer。

        engine.load 会重绑 ``wal.config`` / ``wal._storage``（快照带来的新
        存储路径），finalizer 捕获的是旧对象；每次 flusher 启动时刷新一次，
        保证 GC 补写用的是当前生效的路径。旧的先 detach，避免重复触发。
        """
        if self._finalizer is not None:
            self._finalizer.detach()
        self._finalizer = weakref.finalize(
            self, _gc_finalize,
            self._buf, self._buf_lock, self._wake, self._stop,
            self.config, self._storage, self._static_path, self._sqlite_flag,
        )

    def _ensure_flusher(self) -> None:
        """grouped 模式：惰性启动后台 flusher（close/reset 停掉后可重启）。

        线程目标只持 ``weakref.ref(self)`` + 共享原语——绝不持 WAL 强引用，
        否则 flusher 活着 WAL 就永远不可达，GC 收尾（停线程/关句柄）永不
        触发，热重建会泄漏线程并占住 WAL 文件句柄。
        """
        if self._flusher is not None and self._flusher.is_alive():
            return
        with self._buf_lock:
            if self._flusher is not None and self._flusher.is_alive():
                return
            self._refresh_finalizer()
            self._stop.clear()
            self._flusher = threading.Thread(
                target=WriteAheadLog._flusher_main,
                args=(weakref.ref(self), self._buf, self._buf_lock,
                      self._wake, self._stop),
                name="sme-wal-flusher",
                daemon=True,
            )
            self._flusher.start()

    @staticmethod
    def _flusher_main(wal_ref: "weakref.ReferenceType[WriteAheadLog]",
                      buf: list[str],
                      buf_lock: threading.Lock,
                      wake: threading.Event,
                      stop: threading.Event) -> None:
        """后台攒批落盘循环（daemon 线程）。

        每 group_window_ms（或 buffer 攒满被唤醒）执行一次合并落盘；
        窗口参数实时读配置，热调 group_window_ms 无需重启。异常绝不静默
        杀线程：单次落盘失败记日志，ops 推回 buffer 头部下个窗口重试。
        WAL 被 GC 后 wal_ref() 返回 None，线程自行退出（余量由
        _gc_finalize 补写）。

        阻塞在 ``wake.wait()`` 期间必须不持任何 WAL 强引用（局部 ``wal``
        读完窗口即删，醒后重新解析）——否则等待中的局部变量会把 WAL
        钉在内存里，GC 收尾永远不触发。
        """
        while not stop.is_set():
            wal = wal_ref()
            if wal is None:
                return
            try:
                window = max(1, wal.config.group_window_ms) / 1000.0
                del wal  # 关键：阻塞等待期间允许 WAL 被 GC
                wake.wait(window)
                wake.clear()
                wal = wal_ref()
                if wal is None:
                    return
                wal._flush_buffer()
                del wal
            except Exception:  # noqa: BLE001 - flusher 不能被静默杀死
                logger.warning(
                    "WAL grouped flusher 落盘失败（%d 条滞留攒批，下个窗口重试）",
                    len(buf), exc_info=True,
                )
                time.sleep(0.05)

    def _flush_buffer(self) -> None:
        """把攒批 buffer 一次性落盘。

        文件模式：合并为一次 write + flush + fsync；sqlite 模式：
        executemany + 单次 commit（攒批事务）。失败时把未确认的 lines
        推回 buffer 头部并重新抛出——flusher 重试、close 记日志、reset
        向上抛（保持"reset 不吞错"的既有契约）。部分写入后失败可能在
        重试时产生重复行：重放侧 add 有幂等 guard，update/delete/
        archive/restore 天然幂等，可容忍。
        """
        with self._buf_lock:
            if not self._buf:
                return
            lines = self._buf[:]
            del self._buf[:]
        try:
            if self.sqlite:
                with self._db_lock:
                    # 拿到锁后连接可能已被并发的 reset() 关掉：重开再写
                    if self._conn is None:
                        self._open()  # RLock 可重入，持锁重开安全
                    assert self._conn is not None
                    self._conn.executemany(
                        "INSERT INTO wal_ops (op) VALUES (?)",
                        [(line,) for line in lines],
                    )
                    self._conn.commit()
            else:
                self._open()
                assert self._fh is not None
                self._fh.write("".join(line + "\n" for line in lines))
                self._fh.flush()
                os.fsync(self._fh.fileno())
        except Exception:
            with self._buf_lock:
                self._buf[0:0] = lines
            raise

    # ------------------------------------------------------------------ #
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
        if self.config.sync_mode == "grouped":
            # 组提交：只进内存攒批 + 攒满唤醒；落盘交给后台 flusher
            self._ensure_flusher()
            with self._buf_lock:
                self._buf.append(line)
                self.ops += 1
                full = len(self._buf) >= _GROUP_FLUSH_CAP
            if full:
                self._wake.set()
            return
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

        grouped 模式：checkpoint 语义 = 快照已含全部 ops，截断前必须先
        把攒批 buffer 落盘（覆盖截断瞬间并发写入窗口）；落盘失败同样
        上抛，不静默丢 ops。
        """
        if self._buf:
            self._flush_buffer()  # 失败上抛：截断前 buffer 未落盘
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

    def close(self, stop_flusher: bool = True) -> None:
        """关闭文件/sqlite 句柄；grouped 模式先停后台 flusher 并补写攒批余量。

        ``stop_flusher=False`` 供 has_pending 这类"持 _db_lock 探测"路径
        使用：join flusher 可能等它抢同一把 _db_lock，持锁 join 会白等
        到超时；探测路径只需关探测连接，flusher 继续跑、下次落盘时
        ``_flush_buffer`` 会自行重开连接。进程退出场景由 GC finalizer /
        解释器退出钩子兜底。补写失败只记日志——close 在 engine.load
        重绑等路径上是兜底动作，不能抛。
        """
        if stop_flusher and (
            self.config.sync_mode == "grouped" or self._flusher is not None
        ):
            self._stop.set()
            self._wake.set()
            flusher = self._flusher
            if (flusher is not None and flusher.is_alive()
                    and flusher is not threading.current_thread()):
                flusher.join(2.0)
            self._flusher = None
        if self._buf:
            try:
                self._flush_buffer()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "WAL close：攒批余量落盘失败（%d 条滞留 buffer）",
                    len(self._buf), exc_info=True,
                )
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
        处理，不阻断恢复路径。grouped 模式下仍在内存攒批、未落盘的 ops
        同样算积压。
        """
        if not self.enabled:
            return False
        if self._buf:
            return True
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
                # 关探测连接放在锁外语义不变（锁只保护 execute），且避免
                # 持 _db_lock join flusher（见 close 的 stop_flusher 说明）
                self.close(stop_flusher=False)
                return bool(row and row[0])
            return os.path.exists(self.path) and os.path.getsize(self.path) > 0
        except Exception:  # noqa: BLE001 - 探测失败视为无积压
            return False

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": self.path,
            "backend": "sqlite" if self.sqlite else "file",
            "sync_mode": self.config.sync_mode,
            "pending_ops": self.ops,
            "buffered_ops": len(self._buf),
            "replayed": self.replayed,
            "checkpointed": self.checkpointed,
        }
