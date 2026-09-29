"""WAL 组提交（group commit, iteration 3.4）回归。

覆盖：配置项注册（89 -> 91）、grouped 攒批/落盘/重启恢复、checkpoint
前先落盘 buffer 的顺序保证、fsync 档零漂移、flusher 异常韧性、GC 收尾
（引擎热重建场景停线程 + 补写余量）、sqlite 攒批事务 + synchronous=NORMAL。
"""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import threading
import time
import weakref

import pytest

import sme.config_items as ci
from sme.config import PersistenceConfig, SMEConfig, StorageConfig
from sme.storage.wal import WriteAheadLog


# --------------------------- 配置项注册 ----------------------------------- #
def test_config_items_registered_group_commit():
    # 91 -> 92：新增 retrieval.fusion（混合归一方式 weighted|minmax，见
    # tests/test_fusion_minmax.py；本测试原断言 91 为其加入前的注册数）
    # 92 -> 94：新增 embedding.revision / embedding.mrl_dim（向量空间版本
    # 标记 + MRL 截维，见 tests/test_embedding_revision_rebuild.py）
    # 94 -> 97：新增 maintenance.background/idle_after_s/every_s（空闲期
    # 整理 sleep-time compute，见 tests/test_sleep_maintenance.py）
    assert len(ci.ITEMS) == 97
    sync = ci.ITEM_BY_PATH["persistence.sync_mode"]
    assert sync.kind == "enum"
    assert sync.choices == ("fsync", "grouped")
    assert sync.default == "fsync"
    assert "组提交" in sync.desc and "fsync" in sync.desc
    window = ci.ITEM_BY_PATH["persistence.group_window_ms"]
    assert window.kind == "int"
    assert (window.minimum, window.maximum) == (1, 200)
    assert window.default == 15
    assert "窗口" in window.desc
    # defaults_config 含新键；解析与校验走通
    cfg = ci.defaults_config()
    assert cfg["persistence"]["sync_mode"] == "fsync"
    assert cfg["persistence"]["group_window_ms"] == 15
    assert ci.parse_value(sync, "grouped") == "grouped"
    with pytest.raises(ValueError):
        ci.parse_value(sync, "turbo")
    assert ci.validate_value(window, 15) is None
    assert ci.validate_value(window, 0) is not None
    assert ci.validate_value(window, 201) is not None


def test_persistence_config_roundtrip_compatibility():
    # 旧快照（无新键）-> 默认值不变
    old = SMEConfig.from_dict(
        {"persistence": {"enabled": True, "checkpoint_every": 20}}
    )
    assert old.persistence.sync_mode == "fsync"
    assert old.persistence.group_window_ms == 15
    # 新键读写往返
    cfg = SMEConfig.from_dict(
        {"persistence": {"sync_mode": "grouped", "group_window_ms": 42}}
    )
    assert cfg.persistence.sync_mode == "grouped"
    assert cfg.persistence.group_window_ms == 42
    dumped = cfg.to_dict()["persistence"]
    assert dumped["sync_mode"] == "grouped"
    assert dumped["group_window_ms"] == 42


# --------------------------- WAL 单元（文件模式） ------------------------- #
def _grouped_wal(tmp_path, window=200) -> WriteAheadLog:
    cfg = PersistenceConfig(
        enabled=True, sync_mode="grouped", group_window_ms=window,
        wal_path=str(tmp_path / "t.wal"),
    )
    return WriteAheadLog(cfg)


def _lines(path) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip()]


def test_grouped_appends_buffer_close_flushes(tmp_path):
    wal = _grouped_wal(tmp_path, window=200)
    wal.append({"op": "add", "mid": "m1"})
    wal.append({"op": "add", "mid": "m2"})
    assert wal.ops == 2
    assert wal._flusher is not None and wal._flusher.is_alive()
    wal.close()
    lines = _lines(wal.path)
    assert len(lines) == 2
    assert json.loads(lines[0])["mid"] == "m1"
    assert json.loads(lines[1])["mid"] == "m2"


def test_grouped_flusher_periodic_flush(tmp_path):
    wal = _grouped_wal(tmp_path, window=5)
    for i in range(20):
        wal.append({"op": "add", "mid": f"m{i}"})
    for _ in range(200):  # 等 flusher 窗口到期
        if len(_lines(wal.path)) == 20:
            break
        time.sleep(0.01)
    assert len(_lines(wal.path)) == 20
    wal.close()


def test_fsync_mode_zero_drift_immediate_write(tmp_path):
    cfg = PersistenceConfig(
        enabled=True, sync_mode="fsync",
        wal_path=str(tmp_path / "f.wal"),
    )
    wal = WriteAheadLog(cfg)
    wal.append({"op": "add", "mid": "m1"})
    # fsync 档：append 返回即已在盘上，且从不启动 flusher
    assert len(_lines(wal.path)) == 1
    assert wal._flusher is None
    assert wal.stats()["sync_mode"] == "fsync"
    wal.close()


def test_reset_flushes_buffer_before_truncate(tmp_path, monkeypatch):
    wal = _grouped_wal(tmp_path, window=200)
    wal.append({"op": "add", "mid": "m1"})
    wal.append({"op": "delete", "mid": "mX"})
    captured: dict[str, str] = {}
    real_remove = os.remove

    def spy_remove(p, *args, **kwargs):
        p = str(p)
        if os.path.basename(p) == "t.wal" and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                captured["content"] = fh.read()
        return real_remove(p, *args, **kwargs)

    monkeypatch.setattr(os, "remove", spy_remove)
    wal.reset()
    # checkpoint 语义：截断瞬间 buffer 里的 ops 必须已经落盘
    assert '"m1"' in captured.get("content", "")
    assert '"mX"' in captured.get("content", "")
    assert not os.path.exists(wal.path)
    assert wal.ops == 0


def test_flusher_restarts_after_close(tmp_path):
    wal = _grouped_wal(tmp_path, window=200)
    wal.append({"op": "add", "mid": "m1"})
    wal.close()
    assert len(_lines(wal.path)) == 1
    # engine.load 重绑路径后继续写入：flusher 惰性重启
    wal.append({"op": "add", "mid": "m2"})
    wal.append({"op": "add", "mid": "m3"})
    assert wal._flusher is not None and wal._flusher.is_alive()
    wal.close()
    assert len(_lines(wal.path)) == 3


def test_grouped_replay_roundtrip(tmp_path):
    wal = _grouped_wal(tmp_path, window=200)
    for i in range(4):
        wal.append({"op": "add", "mid": f"m{i}", "text": f"t{i}"})
    wal.close()

    class _Mem:
        pass

    class _FakeEngine:
        def __init__(self):
            self.memories = {}
            self.memory_manager = _Mem()

    engine = _FakeEngine()
    count = wal.replay(engine)
    assert count == 4
    assert wal.ops == 0
    assert not os.path.exists(wal.path)


def test_has_pending_counts_buffered_ops(tmp_path):
    wal = _grouped_wal(tmp_path, window=200)
    assert wal.has_pending() is False
    wal.append({"op": "add", "mid": "m1"})
    # 攒批未落盘：文件还不存在，但内存积压必须算 pending
    assert wal.has_pending() is True
    wal.close()
    assert wal.has_pending() is True  # 已落盘未 checkpoint
    wal.reset()
    assert wal.has_pending() is False


def test_flusher_survives_io_failure_and_retries(tmp_path, monkeypatch):
    wal = _grouped_wal(tmp_path, window=1)
    state = {"fail": True}
    real_fsync = os.fsync

    def flaky_fsync(fd):
        if state["fail"]:
            raise OSError("simulated fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr("sme.storage.wal.os.fsync", flaky_fsync)
    wal.append({"op": "add", "mid": "m1"})
    thread = wal._flusher
    time.sleep(0.3)
    # 落盘持续失败：线程必须活着，行滞留 buffer 等重试（不静默丢失）
    assert thread is not None and thread.is_alive()
    assert len(wal._buf) >= 1
    state["fail"] = False
    for _ in range(200):  # 恢复后下个窗口重试成功
        if _lines(wal.path):
            break
        time.sleep(0.02)
    assert any('"m1"' in l for l in _lines(wal.path))
    assert thread.is_alive()
    wal.close()


def test_gc_stops_flusher_and_flushes_remainder(tmp_path):
    # 引擎热重建只丢引用不调 close：WAL 被 GC 时 finalizer 停线程并补写
    path = tmp_path / "gc.wal"
    wal = WriteAheadLog(PersistenceConfig(
        enabled=True, sync_mode="grouped", group_window_ms=200,
        wal_path=str(path),
    ))
    wal.append({"op": "add", "mid": "m1"})
    assert wal._flusher is not None and wal._flusher.is_alive()
    ref = weakref.ref(wal)
    base_threads = threading.active_count()
    del wal
    for _ in range(300):
        gc.collect()
        if ref() is None:
            break
        time.sleep(0.01)
    assert ref() is None, "flusher 持强引用会阻止 WAL 被 GC"
    for _ in range(300):  # finalizer 置 stop 后线程自行退出
        if threading.active_count() <= base_threads:
            break
        time.sleep(0.01)
    assert threading.active_count() <= base_threads, "GC 后 flusher 线程未退出"
    assert any('"m1"' in l for l in _lines(str(path))), "GC 收尾未补写攒批余量"


# --------------------------- WAL 单元（sqlite 模式） ---------------------- #
def _sqlite_grouped_wal(tmp_path, mode="grouped", window=5) -> WriteAheadLog:
    storage = StorageConfig(path=str(tmp_path / "t.db"), backend="sqlite")
    cfg = PersistenceConfig(enabled=True, sync_mode=mode,
                            group_window_ms=window)
    return WriteAheadLog(cfg, storage, sqlite=True)


def _wal_rows(path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM wal_ops").fetchone()[0]
    finally:
        conn.close()


def test_sqlite_grouped_batches(tmp_path):
    wal = _sqlite_grouped_wal(tmp_path)
    for i in range(7):
        wal.append({"op": "add", "mid": f"m{i}"})
    for _ in range(200):
        if _wal_rows(wal.path) == 7:
            break
        time.sleep(0.01)
    assert _wal_rows(wal.path) == 7
    # grouped 连接走 synchronous=NORMAL
    with wal._db_lock:
        wal._open()
        pragma = wal._conn.execute("PRAGMA synchronous").fetchone()[0]
    assert pragma == 1  # NORMAL
    wal.close()


def test_sqlite_fsync_mode_pragma_unchanged(tmp_path):
    wal = _sqlite_grouped_wal(tmp_path, mode="fsync")
    wal.append({"op": "add", "mid": "m1"})
    with wal._db_lock:
        wal._open()
        pragma = wal._conn.execute("PRAGMA synchronous").fetchone()[0]
    assert pragma == 2  # FULL（现行为不变）
    assert _wal_rows(wal.path) == 1  # fsync 档 append 即落库
    wal.close()


# --------------------------- 引擎级端到端 --------------------------------- #
def test_engine_grouped_file_roundtrip(fresh_engine, new_engine, tmp_path):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "g.json")
    e.config.persistence.enabled = True
    e.config.persistence.sync_mode = "grouped"
    for i in range(5):
        e.add(f"grouped memory {i}")
    e.save(e.config.storage.path)  # checkpoint：buffer 先落盘再截断
    for i in range(3):
        e.add(f"grouped new {i}")
    time.sleep(0.1)  # flusher 窗口
    assert e.wal.ops == 3

    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    e2.config.persistence.enabled = True
    e2.config.persistence.sync_mode = "grouped"
    assert e2.load(e.config.storage.path) is True
    assert len(e2.memories) == 8, "grouped WAL replay failed"
    assert e2.wal.ops == 0
    e.wal.close()


def test_engine_grouped_sqlite_roundtrip(fresh_engine, tmp_path):
    e = fresh_engine
    path = str(tmp_path / "g.db")
    e.config.storage.backend = "sqlite"
    e.config.storage.path = path
    e.config.persistence.enabled = True
    e.config.persistence.sync_mode = "grouped"
    for i in range(4):
        e.add(f"grouped sqlite memory {i}")
    e.save(path)
    e.add("grouped sqlite new one")
    time.sleep(0.1)
    assert e.wal.sqlite is True
    assert e.wal.ops == 1

    from sme.config import SMEConfig as _C
    e2_cfg = _C()
    e2_cfg.storage.backend = "sqlite"
    e2_cfg.storage.path = path
    e2_cfg.storage.autosave = False
    e2_cfg.persistence.enabled = True
    e2_cfg.persistence.sync_mode = "grouped"
    from sme.engine import SpatialMemoryEngine
    e2 = SpatialMemoryEngine(e2_cfg)
    assert e2.load(path) is True
    assert len(e2.memories) == 5, "grouped sqlite WAL replay failed"
    assert e2.wal.ops == 0
    e.wal.close()
    e2.wal.close()
