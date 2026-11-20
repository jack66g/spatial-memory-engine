"""空闲期整理（sleep-time compute，Letta/Anthropic 路线）回归。

覆盖：配置项注册（94 -> 97）、零漂移（默认不启线程 / engine_stats 无
maintenance 段）、空闲判定（写活动未静默不整理）、周期实时读配置、
后台自动融合（不手动调 consolidate）、维护线程持锁整理时并发写排队
不死锁、单步失败不中断其他步骤、维护项全关不空转、GC 停线程（引擎
热重建场景）、load 重绑快照配置后重启线程。
"""

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest

import sme.config_items as ci
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine


# --------------------------- 配置项注册 ----------------------------------- #
def test_config_items_registered_maintenance():
    # 94 -> 97：maintenance.background / idle_after_s / every_s
    # （test_wal_grouped.test_config_items_registered_group_commit 同步断言 97）
    for path in ("maintenance.background", "maintenance.idle_after_s",
                 "maintenance.every_s"):
        assert path in ci.ITEM_BY_PATH
    bg = ci.ITEM_BY_PATH["maintenance.background"]
    assert bg.kind == "bool" and bg.default is False
    assert bg.group == "空闲整理" and "sleep-time" in bg.desc
    idle = ci.ITEM_BY_PATH["maintenance.idle_after_s"]
    assert idle.kind == "float"
    assert (idle.minimum, idle.maximum) == (0, 86400)
    assert idle.default == 30.0
    every = ci.ITEM_BY_PATH["maintenance.every_s"]
    assert every.kind == "float"
    assert (every.minimum, every.maximum) == (1, 86400)
    assert every.default == 300.0
    # defaults_config 含新键；解析与校验走通
    cfg = ci.defaults_config()
    assert cfg["maintenance"] == {
        "background": False, "idle_after_s": 30.0, "every_s": 300.0,
    }
    assert ci.parse_value(bg, "开") is True
    assert ci.parse_value(idle, "45.5") == 45.5
    assert ci.validate_value(every, 300.0) is None
    assert ci.validate_value(every, 0) is not None
    assert ci.validate_value(idle, float("nan")) is not None
    # 预设覆盖不误伤：任何预设都不含 maintenance 键（默认关）
    for preset in ci.PRESETS:
        assert not any(k.startswith("maintenance.")
                       for k in preset["values"])


def test_maintenance_config_roundtrip_compatibility():
    # 旧快照（无 maintenance 组）-> 默认值不变（零漂移：load 旧快照不开线程）
    old = SMEConfig.from_dict({"storage": {"path": "x.json"}})
    assert old.maintenance.background is False
    assert old.maintenance.idle_after_s == 30.0
    assert old.maintenance.every_s == 300.0
    # 新键读写往返
    cfg = SMEConfig.from_dict({
        "maintenance": {"background": True, "idle_after_s": 5.5, "every_s": 60},
    })
    assert cfg.maintenance.background is True
    assert cfg.maintenance.idle_after_s == 5.5
    assert cfg.maintenance.every_s == 60
    dumped = cfg.to_dict()["maintenance"]
    assert dumped == {"background": True, "idle_after_s": 5.5, "every_s": 60}


# --------------------------- 引擎级 --------------------------------------- #
def _engine(tmp_path, background=True, idle_after_s=0.05, every_s=0.1, **cfg_tune):
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "engine.json")
    cfg.maintenance.background = background
    cfg.maintenance.idle_after_s = idle_after_s
    cfg.maintenance.every_s = every_s
    for key, value in cfg_tune.items():
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)
    return SpatialMemoryEngine(cfg)


def _wait(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def _add_clusters(e, n=12):
    for i in range(n):
        e.add(f"user likes apple fruit variant {i}")
        e.add(f"user plays basketball sport day {i}")


def test_zero_drift_background_off(tmp_path):
    e = _engine(tmp_path, background=False)
    assert e._maint_thread is None  # 不启线程
    m = e.add("plain memory")
    assert e.search("plain")[0].memory.id == m.id
    stats = e.engine_stats()
    assert "maintenance" not in stats  # 关 = 无 v1 diff
    assert e._maint_runs == 0


def test_background_thread_runs_and_consolidates(tmp_path):
    e = _engine(tmp_path)  # idle_after_s=0.05, every_s=0.1
    assert e._maint_thread is not None and e._maint_thread.is_alive()
    _add_clusters(e)
    # 不手动调 consolidate：空闲后台自动融合出摘要
    assert _wait(lambda: e.consolidation.consolidation_count > 0), \
        "空闲整理未自动生成融合摘要"
    stats = e.engine_stats()
    maint = stats["maintenance"]
    assert maint["enabled"] is True
    assert maint["thread_alive"] is True
    assert maint["runs"] >= 1
    assert maint["last_run_at"] is not None
    assert maint["last_duration_s"] is not None
    assert maint["last_error"] is None
    assert any(m.source == "summary" for m in e.memories.values())


def test_maintenance_telemetry_event(tmp_path):
    e = _engine(tmp_path, **{"observability.enabled": True})
    _add_clusters(e, n=6)
    assert _wait(lambda: any(
        ev["event"] == "maintenance" for ev in e.telemetry.events
    )), "telemetry 未记录 maintenance 事件"
    ev = [ev for ev in e.telemetry.events if ev["event"] == "maintenance"][-1]
    assert ev["runs"] >= 1 and "duration_s" in ev
    assert "consolidated" in ev or "decayed" in ev


def test_idle_gate_skips_until_quiet(tmp_path):
    # idle_after_s 很大：持续有写活动（静默不足）时永远不整理
    e = _engine(tmp_path, idle_after_s=3600.0)
    _add_clusters(e, n=6)
    time.sleep(0.6)
    assert e._maint_runs == 0
    assert e.consolidation.consolidation_count == 0
    stats = e.engine_stats()
    assert stats["maintenance"]["runs"] == 0


def test_all_steps_disabled_no_spin(tmp_path):
    # decay 关 + 记忆数低于融合/压缩门槛：全关不空转（不进锁、不计 runs）
    e = _engine(tmp_path, **{"policy.decay_enabled": False})
    e.add("only one memory")
    time.sleep(0.6)
    assert e._maint_runs == 0
    assert e.engine_stats()["maintenance"]["runs"] == 0


def test_concurrent_add_queues_on_lock_no_deadlock(tmp_path):
    # 维护线程在引擎锁内整理时，主线程写入在锁上排队、释放后完成
    e = _engine(tmp_path, every_s=0.1)
    _add_clusters(e, n=6)
    real_consolidate = e.consolidate
    entered = threading.Event()

    def slow_consolidate():
        entered.set()  # 此刻维护线程正持有 engine._lock
        time.sleep(0.4)
        return real_consolidate()

    e.consolidate = slow_consolidate
    assert _wait(lambda: entered.is_set()), "维护线程未进入整理"
    t0 = time.time()
    m = e.add("concurrent write during maintenance")
    elapsed = time.time() - t0
    assert m.id in e.memories
    assert elapsed < 10.0, "维护持锁期间并发写死锁"


def test_step_failure_does_not_break_other_steps(tmp_path):
    e = _engine(tmp_path)

    def boom():
        raise RuntimeError("decay exploded")

    e.apply_decay = boom
    _add_clusters(e, n=6)
    assert _wait(lambda: e.consolidation.consolidation_count > 0), \
        "decay 失败中断了 consolidate"
    maint = e.engine_stats()["maintenance"]
    assert maint["last_error"] is not None and "decay" in maint["last_error"]


def test_interval_read_live_from_config(tmp_path):
    # 周期实时读：先给一个长周期，运行中改短，应在一秒内跑起来
    e = _engine(tmp_path, every_s=30.0)
    e.add("a memory to maintain")
    time.sleep(0.3)
    assert e._maint_runs == 0
    e.config.maintenance.every_s = 0.1
    assert _wait(lambda: e._maint_runs >= 1, timeout=3.0), \
        "运行中改 every_s 未生效（周期没有实时读配置）"
    # 配置翻关：线程下一跳读配置后自行退出
    e.config.maintenance.background = False
    thread = e._maint_thread
    assert thread is not None
    assert _wait(lambda: not thread.is_alive(), timeout=5.0), \
        "配置翻关后维护线程未退出"


def test_gc_stops_maintenance_thread(tmp_path):
    # 引擎热重建只丢引用不调 close：GC 后 finalizer 停线程。
    # 先 gc.collect() 清掉前序测试遗留引擎的维护线程，保证计数干净。
    gc.collect()
    e = _engine(tmp_path)
    _add_clusters(e, n=3)
    thread = e._maint_thread
    assert thread is not None and thread.is_alive()
    base = threading.active_count()  # 已含本引擎的维护线程
    ref = weakref.ref(e)
    del e
    for _ in range(300):
        gc.collect()
        if ref() is None:
            break
        time.sleep(0.01)
    assert ref() is None, "维护线程持强引用会阻止引擎被 GC"
    assert _wait(lambda: not thread.is_alive(), timeout=5.0), \
        "GC 后维护线程未退出"
    assert _wait(lambda: threading.active_count() <= base, timeout=5.0)


def test_load_rebind_restarts_thread_from_snapshot(tmp_path):
    # 快照带着 background=True/every_s=0.1/idle_after_s=2 存盘；新引擎默认
    # 关，load 重绑快照配置后线程必须启动并读新配置。存盘引擎的 idle 留
    # 2s 余量：其自身线程在 save 前后不会插进来整理（快照一致性）。
    path = str(tmp_path / "snap.json")
    e = _engine(tmp_path, every_s=0.1, idle_after_s=2.0)
    _add_clusters(e, n=3)
    assert e._maint_runs == 0  # 2s 内 save 完成，源引擎从未整理
    e.save(path)

    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = path
    e2 = SpatialMemoryEngine(cfg)
    assert e2._maint_thread is None  # 构造时 background=False：不启线程
    assert e2.load(path) is True
    assert e2.config.maintenance.background is True  # 快照配置生效
    assert e2.config.maintenance.every_s == 0.1
    assert e2._maint_thread is not None and e2._maint_thread.is_alive()
    # load 视为一次写活动：空闲计时从 load 完成起算，2s 静默后开始整理
    assert _wait(lambda: e2._maint_runs >= 1, timeout=10.0), \
        "load 后维护线程未整理"
    maint = e2.engine_stats()["maintenance"]
    assert maint["every_s"] == 0.1 and maint["runs"] >= 1
