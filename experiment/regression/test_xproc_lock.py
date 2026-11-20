# -*- coding: utf-8 -*-
"""跨进程文件锁 + 丢更新防护回归(2026-11-20 生产化轮)。

覆盖:
L1 并发写风暴:多进程同路径交错 save → 串行化,终态可加载、无撕裂/孤儿
L2 丢更新防护:A/B 都 load 后 B 先 save,A 再 save 必须如实报错(不静默覆盖)
L3 锁超时如实抛错,不静默继续
L4 导出路径(save/load 到非恢复存储路径)不污染丢更新指纹
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sme import SpatialMemoryEngine, SMEConfig


def _engine(path):
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(path)
    return SpatialMemoryEngine(cfg)


# ------------------------------------------------------------------ #
# L1: 并发写风暴
# ------------------------------------------------------------------ #
def _writer(worker_id, state_path, n, q):
    try:
        from sme import SpatialMemoryEngine, SMEConfig
        cfg = SMEConfig()
        cfg.storage.autosave = False
        cfg.storage.path = state_path
        e = SpatialMemoryEngine(cfg)
        # load-first 语义:先与磁盘同步,再写自己的批次
        loaded = e.load(state_path)
        base = len(e.memories)
        for i in range(n):
            e.add(f"进程{worker_id}的记忆{i}")
        saved = 0
        for attempt in range(5):
            try:
                e.save(state_path)
                saved = 1
                break
            except RuntimeError:
                # 别的进程先落盘了 → 重载合并自己的批次再试(丢失更新不走静默)
                e2 = _engine_reload(state_path)
                for i in range(n):
                    e2.add(f"进程{worker_id}的记忆{i}")
                e = e2
        q.put(("ok" if saved else "err", worker_id, len(e.memories) if saved else 0))
    except Exception as ex:
        q.put(("err", worker_id, repr(ex)))


def _engine_reload(state_path):
    e = _engine(state_path)
    e.load(state_path)
    return e


def test_concurrent_save_storm_no_loss_no_corruption(tmp_path):
    state = str(tmp_path / "shared.json")
    seed = _engine(state)
    seed.add("种子记忆")
    seed.save(state)

    q = mp.Queue()
    procs = [mp.Process(target=_writer, args=(w, state, 20, q)) for w in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    statuses = [q.get(timeout=5) for _ in range(3)]
    assert all(s[0] == "ok" for s in statuses), statuses

    e = SpatialMemoryEngine()
    ok = e.load(state)
    assert ok
    texts = [m.text for m in e.memories.values()]
    n_proc = sum(1 for t in texts if t.startswith("进程"))
    # 三进程 load-first + 锁串行 + 冲突重载合并 → 60 条一条不丢
    assert n_proc == 60, f"并发写丢失: {n_proc}/60"


# ------------------------------------------------------------------ #
# L2: 丢更新防护
# ------------------------------------------------------------------ #
def test_lost_update_refused(tmp_path):
    state = str(tmp_path / "lu.json")
    a = _engine(state)
    a.add("A的记忆1")
    a.save(state)

    b = _engine(state)
    b.load(state)
    b.add("B的新记忆")
    b.save(state)                       # B 正常保存

    a.add("A的记忆2")
    with pytest.raises(RuntimeError, match="其他进程"):
        a.save(state)                   # A 再保存必须被拒

    # A 按错误提示行动:先记下自己未落盘的增量 → 重载(看到 B 的数据)→ 重放 → 保存
    pending = [m.text for m in a.memories.values()
               if m.text not in {"A的记忆1", "B的新记忆"}]
    a.load(state)
    assert any(m.text == "B的新记忆" for m in a.memories.values())
    for text in pending:               # 重放被拒保存时未落盘的增量
        a.add(text)
    a.add("A的记忆3")
    a.save(state)
    c = _engine(state)
    c.load(state)
    texts = {m.text for m in c.memories.values()}
    assert {"A的记忆1", "B的新记忆", "A的记忆2", "A的记忆3"} <= texts


# ------------------------------------------------------------------ #
# L3: 锁超时如实抛错
# ------------------------------------------------------------------ #
def test_lock_timeout_raises_honestly(tmp_path):
    import threading
    from sme.storage.xlock import xlock, CrossProcessLockTimeout
    target = str(tmp_path / "t.json")
    holder_ready = threading.Event()
    release = threading.Event()

    def holder():
        with xlock(target):
            holder_ready.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    assert holder_ready.wait(5)
    try:
        # 另一个线程(同进程跨线程)竞争同一路径:必须如实超时,不静默继续
        with pytest.raises(CrossProcessLockTimeout):
            with xlock(target, timeout=0.3):
                pass                    # pragma: no cover - 不应到达
    finally:
        release.set()
        t.join(5)


# ------------------------------------------------------------------ #
# L4: 导出路径不污染指纹
# ------------------------------------------------------------------ #
def test_export_path_does_not_pollute_stamp(tmp_path):
    state = str(tmp_path / "main.json")
    e = _engine(state)
    e.add("主存储记忆")
    e.save(state)
    export = str(tmp_path / "export.json")
    e.save(export)                      # 导出(非恢复路径)
    # 回来再存主存储:指纹未被导出污染,应正常成功
    e.add("第二条")
    e.save(state)
    e2 = _engine(state)
    e2.load(state)
    assert len(e2.memories) == 2
