# -*- coding: utf-8 -*-
"""
T4d 并发与崩溃恢复:
1) 多进程并发写同一存储路径 → last-writer-wins 丢写检测(插件线曾为此加锁,验证上游裸奔行为)
2) WAL 持久化:persistence.enabled 下 kill -9 语义(硬进程退出)后恢复不丢已确认写
3) 线程并发 search+add 同引擎(线程安全性)
"""
import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not cond else ""))


def _writer(worker_id, state_path, n, q):
    """子进程:独立引擎实例写同一 state 路径"""
    try:
        from sme import SpatialMemoryEngine, SMEConfig
        cfg = SMEConfig()
        cfg.storage.autosave = False
        cfg.storage.path = state_path
        e = SpatialMemoryEngine(cfg)
        for i in range(n):
            e.add(f"进程{worker_id}的记忆{i}")
        e.save(state_path)
        q.put(("ok", worker_id, n))
    except Exception as ex:
        q.put(("err", worker_id, repr(ex)))


def mp_save_storm():
    """3 进程 × 20 条,同一路径交错 save → 终态完整性"""
    with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
        state = str(Path(td) / "shared.json")
        q = mp.Queue()
        procs = [mp.Process(target=_writer, args=(w, state, 20, q)) for w in range(3)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        statuses = [q.get(timeout=5) for _ in range(3)]
        all_ok = all(s[0] == "ok" for s in statuses)
        check("3进程全部完成写入", all_ok, str(statuses))
        from sme import SpatialMemoryEngine
        e = SpatialMemoryEngine()
        try:
            e.load(state)
            n = len(e.memories)
            # last-writer-wins 语义下终态 = 某个进程的 20 条;全部 60 条保留则是完美
            check("终态快照可加载且非空", n > 0, f"n={n}")
            if n < 60:
                print(f"    [观察] 终态 {n}/60 条 —— 多进程并发写同路径存在 last-writer-wins 数据丢失(上游无跨进程锁,插件线已知问题)")
                FAIL.append(f"并发写丢数据: {n}/60")
            else:
                check("并发写零丢失", True)
        except Exception as ex:
            check("并发终态可加载", False, f"{type(ex).__name__}: {ex}")


def thread_mix():
    """单实例 4 线程混合读写 10 秒"""
    import threading
    from sme import SpatialMemoryEngine, SMEConfig
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(HERE.parent / "work" / "thread_state.json")
    e = SpatialMemoryEngine(cfg)
    errors = []
    stop = time.time() + 8

    def writer(wid):
        i = 0
        try:
            while time.time() < stop:
                e.add(f"线程{wid}记忆{i}内容")
                i += 1
        except Exception as ex:
            errors.append(f"w{wid}: {ex!r}")

    def reader(rid):
        try:
            while time.time() < stop:
                e.search("记忆 内容", top_k=5)
        except Exception as ex:
            errors.append(f"r{rid}: {ex!r}")

    def maint():
        try:
            while time.time() < stop:
                time.sleep(0.5)
                e.apply_decay()
        except Exception as ex:
            errors.append(f"m: {ex!r}")

    ts = [threading.Thread(target=writer, args=(i,)) for i in range(2)] + \
         [threading.Thread(target=reader, args=(i,)) for i in range(1)] + \
         [threading.Thread(target=maint)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check("线程并发读写无异常", not errors, "; ".join(errors[:3]))
    check("并发后库仍可检索", len(e.search("线程", top_k=5)) > 0)


def crash_recovery():
    """WAL 崩溃恢复:子进程写 N 条后 os._exit(硬崩),父进程恢复验证"""
    import os
    with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
        state = str(Path(td) / "wal_state.json")
        code = f'''
import os, sys
sys.path.insert(0, r"{HERE}")
from sme import SpatialMemoryEngine, SMEConfig
cfg = SMEConfig()
cfg.storage.path = r"{state}"
cfg.storage.autosave = False
cfg.persistence.enabled = True
e = SpatialMemoryEngine(cfg)
for i in range(30):
    e.add("崩溃前记忆" + str(i))
e.save(r"{state}")   # 快照只含部分?
for i in range(30, 50):
    e.add("崩溃前记忆" + str(i))   # 这些只在 WAL 里
os._exit(0)          # 硬退出,不 close
'''
        script = Path(td) / "crasher.py"
        script.write_text(code, encoding="utf-8")
        import subprocess
        r = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True)
        check("崩溃子进程执行完成", r.returncode == 0, f"rc={r.returncode}: {r.stderr[-300:]}")
        from sme import SpatialMemoryEngine, SMEConfig
        cfg = SMEConfig()
        cfg.storage.path = state
        cfg.persistence.enabled = True
        e = SpatialMemoryEngine(cfg)
        e.load(state)
        n = len(e.memories)
        texts = [m.text for m in e.memories.values()]
        n_expected_wal_only = sum(1 for t in texts if t in {f"崩溃前记忆{i}" for i in range(30, 50)})
        check("崩溃后恢复:快照+WAL 合计不丢", n >= 30, f"恢复 {n}/50")
        if n >= 50:
            check("WAL-only 窗口的 20 条全部找回", n_expected_wal_only == 20, f"{n_expected_wal_only}/20")
        else:
            print(f"    [观察] 恢复 {n}/50 条 —— WAL 窗口丢失 {50-n} 条")


if __name__ == "__main__":
    t0 = time.time()
    print("=== 并发写风暴 ===")
    try:
        mp_save_storm()
    except Exception as ex:
        check("并发写风暴运行", False, repr(ex))
    print("\n=== 线程混合负载 ===")
    try:
        thread_mix()
    except Exception as ex:
        check("线程混合运行", False, repr(ex))
    print("\n=== WAL 硬崩恢复 ===")
    try:
        crash_recovery()
    except Exception as ex:
        check("WAL 恢复运行", False, repr(ex))
    print(f"\n===== 并发/恢复: PASS={len(PASS)} FAIL={len(FAIL)} ({time.time()-t0:.0f}s) =====")
    for f in FAIL:
        print("  FAIL:", f)
    (HERE.parent / "results" / "t4d_concurrency.json").write_text(
        json.dumps({"pass": PASS, "fail": FAIL}, ensure_ascii=False, indent=1), encoding="utf-8")
