# -*- coding: utf-8 -*-
"""SME 压测套件（lab/scripts/stress.py）。

独立于 sme 仓库代码：只 import、不修改。四个子命令可单独跑：

    python lab/scripts/stress.py embed_profile [--n 1000]   # 1. 真实 embedding 性能画像
    python lab/scripts/stress.py rest_mix [--minutes 2]     # 2. REST 并发混合负载
    python lab/scripts/stress.py wal_overhead [--n 2000]    # 3. WAL 开/关写入开销对比
    python lab/scripts/stress.py soak [--cycles 30]         # 4. 长跑泄漏（RSS 跟踪）

统一输出 JSON 到 lab/results/stress_<name>.json（seed 固定可复现）。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

RESULTS = os.path.join(REPO, "lab", "results")
os.makedirs(RESULTS, exist_ok=True)

CORPUS = [
    "用户喜欢喝咖啡", "用户周末打篮球", "用户在公司上班", "用户的同事是张三",
    "用户住在北京", "用户养了一只猫", "用户在学习钢琴", "用户偏好深色主题",
    "项目采用 FastAPI 框架", "记忆引擎使用两阶段检索", "Region 会自动演化",
    "衰减只降命中概率不删除记忆", "问答对有有效期限制", "图谱支持多跳查询",
    "WAL 提供崩溃安全", "快照采用原子写入", "支持中文二元分词", "多用户靠命名空间隔离",
]
QUERIES = [
    "用户喜欢什么饮品", "用户的爱好是什么", "用户在哪里工作", "项目用什么技术栈",
    "记忆会被删除吗", "检索是怎么工作的", "中文支持如何", "多人怎么隔离",
]


def _rand_text(rng: random.Random, i: int) -> str:
    return f"{rng.choice(CORPUS)}（#{i}）"


def _save(name: str, data: dict) -> None:
    path = os.path.join(RESULTS, f"stress_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"→ {path}")


def cmd_embed_profile(n: int) -> None:
    """真实 embedding（Qwen3 vs hashing）写入/检索画像，进程内直测引擎。"""
    from sme import SpatialMemoryEngine
    from sme.config import SMEConfig

    out = {"n": n, "seed": 42, "providers": {}}
    for provider, model, dim in (
        ("hashing", None, 64),
        ("sentence-transformers", "Qwen/Qwen3-Embedding-0.6B", 1024),
    ):
        cfg = SMEConfig()
        cfg.storage.autosave = False
        cfg.storage.path = os.path.join(tempfile.mkdtemp(prefix="sme_st_"), "e.json.gz")
        cfg.embedding.provider = provider
        if model:
            cfg.embedding.model = model
            cfg.embedding.dim = dim
        rng = random.Random(42)
        eng = SpatialMemoryEngine(cfg)
        t0 = time.perf_counter()
        for i in range(n):
            eng.add(_rand_text(rng, i))
        t_write = time.perf_counter() - t0
        lat = []
        for q in QUERIES * (max(1, n // 200)):
            t1 = time.perf_counter()
            eng.search(q, top_k=5)
            lat.append((time.perf_counter() - t1) * 1000)
        out["providers"][provider] = {
            "writes_per_s": round(n / t_write, 1),
            "search_p50_ms": round(statistics.median(lat), 2),
            "search_p95_ms": round(sorted(lat)[int(len(lat) * 0.95) - 1], 2),
            "regions": eng.region_stats().count,
            "memories": len(eng.memories),
        }
        print(f"[{provider}] {n/t_write:.0f} 写/s, p50={statistics.median(lat):.1f}ms")
        del eng
        gc.collect()
    _save("embed_profile", out)


def cmd_rest_mix(minutes: float, base: str = "http://127.0.0.1:8760") -> None:
    """并发 REST 混合负载：N 读线程 + 1 写线程打常驻服务。"""
    import httpx

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    stop = threading.Event()
    stats = {"reads": 0, "read_fail": 0, "writes": 0, "write_fail": 0, "lat": []}
    lock = threading.Lock()

    def reader():
        while not stop.is_set():
            try:
                r = httpx.post(f"{base}/memories/search",
                               json={"text": random.choice(QUERIES), "top_k": 5},
                               timeout=60)
                with lock:
                    stats["reads"] += 1
                    stats["lat"].append(r.elapsed.total_seconds() * 1000)
            except Exception:
                with lock:
                    stats["read_fail"] += 1

    def writer():
        i = 0
        while not stop.is_set():
            try:
                r = httpx.post(f"{base}/memories",
                               json={"text": _rand_text(random, i), "importance": 0.5},
                               timeout=60)
                with lock:
                    stats["writes"] += 1 if r.status_code == 200 else 0
                    stats["write_fail"] += 0 if r.status_code == 200 else 1
            except Exception:
                with lock:
                    stats["write_fail"] += 1
            i += 1
            time.sleep(0.05)

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    threads.append(threading.Thread(target=writer, daemon=True))
    t0 = time.time()
    for t in threads:
        t.start()
    time.sleep(minutes * 60)
    stop.set()
    for t in threads:
        t.join(timeout=10)
    dur = time.time() - t0
    lat = sorted(stats["lat"]) or [0]
    out = {
        "duration_s": round(dur, 1),
        "reads": stats["reads"], "read_fail": stats["read_fail"],
        "writes": stats["writes"], "write_fail": stats["write_fail"],
        "read_rps": round(stats["reads"] / dur, 1),
        "read_p50_ms": round(lat[len(lat) // 2], 1),
        "read_p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    _save("rest_mix", out)


def cmd_wal_overhead(n: int) -> None:
    """WAL 开/关写入与保存开销对比（进程内）。"""
    from sme import SpatialMemoryEngine
    from sme.config import SMEConfig

    out = {"n": n, "seed": 42, "wal": {}}
    for enabled in (False, True):
        cfg = SMEConfig()
        cfg.storage.autosave = False
        cfg.persistence.enabled = enabled
        cfg.storage.path = os.path.join(tempfile.mkdtemp(prefix="sme_wal_"), "e.json.gz")
        rng = random.Random(42)
        eng = SpatialMemoryEngine(cfg)
        t0 = time.perf_counter()
        for i in range(n):
            eng.add(_rand_text(rng, i))
        t_write = time.perf_counter() - t0
        t1 = time.perf_counter()
        eng.save()
        t_save = time.perf_counter() - t1
        out["wal"][str(enabled).lower()] = {
            "writes_per_s": round(n / t_write, 1),
            "save_s": round(t_save, 3),
        }
        print(f"[wal={enabled}] {n/t_write:.0f} 写/s, save {t_save:.2f}s")
        del eng
        gc.collect()
    _save("wal_overhead", out)


def cmd_soak(cycles: int) -> None:
    """长跑泄漏：重复 写入+检索+保存+重载，跟踪 RSS。"""
    from sme import SpatialMemoryEngine
    from sme.config import SMEConfig

    import psutil

    proc = psutil.Process()
    cfg = SMEConfig()
    cfg.storage.autosave = False
    path = os.path.join(tempfile.mkdtemp(prefix="sme_soak_"), "e.json.gz")
    cfg.storage.path = path
    eng = SpatialMemoryEngine(cfg)
    out = {"cycles": cycles, "rss_mb": [], "note": "每轮: 50写+10搜+save+load"}
    for c in range(cycles):
        rng = random.Random(c)
        for i in range(50):
            eng.add(_rand_text(rng, c * 50 + i))
        for q in QUERIES:
            eng.search(q, top_k=5)
        eng.save()
        eng.load(path)
        gc.collect()
        out["rss_mb"].append(round(proc.memory_info().rss / 1e6, 1))
    out["rss_delta_mb"] = round(out["rss_mb"][-1] - out["rss_mb"][0], 1)
    out["memories"] = len(eng.memories)
    print(f"RSS: {out['rss_mb'][0]} → {out['rss_mb'][-1]} MB (Δ{out['rss_delta_mb']}MB), "
          f"记忆 {out['memories']} 条")
    _save("soak", out)


def main() -> None:
    p = argparse.ArgumentParser(description="SME 压测套件")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed_profile").add_argument("--n", type=int, default=1000)
    sub.add_parser("rest_mix").add_argument("--minutes", type=float, default=2.0)
    sub.add_parser("wal_overhead").add_argument("--n", type=int, default=2000)
    sub.add_parser("soak").add_argument("--cycles", type=int, default=30)
    a = p.parse_args()
    {"embed_profile": lambda: cmd_embed_profile(a.n),
     "rest_mix": lambda: cmd_rest_mix(a.minutes),
     "wal_overhead": lambda: cmd_wal_overhead(a.n),
     "soak": lambda: cmd_soak(a.cycles)}[a.cmd]()


if __name__ == "__main__":
    main()
