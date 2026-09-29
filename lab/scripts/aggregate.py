# -*- coding: utf-8 -*-
"""汇总 3 seed 对战结果 → 均值排名 + 分题型画像 + 工程指标。

    python lab/scripts/aggregate.py            # 读 battle_seed{1,2,3}.json
    python lab/scripts/aggregate.py --seeds 1,2
输出：终端排名表 + lab/results/battle_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TYPES = ("normal", "paraphrase", "correction", "noise")


def load(seeds: list[int]) -> dict:
    per_seed = {}
    for s in seeds:
        p = os.path.join(RESULTS, f"battle_seed{s}.json")
        if not os.path.exists(p):
            print(f"[skip] seed{s} 未完成: {p}")
            continue
        per_seed[s] = json.load(open(p, encoding="utf-8"))
    return per_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    seeds = [int(x) for x in ap.parse_args().seeds.split(",")]
    per_seed = load(seeds)
    if not per_seed:
        print("无可用 seed 结果")
        return

    names: list[str] = []
    for d in per_seed.values():
        for n, v in d["contestants"].items():
            if v.get("status") == "ok" and n not in names:
                names.append(n)

    agg = {}
    for n in names:
        rows = [d["contestants"][n] for d in per_seed.values()
                if d["contestants"].get(n, {}).get("status") == "ok"]
        if not rows:
            continue
        by_type = defaultdict(lambda: [0, 0])  # type -> [correct, total]
        for r in rows:
            for q in r["details"]:
                t = q["type"]
                by_type[t][0] += q["correct"]
                by_type[t][1] += 1
        agg[n] = {
            "seeds": len(rows),
            "acc_mean": round(statistics.mean(r["acc"] for r in rows), 4),
            "acc_range": [min(r["acc"] for r in rows), max(r["acc"] for r in rows)],
            "misled_total": sum(r["misled"] for r in rows),
            "storage_mean": round(statistics.mean(r["storage"] for r in rows), 1),
            "add_p50_ms_mean": round(statistics.mean(r["add_p50_ms"] for r in rows), 1),
            "retrieval_p50_ms_mean": round(statistics.mean(r["retrieval_p50_ms"] for r in rows), 1),
            "total_s_mean": round(statistics.mean(r["total_s"] for r in rows), 1),
            "by_type": {t: (round(c / tot, 3) if tot else None, c, tot) for t, (c, tot) in by_type.items()},
        }

    ranking = sorted(agg.items(), key=lambda kv: -kv[1]["acc_mean"])
    n_q = len(next(iter(per_seed.values()))["contestants"][ranking[0][0]]["details"])

    print("=" * 100)
    print(f"3-seed 均值排名（{len(per_seed)} seed × {n_q} 题，每选手数据点 {n_q*len(per_seed)}）")
    print("=" * 100)
    print(f"{'选手':<16}{'acc':>8}{'波动':>14}{'带偏':>6}{'存储':>8}{'入库p50':>10}{'检索p50':>10}{'总耗时':>9}")
    for n, a in ranking:
        lo, hi = a["acc_range"]
        print(f"{n:<16}{a['acc_mean']*100:>7.1f}%{f'{lo*100:.0f}-{hi*100:.0f}%':>14}"
              f"{a['misled_total']:>6}{a['storage_mean']:>8.0f}{a['add_p50_ms_mean']:>9.0f}ms"
              f"{a['retrieval_p50_ms_mean']:>9.0f}ms{a['total_s_mean']:>8.0f}s")
    print("-" * 100)
    print("分题型命中率（correct/total → rate）：")
    print(f"{'选手':<16}" + "".join(f"{t:>18}" for t in TYPES))
    for n, a in ranking:
        cells = []
        for t in TYPES:
            r, c, tot = a["by_type"].get(t, (None, 0, 0))
            cells.append(f"{c}/{tot}={r:.2f}" if r is not None else "—")
        print(f"{n:<16}" + "".join(f"{c:>18}" for c in cells))

    out = {"meta": {"seeds": seeds, "questions_per_seed": n_q}, "ranking": [n for n, _ in ranking], "agg": agg}
    path = os.path.join(RESULTS, "battle_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
