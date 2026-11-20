# -*- coding: utf-8 -*-
"""
T3b 坐标下降精修:读已有 results.json 的最优点,围绕它做逐坐标线搜索。
用法: python gen_refine.py results/t3_random/results.json results/jobs/t3_refine.json [轮次标签]
"""
import json
import sys
from pathlib import Path

RANK_KEYS = ["semantic", "importance", "freshness", "weight", "decay", "hit_count", "recency", "region"]


def renorm_around(base_w, key, mult):
    w = dict(base_w)
    newv = round(min(max(w[key] * mult, 0.0), 1.0), 4)
    rest = round(1 - newv, 4)
    old_rest = round(1 - w[key], 4)
    if old_rest <= 0:
        return None
    scale = rest / old_rest
    out = {k: (newv if k == key else round(v * scale, 4)) for k, v in w.items()}
    return {f"ranking.{k}": v for k, v in out.items()}


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "r1"
    results = json.loads(src.read_text(encoding="utf-8"))
    best = max((r for r in results if r.get("status") == "ok" and r.get("overrides")), key=lambda r: r["composite"])
    ov = best["overrides"]
    base_w = {k: ov.get(f"ranking.{k}") for k in RANK_KEYS}
    missing = [k for k, v in base_w.items() if v is None]
    if missing:  # 最优点没动过 ranking → 用默认值打底
        defaults = dict(zip(RANK_KEYS, [0.40, 0.12, 0.10, 0.10, 0.08, 0.08, 0.06, 0.06]))
        for k in missing:
            base_w[k] = defaults[k]
    # 携带最优点携带的其他覆盖(fusion 等)
    other = {k: v for k, v in ov.items() if not k.startswith("ranking.")}

    jobs = [{"name": f"refine_{tag}_center", "overrides": {**{f'ranking.{k}': v for k, v in base_w.items()}, **other}}]
    for k in RANK_KEYS:
        for mult in (0.6, 0.8, 1.25, 1.6):
            nov = renorm_around(base_w, k, mult)
            if nov:
                jobs.append({"name": f"refine_{tag}_{k}_x{mult}", "overrides": {**nov, **other}})
    dst.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"center={best['name']} composite={best['composite']}")
    print(f"jobs={len(jobs)} -> {dst}")


if __name__ == "__main__":
    main()
