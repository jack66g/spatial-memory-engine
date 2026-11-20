# -*- coding: utf-8 -*-
"""
T3 权重寻优任务生成:
1) 随机搜索:8 排序权重 Dirichlet 邻域采样 + 全局探索,80 个
2) 定向候选:语义主导/时序增强/关键词主导等 6 个结构化假设
3) 融合三通道随机采样,30 个(仅 minmax 模式)
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "jobs"
OUT.mkdir(parents=True, exist_ok=True)

RANK_KEYS = ["semantic", "importance", "freshness", "weight", "decay", "hit_count", "recency", "region"]
RANK_DEFAULT = [0.40, 0.12, 0.10, 0.10, 0.08, 0.08, 0.06, 0.06]

jobs = []
rnd = random.Random(20261120)


def rank_ov(w):
    return {f"ranking.{k}": round(x, 4) for k, x in zip(RANK_KEYS, w)}


def dirichlet_around(center, conc):
    """以 center 为均值、浓度 conc(越小越散)的 Dirichlet 采样"""
    import math
    alpha = [max(c * conc, 0.01) for c in center]
    s = [rnd.gammavariate(a, 1.0) for a in alpha]
    t = sum(s)
    w = [x / t for x in s]
    # 5% 概率猛推某个坐标(探索极端)
    if rnd.random() < 0.08:
        i = rnd.randrange(8)
        w = [0.02] * 8
        w[i] = 1 - 0.02 * 7
        rnd.shuffle(w)
    return w


# ---- 1) 随机搜索:邻域(中浓度)×45 + 全局(低浓度)×35 ----
for i in range(45):
    jobs.append({"name": f"rs_rank_near_{i:02d}", "overrides": rank_ov(dirichlet_around(RANK_DEFAULT, 30))})
for i in range(35):
    jobs.append({"name": f"rs_rank_far_{i:02d}", "overrides": rank_ov(dirichlet_around(RANK_DEFAULT, 4))})

# ---- 2) 定向结构化假设 ----
structured = {
    "sem_dominant":   [0.60, 0.10, 0.08, 0.08, 0.06, 0.04, 0.02, 0.02],
    "temporal_boost": [0.30, 0.10, 0.20, 0.08, 0.10, 0.04, 0.14, 0.04],
    "pure_semantic":  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "fresh_heavy":    [0.35, 0.08, 0.30, 0.08, 0.08, 0.03, 0.05, 0.03],
    "usage_heavy":    [0.35, 0.10, 0.05, 0.15, 0.05, 0.15, 0.10, 0.05],
    "region_heavy":   [0.35, 0.08, 0.06, 0.06, 0.05, 0.04, 0.04, 0.32],
}
for name, w in structured.items():
    jobs.append({"name": f"struct_{name}", "overrides": rank_ov(w)})

# ---- 3) 融合三通道随机采样(minmax 下 metadata 被折入,扫 v/k 比)----
for i in range(30):
    v = round(rnd.uniform(0.1, 0.95), 3)
    k = round(rnd.uniform(0.05, 1 - v if 1 - v > 0.05 else 0.05), 3)
    jobs.append({"name": f"rs_fusion_{i:02d}", "overrides": {
        "retrieval.vector_weight": v, "retrieval.keyword_weight": k}})

jobs.insert(0, {"name": "t3_baseline", "overrides": {}})
print(f"total T3 jobs: {len(jobs)}")
(OUT / "t3_random.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved ->", OUT / "t3_random.json")
