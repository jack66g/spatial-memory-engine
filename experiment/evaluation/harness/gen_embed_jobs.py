# -*- coding: utf-8 -*-
"""真模型(bge-small-zh-v1.5)重跑融合/排序权重:任务生成。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "jobs"

PROVIDER = {"embedding.provider": "fastembed",
            "embedding.model": "BAAI/bge-small-zh-v1.5",
            "embedding.dim": 512}

jobs = []
jobs.append({"name": "fe_baseline", "overrides": dict(PROVIDER)})

# 融合网格:真向量应该翻身,不再像 hashing 那样被 BM25 压制
for v in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    for k in [0.0, 0.15, 0.3, 0.5]:
        if v == 0 and k == 0:
            continue
        ov = dict(PROVIDER)
        ov["retrieval.vector_weight"] = v
        ov["retrieval.keyword_weight"] = k
        jobs.append({"name": f"fe_fusion_v{v}_k{k}", "overrides": ov})

# 排序 OAT(基于融合网格之后的最优融合再精调,先粗扫)
RANK_DEFAULTS = {"semantic": 0.40, "importance": 0.12, "freshness": 0.10, "weight": 0.10,
                 "decay": 0.08, "hit_count": 0.08, "recency": 0.06, "region": 0.06}

def renorm(key, mult):
    d = dict(RANK_DEFAULTS)
    newv = round(d[key] * mult, 4)
    rest, old_rest = round(1 - newv, 4), round(1 - d[key], 4)
    scale = rest / old_rest
    out = {k: (newv if k == key else round(v * scale, 4)) for k, v in d.items()}
    return {f"ranking.{k}": v for k, v in out.items()}

for key in RANK_DEFAULTS:
    for mult in [0.0, 0.5, 2.0]:
        ov = dict(PROVIDER)
        ov.update(renorm(key, mult))
        jobs.append({"name": f"fe_rank_{key}_x{mult}", "overrides": ov})

# 动力学开关对照(静态 vs 默认)
ov = dict(PROVIDER); ov["policy.decay_enabled"] = False; ov["decay.max_weight_decay"] = 0.0
jobs.append({"name": "fe_dynoff", "overrides": ov})

# 旧冠军 UNITED 在真模型下是否还成立(纯关键词+纯语义)
ov = dict(PROVIDER)
ov.update({"policy.decay_enabled": False, "decay.max_weight_decay": 0.0,
           "retrieval.keyword_candidates": 20, "retrieval.top_regions": 1,
           "retrieval.vector_weight": 0.0, "retrieval.keyword_weight": 0.3,
           "ranking.semantic": 1.0, "ranking.importance": 0.0, "ranking.freshness": 0.0,
           "ranking.weight": 0.0, "ranking.decay": 0.0, "ranking.hit_count": 0.0,
           "ranking.recency": 0.0, "ranking.region": 0.0})
jobs.append({"name": "fe_united_old", "overrides": ov})

print(f"total: {len(jobs)}")
(OUT / "fe_small.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved ->", OUT / "fe_small.json")
