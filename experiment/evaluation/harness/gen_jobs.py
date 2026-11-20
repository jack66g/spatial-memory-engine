# -*- coding: utf-8 -*-
"""T2 扫参任务生成器:输出 jobs JSON 给 sweep.py。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "jobs"
OUT.mkdir(parents=True, exist_ok=True)

RANK_DEFAULTS = {"semantic": 0.40, "importance": 0.12, "freshness": 0.10, "weight": 0.10,
                 "decay": 0.08, "hit_count": 0.08, "recency": 0.06, "region": 0.06}

def renorm(changed_key, mult):
    """改一个权重 ×mult,其余按默认比例归一,总和恒 1"""
    d = dict(RANK_DEFAULTS)
    newv = round(d[changed_key] * mult, 4)
    rest = round(1 - newv, 4)
    old_rest = round(1 - d[changed_key], 4)
    scale = rest / old_rest
    out = {k: (newv if k == changed_key else round(v * scale, 4)) for k, v in d.items()}
    return {f"ranking.{k}": v for k, v in out.items()}

jobs = []

# ---- 战役1:8 排序权重单坐标 ×6 乘数(0.25~2.0;0 是极端档单独加)----
for key in RANK_DEFAULTS:
    for mult in [0.25, 0.5, 0.75, 1.5, 2.0]:
        jobs.append({"name": f"rank_{key}_x{mult}", "overrides": renorm(key, mult)})
    jobs.append({"name": f"rank_{key}_zero", "overrides": renorm(key, 0)})

# ---- 战役2:三通道融合权重(vector/keyword/metadata;metadata 在 minmax 下被丢弃并入前两者)----
for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for k in [0.0, 0.1, 0.3, 0.5, 0.7]:
        if v == 0 and k == 0:
            continue  # 全零没意义
        jobs.append({"name": f"fusion_v{v}_k{k}", "overrides": {
            "retrieval.vector_weight": v, "retrieval.keyword_weight": k, "retrieval.metadata_weight": round(1 - v - k, 4) if v + k <= 1 else 0}})
for mw in [0.0, 0.3, 0.5]:
    jobs.append({"name": f"fusion_meta{mw}_weighted", "overrides": {
        "retrieval.vector_weight": 0.6, "retrieval.keyword_weight": 0.3, "retrieval.metadata_weight": mw,
        "retrieval.fusion": "weighted"}})
jobs.append({"name": "fusion_weighted_default", "overrides": {"retrieval.fusion": "weighted"}})

# ---- 战役3:检索旋钮 ----
for v in [1, 2, 5, 8]:
    jobs.append({"name": f"top_regions_{v}", "overrides": {"retrieval.top_regions": v}})
for v in [4, 50, 100, 200]:
    jobs.append({"name": f"cand_win_{v}", "overrides": {"retrieval.candidate_window": v}})
for v in [20, 50, 100]:
    jobs.append({"name": f"kw_cand_{v}", "overrides": {"retrieval.keyword_candidates": v}})
for v in [0.0, 0.3, 0.5]:
    jobs.append({"name": f"region_damp_{v}", "overrides": {"retrieval.region_dampening": v}})
for v in [-0.5, 0.0, 0.1, 0.2, 0.3]:
    jobs.append({"name": f"min_sim_{v}", "overrides": {"retrieval.min_similarity": v}})
jobs.append({"name": "cjk_bigram_off", "overrides": {"retrieval.cjk_bigram": False}})
for v in [0.0, 0.25, 0.75, 1.0]:
    jobs.append({"name": f"summary_pen_{v}", "overrides": {"retrieval.summary_penalty": v}})
for v in [5, 20, 50]:
    jobs.append({"name": f"query_cache_{v}", "overrides": {"retrieval.query_cache_size": v}})

# ---- 战役4:布尔消融(20 项逐一翻转)----
BOOLS = ["memory.reinforce_on", "policy.decay_enabled", "policy.full_memory", "region.auto_evolve",
         "region.ann_enabled", "retrieval.cjk_bigram", "storage.autosave", "storage.compress",
         "maintenance.background", "extraction.enabled", "factversion.enabled", "noise.enabled",
         "qapair.enabled", "factgraph.enabled", "profile.enabled", "persistence.enabled",
         "observability.enabled", "context.enabled", "namespaces.enabled", "rerank.enabled"]
DEFAULT_ON = {"memory.reinforce_on", "policy.decay_enabled", "policy.full_memory", "region.auto_evolve",
              "region.ann_enabled", "retrieval.cjk_bigram", "storage.autosave", "storage.compress"}
for b in BOOLS:
    flip = b not in DEFAULT_ON  # 默认开的测关,默认关的测开
    jobs.append({"name": f"bool_{b.replace('.', '_')}_{'off' if b in DEFAULT_ON else 'on'}",
                 "overrides": {b: flip}})

# ---- 战役5:动力学参数(影响 importance/decay/weight 信号的实际取值)----
for v in [7, 90, 365]:
    jobs.append({"name": f"decay_halflife_{v}", "overrides": {"decay.half_life_days": v}})
for v in [0.0, 0.5, 1.0]:
    jobs.append({"name": f"max_wdecay_{v}", "overrides": {"decay.max_weight_decay": v}})
for v in [1, 30, 90]:
    jobs.append({"name": f"ebb_halflife_{v}", "overrides": {"reinforcement.ebbinghaus_half_life_days": v}})
for v in [0.0, 0.5, 0.75]:
    jobs.append({"name": f"imp_thresh_{v}", "overrides": {"policy.importance_threshold": v}})
for v in [0.05, 0.2, 0.4]:
    jobs.append({"name": f"hit_delta_{v}", "overrides": {"reinforcement.hit_weight_delta": v}})

# ---- 基线 ----
jobs.insert(0, {"name": "baseline_default", "overrides": {}})

print(f"total jobs: {len(jobs)}")
(OUT / "t2_all.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved ->", OUT / "t2_all.json")
