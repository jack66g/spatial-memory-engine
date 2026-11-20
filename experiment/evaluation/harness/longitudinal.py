# -*- coding: utf-8 -*-
"""
纵向实验:动力学信号的"收益"侧证据。
设计:同一数据集,把每条查询的 gold 记忆强化 3 次(模拟被反复使用),
对比 [默认配置] vs [动力学信号全关] 在强化前后的 gold 排名提升。
期望:若 hit_count/weight 信号有效,默认配置对"强化过的gold"排名提升应显著大于关动力学时。
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import mrr, hit_at_k  # noqa: E402

DS = HERE.parent.parent / "datasets" / "synth"


def load(name):
    mems = [json.loads(l) for l in (DS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [json.loads(l) for l in (DS / f"{name}.queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return mems, qs


def build(overrides, td):
    from sme import SpatialMemoryEngine, SMEConfig
    cfg = SMEConfig.from_dict(overrides) if overrides else SMEConfig()
    cfg.storage.autosave = False
    cfg.maintenance.background = False
    cfg.storage.path = str(td / "state.json")
    return SpatialMemoryEngine(cfg)


def evalq(engine, id_map, queries):
    rev = {v: k for k, v in id_map.items()}
    out = []
    for q in queries:
        hits = engine.search(q["text"], top_k=10)
        out.append({"ranked": [rev.get(h.memory.id, "?") for h in hits], "gold": q["gold"]})
    return out


def run(name="small"):
    mems, queries = load(name)
    queries = queries[:300]
    configs = {
        "default(动力学开)": {},
        "dyn_off(动力学关)": {"policy": {"decay_enabled": False},
                              "ranking": {"weight": 0.0, "hit_count": 0.0, "decay": 0.0, "freshness": 0.0, "recency": 0.0,
                                          "semantic": 0.40, "importance": 0.12, "region": 0.06}},
    }
    report = {}
    for label, ov in configs.items():
        wd = HERE.parent / "work" / ("longi_" + label.split("(")[0])
        wd.mkdir(parents=True, exist_ok=True)
        e = build(ov, wd)
        id_map = {}
        for m in mems:
            mm = e.add(m["text"], metadata={"synth_id": m["id"]}, importance=m.get("importance", 0.5))
            id_map[m["id"]] = mm.id
            o = e.memories[mm.id]
            o.created_at = m["ts"]; o.last_hit = m["ts"]
        before = evalq(e, id_map, queries)
        # 强化:每条查询的 gold 强化 3 次(隔虚拟 7 天)
        gold_engine_ids = {id_map[g] for q in queries for g in q["gold"]}
        DAY = 86400
        for rnd_i, offset in enumerate([3, 10, 30]):
            for gid in gold_engine_ids:
                m = e.memories[gid]
                m.last_hit = time.time() - offset * DAY  # 回拨 last_hit 再强化 → 模拟历史使用
                e.reinforce(gid)
        after = evalq(e, id_map, queries)
        mrr_b = sum(mrr(p["ranked"], p["gold"]) for p in before) / len(before)
        mrr_a = sum(mrr(p["ranked"], p["gold"]) for p in after) / len(after)
        h3_b = sum(hit_at_k(p["ranked"], p["gold"], 3) for p in before) / len(before)
        h3_a = sum(hit_at_k(p["ranked"], p["gold"], 3) for p in after) / len(after)
        report[label] = {"mrr_before": round(mrr_b, 4), "mrr_after": round(mrr_a, 4),
                         "mrr_gain": round(mrr_a - mrr_b, 4),
                         "hit@3_before": round(h3_b, 4), "hit@3_after": round(h3_a, 4),
                         "hit3_gain": round(h3_a - h3_b, 4)}
        print(label, json.dumps(report[label], ensure_ascii=False))
    (HERE.parent / "results" / "t4e_longitudinal.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    d = report["default(动力学开)"]; o = report["dyn_off(动力学关)"]
    print(f"\n结论: 强化收益(默认) mrr +{d['mrr_gain']} vs 动力学关 +{o['mrr_gain']}")
    print(f"      强化收益(默认) hit@3 +{d['hit3_gain']} vs 动力学关 +{o['hit3_gain']}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "small")
