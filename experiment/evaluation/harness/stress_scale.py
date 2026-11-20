# -*- coding: utf-8 -*-
"""
T4b 规模压力:1k/5k/20k/60k 梯度上测 写入吞吐、检索延迟曲线、检索质量衰变。
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import summarize  # noqa: E402

DS = HERE.parent.parent / "datasets" / "synth"
PY_TAGS = ["small", "medium", "large", "xl"]


def bench(name):
    from sme import SpatialMemoryEngine, SMEConfig
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.maintenance.background = False
    cfg.storage.path = str(HERE.parent / "work" / f"stress_{name}.json")
    e = SpatialMemoryEngine(cfg)

    mems = [json.loads(l) for l in (DS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [json.loads(l) for l in (DS / f"{name}.queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    t0 = time.perf_counter()
    id_map = {}
    for m in mems:
        mem = e.add(m["text"], metadata={"synth_id": m["id"]}, importance=m.get("importance", 0.5))
        id_map[m["id"]] = mem.id
        o = e.memories[mem.id]
        o.created_at = m["ts"]; o.last_hit = m["ts"]
    t_add = time.perf_counter() - t0
    rev = {v: k for k, v in id_map.items()}

    for q in qs[:3]:
        e.search(q["text"], top_k=10)

    per_query, lat = [], []
    for q in qs[:1000]:
        t1 = time.perf_counter()
        hits = e.search(q["text"], top_k=10)
        lat.append((time.perf_counter() - t1) * 1000)
        per_query.append({"ranked": [rev.get(h.memory.id, "?") for h in hits],
                          "gold": q["gold"], "type": q["type"]})

    lat.sort()
    rec = {
        "dataset": name, "n_mem": len(mems), "n_q": len(per_query),
        "add_ops_per_s": round(len(mems) / t_add, 1), "add_total_s": round(t_add, 1),
        **summarize(per_query, lat),
    }
    import gc
    del e
    gc.collect()
    return rec


if __name__ == "__main__":
    out = []
    for name in (sys.argv[1].split(",") if len(sys.argv) > 1 else PY_TAGS):
        print(f"\n=== 规模档位: {name} ===", flush=True)
        try:
            r = bench(name)
            out.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "by_type"}, ensure_ascii=False))
        except MemoryError:
            out.append({"dataset": name, "crash": "MemoryError"})
            print("MemoryError!")
            break
        except Exception as ex:
            out.append({"dataset": name, "crash": f"{type(ex).__name__}: {ex}"})
            print("CRASH:", ex)
    (HERE.parent / "results" / "t4b_scale.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsaved -> results/t4b_scale.json")
