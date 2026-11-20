# -*- coding: utf-8 -*-
"""
评测执行器(核心):数据集 + 配置覆盖 → 引擎 → 指标 JSON。
子进程运行(sweep.py 调度),崩溃不连坐。绝不写仓库目录,状态全落 --workdir。

用法:
  python runner.py --dataset small --config overrides.json --out result.json \
                   --workdir wd [--top-k 10] [--limit 200] [--hits-off]
overrides.json 格式:扁平注册表路径 → 值,如 {"ranking.semantic": 0.5, "retrieval.fusion": "weighted"}
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import summarize, by_type  # noqa: E402

DS_DIR = HERE.parent.parent / "datasets" / "synth"  # experiment/datasets/synth


def nest(flat: dict) -> dict:
    """{'ranking.semantic': 0.5} → {'ranking': {'semantic': 0.5}};类型按字面解析(JSON 已带类型)"""
    tree: dict = {}
    for path, val in flat.items():
        parts = path.split(".")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return tree


def load_dataset(name: str):
    mems = [json.loads(l) for l in (DS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [json.loads(l) for l in (DS_DIR / f"{name}.queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return mems, qs


def build_engine(overrides: dict, workdir: Path):
    from sme.config import SMEConfig
    cfg = SMEConfig.from_dict(nest(overrides))
    # 统一测试基座:关自动保存、状态钉在 workdir、关空闲整理后台线程
    cfg.storage.autosave = False
    cfg.storage.path = str(workdir / "engine_state.json")
    cfg.maintenance.background = False
    return __import__("sme").SpatialMemoryEngine(cfg)


def populate(engine, mems):
    """写入全部记忆;返回 synth_id → engine_id 映射;注入 dataset 时间戳与 last_hit"""
    id_map = {}
    for m in mems:
        mem = engine.add(m["text"], metadata={"synth_id": m["id"]}, importance=m.get("importance", 0.5))
        id_map[m["id"]] = mem.id
        # 时间控制:created_at 决定 decay/freshness,last_hit 决定 recency
        obj = engine.memories[mem.id]
        obj.created_at = m["ts"]
        obj.last_hit = m["ts"]
    return id_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="small")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 条查询(快速迭代)")
    ap.add_argument("--hits-off", action="store_true", help="查询后把 hit_count/last_hit/weight 回滚,完全消除检索副作用")
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}

    t0 = time.time()
    try:
        mems, queries = load_dataset(args.dataset)
        if args.limit:
            queries = queries[: args.limit]
        engine = build_engine(overrides, wd)
        id_map = populate(engine, mems)
        rev = {v: k for k, v in id_map.items()}
        n_mem = len(engine.memories)

        # 预热(建索引/缓存,不计时):3 条真实查询
        for q in queries[:3]:
            engine.search(q["text"], top_k=args.top_k)

        per_query, lat = [], []
        for q in queries:
            if args.hits_off:
                snap = {mid: (mm.hit_count, mm.last_hit, mm.weight, mm.importance)
                        for mid, mm in engine.memories.items()}
            t1 = time.perf_counter()
            hits = engine.search(q["text"], top_k=args.top_k)
            lat.append((time.perf_counter() - t1) * 1000)
            ranked = [rev.get(h.memory.id, "?") for h in hits]
            per_query.append({"qid": q["qid"], "ranked": ranked, "gold": q["gold"], "type": q["type"],
                              "scores": [round(h.score, 4) for h in hits]})
            if args.hits_off:
                for mid, (hc, lh, w, imp) in snap.items():
                    mm = engine.memories[mid]
                    mm.hit_count, mm.last_hit, mm.weight, mm.importance = hc, lh, w, imp

        result = {
            "dataset": args.dataset, "n_memories": n_mem, "top_k": args.top_k,
            "overrides": overrides, "build_s": round(time.time() - t0, 1),
            "metrics": summarize(per_query, lat), "by_type": by_type(per_query, lat),
        }
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print("OK", result["metrics"]["composite"])
    except Exception:
        err = traceback.format_exc()
        (wd / "error.log").write_text(err, encoding="utf-8")
        print("FAIL:", err[-500:])
        sys.exit(1)


if __name__ == "__main__":
    main()
