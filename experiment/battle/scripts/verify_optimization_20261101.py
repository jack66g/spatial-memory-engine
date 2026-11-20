"""Isolated, reproducible validation of the 2026-11-01 optimization.

Examples (all output must be outside the production data directory):
  python -B lab/scripts/verify_optimization_20261101.py prepare --out <lab-dir>
  python -B lab/scripts/verify_optimization_20261101.py quality --out <lab-dir> --source <checkout>
  python -B lab/scripts/verify_optimization_20261101.py performance --out <lab-dir> --source <checkout>

The baseline must be the frozen dirty working tree, not just Git HEAD.
No cloud requests. Models are loaded from the existing offline cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

import numpy as np

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("task", choices=["prepare", "quality", "performance", "paired-performance", "controlled-performance", "stress"])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--label", default="current")
parser.add_argument("--baseline", type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.source.resolve()))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_PROXY="127.0.0.1,localhost")

from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine


def write(name, data):
    (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def engine(path, mode="off", backend="json", dim=64):
    c = SMEConfig()
    c.storage.path, c.storage.backend, c.storage.autosave = str(path), backend, False
    c.persistence.enabled = mode != "off"
    c.persistence.sync_mode = "fsync" if mode == "off" else mode
    c.region.auto_evolve = False
    c.embedding.dim = dim
    c.policy.decay_enabled = False
    return SpatialMemoryEngine(c)


CORRECTIONS = [
    ("city", "用户现在住在北京。", "纠正一下，用户现在搬到上海了。", "用户现在住在哪里？", "上海"),
    ("pet", "用户养的猫叫团子。", "纠正一下，用户的猫现在改名叫豆包。", "用户的猫现在叫什么名字？", "豆包"),
    ("drink", "用户喜欢每天喝咖啡。", "纠正一下，用户现在不喝咖啡了，改喝绿茶。", "用户现在喝什么饮料？", "绿茶"),
    ("job", "用户在海星科技公司上班。", "纠正一下，用户现在已经跳槽到晨光公司上班。", "用户现在在哪家公司工作？", "晨光"),
]


if args.task == "prepare":
    import torch
    from sme.modules.extraction import _rules_extract
    from sme.embedding.sentence_transformers_provider import SentenceTransformersProvider
    torch.set_num_threads(4)
    report = json.loads((args.source / "lab/results/healthcheck_2026-10-23.json").read_text(encoding="utf-8"))
    fixture = report["quality_fixture"]
    texts = {d["text"] for case in fixture for d in case["docs"]}
    texts |= {q["text"] for case in fixture for q in case["questions"]}
    for _, old, new, q, _ in CORRECTIONS:
        texts.update([old, new, q])
        for text in [old, new]:
            texts.update(f.text for f in _rules_extract(text))
    texts = sorted(texts)
    write("quality_fixture.json", fixture)
    write("quality_texts.json", texts)
    print(f"Offline encoding {len(texts)} frozen texts, CPU 4 threads", flush=True)
    provider = SentenceTransformersProvider("Qwen/Qwen3-Embedding-0.6B", batch_size=8)
    vectors = []
    start = time.perf_counter()
    for i in range(0, len(texts), 16):
        vectors.extend(provider.embed(texts[i:i+16]))
        print(f"Encoded {min(i+16,len(texts))}/{len(texts)}", flush=True)
    np.savez_compressed(args.out / "quality_vectors.npz", vectors=np.asarray(vectors))
    write("encoding.json", {"texts": len(texts), "seconds": time.perf_counter()-start,
                            "provider": provider.model_name, "threads": 4,
                            "vectors_sha256": hashlib.sha256((args.out / "quality_vectors.npz").read_bytes()).hexdigest()})


if args.task == "quality":
    fixture = json.loads((args.out / "quality_fixture.json").read_text(encoding="utf-8"))
    texts = json.loads((args.out / "quality_texts.json").read_text(encoding="utf-8"))
    vectors = dict(zip(texts, np.load(args.out / "quality_vectors.npz")["vectors"]))
    class Frozen:
        name = "sentence-transformers"
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        dim = effective_dim = 1024
        mrl_dim = 0
        query_prefix = ""
        def embed(self, texts): return [vectors[t] for t in texts]
        def embed_one(self, text): return vectors[text]
        def embed_query(self, text): return vectors[text]
    def frozen_engine(path):
        e = engine(path, dim=1024)
        e.embeddings = e.memory_manager.embeddings = Frozen()
        return e
    rows, corrections = [], []
    with tempfile.TemporaryDirectory(prefix="sme-quality-", dir=args.out) as tmp:
        for case in fixture:
            e = frozen_engine(Path(tmp)/"state.json")
            for doc in case["docs"]:
                e.memory_manager.add_memory(doc["text"], embedding=vectors[doc["text"]], memory_id=doc["id"])
            for q in case["questions"]:
                hits = e.search(q["text"], top_k=5)
                ids = [h.memory.id for h in hits]
                rows.append({"seed": case["seed"], **q, "ids": ids,
                             "rank": ids.index(q["gold"])+1 if q["gold"] in ids else None})
        for label, old, new, q, expected in CORRECTIONS:
            e = frozen_engine(Path(tmp)/(label+".json"))
            e.config.extraction.enabled = e.config.factversion.enabled = True
            e.config.extraction.mode = "rules"
            e.add(old); e.add(new)
            hits = e.search(q, top_k=5)
            corrections.append({"case": label, "top1_correct": bool(hits and expected in hits[0].memory.text),
                                "texts": [h.memory.text for h in hits],
                                "linked": any(m.metadata.get("superseded_by") for m in e.memories.values())})
    positives = [r for r in rows if r["gold"] is not None]
    write("quality_"+args.label+".json", {"rows": rows, "corrections": corrections,
        "summary": {"positive_count": len(positives), "top1": sum(r["rank"]==1 for r in positives),
                    "top5": sum(r["rank"] is not None for r in positives)}})
    print(args.label, "quality", sum(r["rank"]==1 for r in positives), "/", len(positives), corrections, flush=True)


if args.task == "performance":
    if os.environ.get("SME_BENCH_FIXED_IDS") == "1":
        import itertools
        import sme.models as models
        import sme.space.region_manager as region_manager
        id_counter = itertools.count()
        def deterministic_id(prefix="m"):
            return f"{prefix}_{next(id_counter):012x}"
        models.new_id = region_manager.new_id = deterministic_id
    results = []
    rng = np.random.default_rng(1147)
    with tempfile.TemporaryDirectory(prefix="sme-perf-", dir=args.out) as tmp:
        for size in [100, 1000, 10000]:
            vectors = rng.normal(size=(size+60,64)); vectors /= np.linalg.norm(vectors,axis=1)[:,None]
            for mode in ["off", "fsync", "grouped"]:
                path = Path(tmp)/(f"{size}-{mode}.json")
                e = engine(path, mode)
                e.config.region.min_join_cosine = 0
                e.space.set_bulk(True)
                for i in range(size):
                    e.memory_manager.add_memory(f"project {i} configuration storage", embedding=vectors[i])
                e.space.set_bulk(False)
                e.save()
                original_embed = e.embeddings.embed_one
                cached = {f"additional record {i}": original_embed(f"additional record {i}") for i in range(40)}
                cached["tail"] = original_embed("tail")
                e.embeddings.embed_one = lambda text: cached.get(text, vectors[0])
                times = {}
                for label, queries in [("cold_query", [f"query {i}" for i in range(30)]),
                                       ("cached_query", ["query 0"]*30)]:
                    samples = []
                    for q in queries:
                        start = time.perf_counter(); e.search(q, top_k=5)
                        samples.append((time.perf_counter()-start)*1000)
                    times[label] = {"p50_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples,95))}
                start = time.perf_counter()
                added = [e.add(f"additional record {i}") for i in range(40)]
                e.wal._flush_buffer()  # include grouped flush, not just buffering
                times["single_40_ms"] = (time.perf_counter()-start)*1000
                ids = list(e.memories)[:20]
                start = time.perf_counter()
                for mid in ids: e.reinforce(mid)
                e.wal._flush_buffer()
                times["reinforce_20_ms"] = (time.perf_counter()-start)*1000
                # Record the old durable alternative separately, not as the
                # headline baseline: old reinforcement + one checkpoint.
                start = time.perf_counter(); e.save()
                times["checkpoint_ms"] = (time.perf_counter()-start)*1000
                if mode != "off":
                    e.add("tail"); e.wal._flush_buffer(); e.wal.close()
                    start = time.perf_counter(); r = engine(path, mode); r.load()
                    times["recover_ms"] = (time.perf_counter()-start)*1000
                    r.wal.close()
                e.wal.close()
                results.append({"size": size, "mode": mode, **times})
                print("performance",args.label,size,mode,"complete",flush=True)
    write("performance_"+args.label+".json", results)
    print("performance", args.label, "complete", flush=True)


if args.task in ("paired-performance", "controlled-performance"):
    import subprocess
    if args.baseline is None:
        parser.error("paired-performance requires --baseline (frozen working tree)")
    for repeat in range(5):
        order = ["baseline", "current"] if repeat % 2 == 0 else ["current", "baseline"]
        for name in order:
            source = args.baseline if name == "baseline" else args.source
            label = ("controlled_" if args.task == "controlled-performance" else "") + f"{name}_{repeat}"
            subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"performance",
                            "--out",str(args.out),"--source",str(source),"--label",label],
                           check=True, timeout=240,
                           env={**os.environ,"OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4",
                                "PYTHONHASHSEED":"1147",
                                "SME_BENCH_FIXED_IDS":"1" if args.task == "controlled-performance" else "0"})


if args.task == "stress":
    results = []
    for seed in [101,202,303]:
        for backend in ["json","sqlite"]:
            for mode in ["fsync","grouped"]:
                with tempfile.TemporaryDirectory(prefix="sme-stress-", dir=args.out) as tmp:
                    path = Path(tmp)/("state.db" if backend=="sqlite" else "state.json")
                    e = engine(path, mode, backend)
                    rng = random.Random(seed)
                    checks = 0
                    for step in range(200):
                        mids = list(e.memories)
                        op = rng.choice(["add","update","reinforce","link","archive","restore","delete"])
                        if not mids or op=="add": e.add(f"{seed} step {step} fact", ns=rng.choice(["alice","bob"]))
                        elif op=="update": e.update(rng.choice(mids), importance=rng.random())
                        elif op=="reinforce": e.reinforce(rng.choice(mids))
                        elif op=="archive": e.archive(rng.choice(mids))
                        elif op=="restore": e.restore(rng.choice(mids))
                        elif op=="delete": e.delete(rng.choice(mids))
                        elif len(mids)>1: e.link(*rng.sample(mids,2), weight=rng.random())
                        if step%25==24:
                            expected = {mid:m.to_dict() for mid,m in e.memories.items()}
                            edges = e.graph.to_dict()
                            e.wal._flush_buffer(); e.wal.close()
                            r = engine(path, mode, backend); assert r.load()
                            assert set(r.memories)==set(expected)
                            for mid, before in expected.items():
                                after = r.get(mid).to_dict()
                                for key in before.keys()-{"region_id"}:
                                    assert before[key]==after[key], (seed,backend,mode,step,mid,key)
                            assert r.graph.to_dict()==edges
                            assert set(r.archive_manager.cold)=={m.id for m in r.memories.values() if m.archived}
                            e = r; checks += 1
                    e.wal.close()
                    results.append({"seed":seed,"backend":backend,"mode":mode,"steps":200,"checks":checks,"passed":True})
                    print("stress", seed, backend, mode, "PASS", flush=True)
    write("stress.json", results)
