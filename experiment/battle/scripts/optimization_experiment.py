"""Offline, paired experiments against an untouched source checkout.

No LLM judge, credential, network request or production data is used. Synthetic
corpora have fixed seeds and explicit evidence labels. Frozen lab assets are
hashed, never regenerated. Qwen vectors are computed once and shared by both
arms; a separate timing experiment includes actual CPU query inference.

Example:
  python -B lab/scripts/optimization_experiment.py --baseline PATH_TO_OLD_SOURCE \
      --workdir TEMP_DIRECTORY --out lab/results/optimization_2026-10-23.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PREFIX = "Instruct: Given a question about a user's memories, retrieve the relevant facts that answer it.\nQuery: "
MODEL = "Qwen/Qwen3-Embedding-0.6B"


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def corpus():
    cases = []
    roles = ["登录鉴权", "账单核对", "照片上传", "短信通知", "日志归档", "库存同步",
             "搜索建议", "订单结算", "指标告警", "文件备份", "报表生成", "视频转码"]
    for seed in (11, 29, 47):
        rng = random.Random(seed)
        docs, questions = [], []
        for i in range(24):
            name = f"{rng.choice(['Cedar','Maple','Birch','Willow'])}-{seed}{i:02d}"
            label = f"fact-{i}"
            role = roles[i % len(roles)]
            if i < 12:
                text = f"{role}服务的项目代号是 {name}，错误码 ERR_QX{seed}{i:02d} 表示连接超时，重试次数设为 {i+2}。"
                qs = [f"ERR_QX{seed}{i:02d} 是什么意思？", f"{name} 出现连接问题时允许尝试几次？"]
            else:
                text = f"{role}工具 {name} 的配置文件是 config_{seed}_{i}.yaml，维护人是员工{seed}{i}，输出目录 /srv/{name}/logs。"
                qs = [f"config_{seed}_{i}.yaml 由谁维护？", f"{name} 的运行日志放在哪里？"]
            docs.append({"id": label, "text": text})
            questions.extend({"text": q, "gold": label, "type": "positive"} for q in qs)
        for i in range(120):
            text = f"随机生活记录 {seed}-{i}：{rng.choice(['午后散步','周末看电影','打扫房间','早餐面包','地铁通勤','公园拍照'])}，今天感觉{rng.choice(['平静','开心','忙碌'])}。"
            docs.append({"id": f"noise-{i}", "text": text})
        for q in ("我的护照号码是什么？", "我上个月的血压测量值是多少？", "我高中班主任叫什么名字？",
                  "我最喜欢哪位爵士乐歌手？", "我的银行卡开户行在哪里？", "我去年参加马拉松的成绩是多少？",
                  "我妹妹出生的日期是哪天？", "我家冰箱保修还有几个月？"):
            questions.append({"text": q, "gold": None, "type": "unknown"})
        cases.append({"name": f"synthetic-{seed}", "seed": seed, "docs": docs, "questions": questions})
    assets = REPO / "lab" / "assets"
    hashes = {}
    if (assets / "dialogue.json").exists() and (assets / "quiz.json").exists():
        dialogue = json.loads((assets / "dialogue.json").read_text(encoding="utf-8"))
        quiz = json.loads((assets / "quiz.json").read_text(encoding="utf-8"))
        if dialogue["seed"] != quiz["seed"]:
            raise ValueError("frozen asset seeds disagree")
        design = dialogue["design"]
        docs = [{"id": f["id"], "text": f["content"]} for f in design["facts"]]
        for c in design["chains"]:
            docs.extend([{"id": c["id"], "text": c["new"]}, {"id": "old-"+c["id"], "text": c["old"]}])
        cases.append({"name": f"canonical-frozen-{dialogue['seed']}", "seed": dialogue["seed"],
                      "docs": docs, "questions": [{"text": q["question"], "gold": q["source"], "type": q["type"]}
                                                  for q in quiz["questions"]]})
        for name in ("dialogue.json", "quiz.json"):
            hashes[name] = hashlib.sha256((assets/name).read_bytes()).hexdigest()
    return cases, hashes


def prepare(work):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    cases, hashes = corpus()
    texts = sorted({d["text"] for c in cases for d in c["docs"]}
                   | {q["text"] for c in cases for q in c["questions"]}
                   | {PREFIX+q["text"] for c in cases for q in c["questions"]})
    fixture = {"cases": cases, "texts": texts, "asset_sha256": hashes,
               "model": MODEL, "query_prefix": PREFIX, "torch_threads": 4}
    fixture["corpus_sha256"] = hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    write_json(work/"fixture.json", fixture)
    print(f"Embedding {len(texts)} frozen texts with local CPU Qwen (4 threads)", flush=True)
    model = SentenceTransformer(MODEL, device="cpu", local_files_only=True)
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=16,
                           show_progress_bar=False, convert_to_numpy=True)
    np.savez_compressed(work/"vectors.npz", vectors=vectors)
    print("Shared embedding fixture ready", flush=True)


def worker(args):
    sys.path.insert(0, str(Path(args.source).resolve()))
    from sme.config import SMEConfig
    from sme.engine import SpatialMemoryEngine
    import sme.retrieval.retriever as retrieval_module
    import logging

    logging.getLogger("sme").setLevel(logging.ERROR)
    work = Path(args.workdir)
    fixture = json.loads((work/"fixture.json").read_text(encoding="utf-8"))
    vectors = np.load(work/"vectors.npz")["vectors"]
    lookup = dict(zip(fixture["texts"], vectors))

    class FrozenProvider:
        name = "hashing"
        dim = 1024
        effective_dim = 1024
        mrl_dim = 0
        model_name = MODEL
        query_prefix = ""
        def embed(self, texts):
            return [np.asarray(lookup[text], dtype=np.float64) for text in texts]
        def embed_one(self, text):
            return self.embed([text])[0]
        def embed_query(self, text):
            return self.embed_one(self.query_prefix + text)

    def make(case, keyword=0, prefix=""):
        cfg = SMEConfig()
        cfg.embedding.dim = 1024
        cfg.storage.autosave = False
        cfg.storage.path = str(work/"unused-state.json")
        cfg.region.min_join_cosine = 0.55
        cfg.policy.decay_enabled = False
        if hasattr(cfg.retrieval, "keyword_candidates"):
            cfg.retrieval.keyword_candidates = keyword
        e = SpatialMemoryEngine(cfg)
        e.embeddings = FrozenProvider()
        e.embeddings.query_prefix = prefix
        for d in case["docs"]:
            m = e.memory_manager.add_memory(d["text"], memory_id=d["id"], embedding=lookup[d["text"]])
            m.created_at = m.last_hit = 1750000000.0
        return e

    retrieval_module.now = lambda: 1750000000.0
    digest = hashlib.sha256()
    for source_file in sorted((Path(args.source)/"sme").rglob("*.py")):
        digest.update(str(source_file.relative_to(args.source)).replace("\\", "/").encode())
        digest.update(source_file.read_bytes())
    result = {"source": str(Path(args.source).resolve()), "source_sha256": digest.hexdigest(), "variants": {}}
    variants = [("default", 0, "")]
    if args.optimized:
        variants += [("bm25_20", 20, ""), ("query_instruction", 0, PREFIX), ("bm25_and_instruction", 20, PREFIX)]
    for name, keyword, prefix in variants:
        cases = []
        for case in fixture["cases"]:
            e = make(case, keyword, prefix)
            details = []
            for q in case["questions"]:
                hits = e.search(q["text"], top_k=5)
                ids = [h.memory.id for h in hits]
                rank = ids.index(q["gold"])+1 if q["gold"] in ids else 0
                qv = lookup[prefix+q["text"]]
                similarities = [float(np.dot(qv, h.memory.embedding)) for h in hits]
                details.append({"question": q["text"], "gold": q["gold"], "ids": ids,
                                "rank": rank, "cosines": similarities})
            positive = [d for d in details if d["gold"] is not None]
            cases.append({"name": case["name"], "positive_n": len(positive),
                          "hit1": sum(d["rank"] == 1 for d in positive)/len(positive),
                          "hit5": sum(d["rank"] > 0 for d in positive)/len(positive),
                          "mrr5": sum(1/d["rank"] if d["rank"] else 0 for d in positive)/len(positive),
                          "details": details})
        result["variants"][name] = cases
    if args.timing:
        import torch
        from sme.embedding.sentence_transformers_provider import SentenceTransformersProvider

        torch.set_num_threads(4)
        case = fixture["cases"][0]
        e = make(case)
        e.embeddings = SentenceTransformersProvider(MODEL)
        queries = [q["text"] for q in case["questions"][:6]]
        calls = [0]
        original = e.embeddings.embed_one
        def counted(text):
            calls[0] += 1
            return original(text)
        e.embeddings.embed_one = counted
        for q in queries:
            e.search(q, top_k=5)
        before = calls[0]
        repeat_ms = []
        for _ in range(8):
            for q in queries:
                t = time.perf_counter()
                e.search(q, top_k=5)
                repeat_ms.append((time.perf_counter()-t)*1000)
        repeat_calls = calls[0]-before
        unique_ms = []
        for i in range(24):
            t = time.perf_counter()
            e.search(queries[i % len(queries)] + f" (request {i})", top_k=5)
            unique_ms.append((time.perf_counter()-t)*1000)
        result["timing"] = {"repeat_n": len(repeat_ms), "repeat_p50_ms": statistics.median(repeat_ms),
                            "repeat_p95_ms": float(np.percentile(repeat_ms,95)), "repeat_embedding_calls": repeat_calls,
                            "unique_n": len(unique_ms), "unique_p50_ms": statistics.median(unique_ms),
                            "unique_p95_ms": float(np.percentile(unique_ms,95)), "threads": 4}
        storage = {}
        for mode in ("fsync", "grouped"):
            runs = []
            for seed in (11, 29, 47):
                with tempfile.TemporaryDirectory(prefix="storage-", dir=work) as temporary:
                    cfg = SMEConfig()
                    cfg.storage.path = str(Path(temporary)/"state.json")
                    cfg.storage.autosave = False
                    cfg.persistence.enabled = True
                    cfg.persistence.sync_mode = mode
                    e = SpatialMemoryEngine(cfg)
                    t = time.perf_counter()
                    for i in range(300):
                        e.add(f"storage benchmark seed {seed} entry {i}")
                    e.wal.close()
                    write_s = time.perf_counter()-t
                    t = time.perf_counter()
                    e.save()
                    checkpoint_ms = (time.perf_counter()-t)*1000
                    e.wal.close()
                    runs.append({"seed": seed, "writes_per_s": 300/write_s,
                                 "checkpoint_ms": checkpoint_ms})
            storage[mode] = runs
        result["storage_timing"] = storage
    write_json(args.worker_out, result)


def compare(records):
    old = records["baseline"]["variants"]["default"]
    new = records["optimized"]["variants"]["default"]
    changes = []
    for before, after in zip(old, new):
        for b, a in zip(before["details"], after["details"]):
            if b["ids"] != a["ids"]:
                changes.append({"case": before["name"], "question": a["question"],
                                "before": b["ids"], "after": a["ids"]})
    rejection = {}
    for variant, cases in records["optimized"]["variants"].items():
        calibration = next(c for c in cases if c["name"] == "synthetic-11")
        def evaluate(case, floor):
            positives = unknowns = found = rejected = 0
            for q in case["details"]:
                retained = [mid for mid, cosine in zip(q["ids"], q["cosines"]) if cosine >= floor]
                if q["gold"] is None:
                    unknowns += 1
                    rejected += not retained
                else:
                    positives += 1
                    found += q["gold"] in retained
            return {"positive_n": positives, "hit5": found/max(1, positives),
                    "unknown_n": unknowns, "unknown_rejected": rejected}
        # Calibrate on seed 11 only. Prefer the lowest floor that maximizes
        # rejection without reducing evidence recall on this calibration set.
        candidates = []
        for floor in np.arange(0.20, 0.81, 0.02):
            metrics = evaluate(calibration, floor)
            if metrics["hit5"] >= calibration["hit5"]:
                candidates.append((metrics["unknown_rejected"], -float(floor), float(floor)))
        floor = round(max(candidates)[2], 2) if candidates else -1
        rejection[variant] = {"floor_calibrated_on": "synthetic-11", "floor": floor,
                              "results": [{"case": c["name"], **evaluate(c, floor)} for c in cases]}
    return {"default_top5_changed_queries": len(changes), "changes": changes,
            "rejection_calibration": rejection}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline")
    p.add_argument("--workdir", required=True)
    p.add_argument("--out")
    p.add_argument("--source")
    p.add_argument("--worker-out")
    p.add_argument("--optimized", action="store_true")
    p.add_argument("--timing", action="store_true")
    p.add_argument("--reuse-fixture", action="store_true", help="reuse the already frozen, hashed vectors")
    args = p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    if args.source:
        worker(args)
        return
    if not args.baseline or not args.out:
        p.error("--baseline and --out required")
    work = Path(args.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if not args.reuse_fixture:
        prepare(work)
    records = {}
    for name, source in (("baseline", args.baseline), ("optimized", str(REPO))):
        print(f"Running paired {name} quality + actual CPU timing", flush=True)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--source", source,
                   "--workdir", str(work), "--worker-out", str(work/f"{name}.json"), "--timing"]
        if name == "optimized":
            command.append("--optimized")
        subprocess.run(command, check=True, env={**os.environ, "PYTHONHASHSEED": "0"})
        records[name] = json.loads((work/f"{name}.json").read_text(encoding="utf-8"))
    fixture = json.loads((work/"fixture.json").read_text(encoding="utf-8"))
    output = {"protocol": {k:v for k,v in fixture.items() if k not in ("texts", "cases")},
              "boundary": "Evidence retrieval, not end-to-end QA; deterministic synthetic + frozen canonical facts; no LLM judge",
              **records}
    output["comparison"] = compare(records)
    write_json(args.out, output)
    print(f"Experiment saved: {args.out}", flush=True)
    for name in ("baseline", "optimized"):
        print(name, json.dumps(records[name].get("timing", {})), flush=True)
        for variant, cases in records[name]["variants"].items():
            print(variant, [(c["name"],round(c["hit1"],3),round(c["hit5"],3)) for c in cases], flush=True)


if __name__ == "__main__":
    main()
