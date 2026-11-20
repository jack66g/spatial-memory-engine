"""Isolated correction experiment. No product source changes or production writes.

Stages: plan -> extract -> encode -> run -> crash -> report.
LLM responses and real embedding vectors are frozen before paired comparisons.
No generated answers are scored. Runtime wrappers only observe existing methods.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent.parent
RESULTS = EXP / "results"
TMP = EXP / "temporary"
RESULTS.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
for key, value in {
    "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4", "NO_PROXY": "127.0.0.1,localhost",
    "no_proxy": "127.0.0.1,localhost", "TEMP": str(TMP), "TMP": str(TMP),
}.items():
    os.environ[key] = value
logging.basicConfig(level=logging.ERROR)


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def write(name, data):
    target = RESULTS / name
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(target)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def turn(text, ns="person_a", conv="session_1"):
    return {"text": text, "ns": ns, "conversation_id": conv}


def target(index, *tokens, ns=None, absent=()):
    obj = {"turn": index, "all": list(tokens), "absent": list(absent)}
    if ns is not None:
        obj["ns"] = ns
    return obj


def question(text, gold, ns="person_a", history=False):
    return {"text": text, "gold": gold, "ns": ns, "include_history": history}


def make_plan():
    cases = []

    def core(cid, title, old, new, query, old_tokens, new_tokens):
        cases.append({"id": cid, "category": "explicit", "title": title,
                      "turns": [turn(old), turn(new, conv="session_2")],
                      "queries": [question(query, target(1, *new_tokens))],
                      "retire": [{"old": target(0, *old_tokens), "new": target(1, *new_tokens)}],
                      "protect": []})

    core("C01", "住址变更", "用户现在住在北京。", "纠正一下，用户现在搬到上海了。", "用户现在住在哪里？", ["北京"], ["上海"])
    core("C02", "宠物改名", "用户养的猫叫团子。", "纠正一下，用户的猫现在改名叫豆包。", "用户的猫现在叫什么名字？", ["团子"], ["豆包"])
    core("C03", "饮料偏好变更", "用户喜欢每天喝咖啡。", "纠正一下，用户现在不喝咖啡了，改喝绿茶。", "用户现在喝什么饮料？", ["咖啡"], ["绿茶"])
    core("C04", "工作单位变更", "用户在海星科技公司上班。", "纠正一下，用户现在已经跳槽到晨光公司上班。", "用户现在在哪家公司工作？", ["海星"], ["晨光"])
    core("C05", "颜色偏好", "用户最喜欢的颜色是蓝色。", "更正一下，用户最喜欢的颜色是紫色，不是蓝色。", "用户最喜欢什么颜色？", ["蓝色"], ["紫色"])
    core("C06", "配置文件变更", "Atlas 服务的生产配置文件是 atlas_v1.yaml。", "更正一下，Atlas 服务的生产配置文件已经改为 atlas_v2.yaml。", "Atlas 服务现在使用哪个生产配置文件？", ["atlas_v1.yaml"], ["atlas_v2.yaml"])
    core("C07", "日志路径变更", "Atlas 服务的日志存放在 /var/log/atlas_old。", "纠正一下，Atlas 服务的日志现在存放在 /srv/atlas/logs。", "Atlas 服务现在的日志路径是什么？", ["/var/log/atlas_old"], ["/srv/atlas/logs"])
    core("C08", "租金数字更正", "用户的房租每月是2300元。", "记错了，用户的房租每月是1900元，不是2300元。", "用户现在每月房租是多少？", ["2300"], ["1900"])
    core("C09", "同词否定", "用户喜欢喝咖啡。", "其实用户不喜欢喝咖啡了。", "用户现在还喜欢喝咖啡吗？", ["咖啡"], ["不喜欢", "咖啡"])
    core("C10", "端口数字更正", "Atlas 服务的监听端口是8080。", "更正一下，Atlas 服务的监听端口是9090，不是8080。", "Atlas 服务现在监听哪个端口？", ["8080"], ["9090"])
    core("C11", "礼貌前缀下的明确更正", "用户现在住在北京。", "好的，纠正一下，用户现在住在上海。", "用户现在住在哪里？", ["北京"], ["上海"])
    core("C12", "明确否定旧宠物归属", "用户养了一只猫叫宽粉。", "纠正一下，宽粉是用户室友的猫，用户自己养的是柯基锅盔。", "用户自己养的宠物叫什么？", ["宽粉"], ["锅盔"])

    def add(cid, category, title, turns, queries, protect=(), retire=()):
        cases.append({"id": cid, "category": category, "title": title, "turns": turns,
                      "queries": queries, "protect": list(protect), "retire": list(retire)})

    add("G01", "safety", "朋友改口不应使本人偏好失效", [turn("用户每天喜欢喝咖啡。"), turn("更正一下，用户的朋友小周现在不喝咖啡了，改喝绿茶。")], [question("用户本人每天喜欢喝什么？", target(0, "咖啡"))], [target(0, "咖啡")])
    add("G02", "safety", "猫狗名字是不同属性", [turn("用户养的猫叫团子。"), turn("纠正一下，用户养的狗叫豆包。")], [question("用户的猫叫什么？", target(0, "猫", "团子"))], [target(0, "猫", "团子")])
    add("G03", "safety", "同次写入多个字段只更正住址", [turn("用户住在北京，用户在海星科技公司工作。"), turn("纠正一下，用户现在住在上海，工作单位没有变化。")], [question("用户现在在哪家公司工作？", target(0, "海星")), question("用户现在住在哪里？", target(1, "上海"))], [target(0, "海星")])
    add("G04", "safety", "同名服务不同环境", [turn("Atlas 生产环境的端口是8080。"), turn("纠正一下，Atlas 测试环境的端口是9090。")], [question("Atlas 生产环境端口是多少？", target(0, "生产", "8080"))], [target(0, "生产", "8080")])
    add("G05", "safety", "其实引出补充而非替换", [turn("用户喜欢喝咖啡。"), turn("其实用户也喜欢喝绿茶，咖啡仍然喜欢。")], [question("用户还喜欢喝咖啡吗？", target(0, "喜欢", "咖啡"))], [target(0, "喜欢", "咖啡")])
    add("G06", "safety", "条件句不应变成已发生搬家", [turn("用户现在住在北京。"), turn("如果用户以后搬到上海，就会换工作，目前还没有搬家。")], [question("用户目前住在哪里？", target(0, "北京"))], [target(0, "北京")])
    add("G07", "safety", "引述室友的话不应改变本人住址", [turn("用户现在住在北京。"), turn("用户转述室友小周的话：纠正一下，我现在搬到上海了。这句话说的是小周，不是用户。")], [question("用户本人现在住在哪里？", target(0, "北京"))], [target(0, "北京")])
    add("G08", "safety", "同人不同属性不应互相覆盖", [turn("用户喜欢喝咖啡。"), turn("更正一下，用户喜欢的咖啡杯颜色是绿色。")], [question("用户喜欢喝什么饮料？", target(0, "喝", "咖啡"))], [target(0, "喝", "咖啡")])

    add("T01", "temporal", "晚写入的历史不覆盖现在", [turn("截至2026年11月20日，用户现在住在上海。"), turn("补充历史记录：用户在2025年住在北京，这不是当前住址。", conv="session_2")], [question("用户现在住在哪里？", target(0, "上海")), question("用户2025年住在哪里？", target(1, "2025", "北京"), history=True)], [target(0, "上海")])
    add("T02", "temporal", "未来计划不覆盖当前住址", [turn("用户目前住在北京。"), turn("更正一下搬家计划：用户计划2027年搬去上海，目前还没有搬。")], [question("用户目前住在哪里？", target(0, "北京"))], [target(0, "北京")])
    add("T03", "temporal", "出差不等于迁居", [turn("用户长期住在北京。"), turn("其实用户这周在上海出差，周五回北京，并没有搬家。")], [question("用户长期住在哪里？", target(0, "长期", "北京"))], [target(0, "长期", "北京")])
    add("T04", "temporal", "明确纠正后保留历史", [turn("用户现在住在北京。"), turn("纠正一下，用户现在已经搬到上海了。", conv="session_2")], [question("用户现在住在哪里？", target(1, "上海")), question("用户搬去上海之前住在哪里？", target(0, "北京"), history=True)], retire=[{"old": target(0, "北京"), "new": target(1, "上海")}])

    add("S01", "scope", "用户B纠正不得使用户A失效", [turn("用户现在住在北京。", ns="person_a"), turn("纠正一下，用户现在搬到上海了。", ns="person_b")], [question("用户现在住在哪里？", target(0, "北京"), ns="person_a"), question("用户现在住在哪里？", target(1, "上海"), ns="person_b")], [target(0, "北京")])
    add("S02", "scope", "用户B否定不得使用户A偏好失效", [turn("用户喜欢喝咖啡。", ns="person_a"), turn("其实用户不喜欢喝咖啡了。", ns="person_b")], [question("用户喜欢喝什么？", target(0, "喜欢", "咖啡"), ns="person_a")], [target(0, "喜欢", "咖啡")])
    add("S03", "scope", "同用户跨会话明确纠正", [turn("用户养的猫叫团子。", conv="old_session"), turn("纠正一下，用户的猫现在改名叫豆包。", conv="new_session")], [question("用户的猫现在叫什么？", target(1, "豆包"))], retire=[{"old": target(0, "团子"), "new": target(1, "豆包")}])

    add("A01", "context", "省略主语的简短纠正", [turn("用户现在住在北京。"), turn("不对，是上海。", conv="session_2")], [question("用户现在住在哪里？", target(1, "上海"))], retire=[{"old": target(0, "北京"), "new": target(1, "上海")}])
    add("A02", "context", "两个同名人下指代不明确", [turn("研发部张伟负责 Atlas 服务。"), turn("财务部张伟负责报销系统。"), turn("更正一下，张伟已经调到市场部了。", conv="session_2")], [], [target(0, "研发", "张伟"), target(1, "财务", "张伟")])
    add("A03", "context", "连续三次明确改口", [turn("用户喜欢喝咖啡。"), turn("其实用户不喜欢喝咖啡了。"), turn("其实用户又喜欢喝咖啡了。", conv="session_3")], [question("用户现在还喜欢喝咖啡吗？", target(2, "又喜欢", "咖啡"))], retire=[{"old": target(0, "咖啡"), "new": target(1, "不喜欢")}, {"old": target(1, "不喜欢"), "new": target(2, "又喜欢")}])

    filler = "这段是项目运行记录，记录普通状态，不包含住址更新。" * 35
    add("L01", "boundary", "纠正在600字符之后", [turn("用户现在住在北京。"), turn(filler + "纠正一下，用户现在住在上海。")], [question("用户现在住在哪里？", target(1, "上海"))], retire=[{"old": target(0, "北京"), "new": target(1, "上海")}])
    add("L02", "boundary", "一条消息同时更正和补充多个字段", [turn("用户住在北京。"), turn("用户在海星科技公司工作。"), turn("纠正一下，用户住在上海；另外用户的朋友小周在晨光公司工作，用户自己的公司没有变化。")], [question("用户本人在哪家公司工作？", target(1, "海星")), question("用户现在住在哪里？", target(2, "上海"))], [target(1, "海星")], [{"old": target(0, "北京"), "new": target(2, "上海")}])
    assert len(cases) == 32
    return {"date": "2026-11-20", "head": "ed5378a", "cases": cases,
            "scoring": {"query": "top-1/top-5 gold = input turn provenance AND frozen literal anchors; no answer generation or LLM judge", "link": "each matching old fact must reach an appropriate successor in same namespace (chain allowed)", "protect": "matching non-raw evidence from protected turn must exist and remain unretired", "layers": "explicit cases scored separately from ambiguous/boundary cases; no universal accuracy claim", "embedding": "hashing and locally cached Qwen3-Embedding-0.6B; real vectors frozen for paired runs, not a latency benchmark", "arms": ["raw", "rules_no_version", "rules_version", "llm_no_version", "llm_version"], "llm": "existing laboratory DeepSeek configuration; synthetic inputs only; raw completions frozen and replayed identically; rules fallback tracked", "time": "separate conversation IDs, explicit dates; no claim of real ten-day wait", "production": "never called or written", "defaults": "version thresholds unchanged; decay disabled and background maintenance off to isolate correction"}}


def load_secrets():
    from dotenv import dotenv_values
    from sme.config import LLMConfig
    env = dotenv_values(ROOT / "lab" / ".env")
    return LLMConfig(base_url=env.get("LAB_LLM_BASE_URL", ""), api_key=env.get("LAB_LLM_API_KEY", ""), model=env.get("LAB_LLM_MODEL", ""), timeout=35, reasoning_effort="none")


def extract():
    from sme.modules.extraction import EXTRACTION_PROMPT, parse_llm_facts
    from sme.llm.client import LLMClient
    from urllib.parse import urlparse
    cache = read("llm_responses.json") if (RESULTS / "llm_responses.json").exists() else {}
    client = LLMClient(load_secrets())
    cases = read("plan.json")["cases"]
    if (RESULTS / "natural_language_plan.json").exists():
        cases += read("natural_language_plan.json")["cases"]
    if (RESULTS / "targeted_safety_plan.json").exists():
        cases += read("targeted_safety_plan.json")["cases"]
    texts = sorted({t["text"] for c in cases for t in c["turns"]})
    consecutive_failures = 0
    for i, text in enumerate(texts):
        prompt = EXTRACTION_PROMPT.format(text=text[:600])
        key = digest(prompt)
        if key in cache:
            continue
        start = time.perf_counter()
        try:
            raw = client.chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=600)
            facts = parse_llm_facts(raw)
            cache[key] = {"input": text, "prompt": prompt, "raw": raw,
                          "facts": [vars(f) for f in facts], "usage": dict(client.last_usage),
                          "seconds": time.perf_counter() - start, "status": "ok"}
            consecutive_failures = 0
        except Exception as exc:
            cache[key] = {"input": text, "prompt": prompt, "status": "error", "error_type": type(exc).__name__, "seconds": time.perf_counter() - start}
            consecutive_failures += 1
        write("llm_responses.json", cache)
        print(f"extract {i+1}/{len(texts)} {cache[key]['status']} facts={len(cache[key].get('facts', []))}", flush=True)
        if consecutive_failures >= 3:
            print("Three consecutive endpoint failures; preserving evidence and continuing offline arms.", flush=True)
            break
    write("llm_metadata.json", {"model": client.config.model, "host": urlparse(client.base_url).hostname,
                                "responses": len(cache), "successful": sum(v["status"] == "ok" for v in cache.values()),
                                "usage_this_process": client.total_usage,
                                "total_frozen_usage": {k: sum(row.get("usage", {}).get(k, 0) for row in cache.values()) for k in ["prompt_tokens", "completion_tokens", "total_tokens"]},
                                "prompt_sha256": digest(EXTRACTION_PROMPT),
                                "secrets_saved": False})


def all_texts():
    from sme.modules.extraction import _rules_extract
    texts = set()
    for c in read("plan.json")["cases"]:
        for t in c["turns"]:
            texts.add(t["text"])
            texts.update(f.text for f in _rules_extract(t["text"]))
        texts.update(q["text"] for q in c["queries"])
    for row in read("llm_responses.json").values():
        texts.update(f["text"] for f in row.get("facts", []))
    return sorted(texts)


def encode():
    import numpy as np
    import torch
    from sme.embedding.sentence_transformers_provider import SentenceTransformersProvider
    torch.set_num_threads(4)
    texts = all_texts()
    write("embedding_texts.json", texts)
    cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B"
    ref = (cache / "refs/main").read_text().strip()
    model_path = cache / "snapshots" / ref
    print("Loading existing local Qwen model; downloads disabled.", flush=True)
    provider = SentenceTransformersProvider(str(model_path), batch_size=8)
    vectors = []
    start = time.perf_counter()
    for i in range(0, len(texts), 12):
        vectors.extend(provider.embed(texts[i:i+12]))
        print(f"encode {min(i+12,len(texts))}/{len(texts)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    output = RESULTS / "qwen_vectors.npz"
    np.savez_compressed(output, vectors=np.asarray(vectors))
    write("embedding_metadata.json", {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": ref, "dim": provider.dim, "texts": len(texts), "seconds": time.perf_counter()-start, "vectors_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "offline": True, "threads": 4})


class FrozenChat:
    configured = True
    def __init__(self):
        self.cache = read("llm_responses.json")
        self.events = []
    def chat(self, messages, **kwargs):
        key = digest(messages[0]["content"])
        row = self.cache.get(key)
        self.events.append({"key": key, "status": row.get("status") if row else "missing"})
        if not row or row["status"] != "ok":
            raise RuntimeError("frozen LLM input unavailable")
        return row["raw"]


def frozen_provider():
    import numpy as np
    from sme.embedding.base import EmbeddingProvider
    texts = read("embedding_texts.json")
    vectors = np.load(RESULTS / "qwen_vectors.npz")["vectors"]
    table = dict(zip(texts, vectors))
    if (RESULTS / "supplement_vectors.npz").exists():
        extra = np.load(RESULTS / "supplement_vectors.npz")["vectors"]
        table.update(zip(read("supplement_embedding_texts.json"), extra))
    class Frozen(EmbeddingProvider):
        name = "sentence-transformers"
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        dim = 1024
        mrl_dim = 0
        query_prefix = ""
        def embed(self, values):
            return [table[t].copy() for t in values]
    return Frozen()


def new_engine(path, provider, arm, wal=False, backend="json"):
    from sme.config import SMEConfig
    from sme.engine import SpatialMemoryEngine
    cfg = SMEConfig()
    cfg.storage.path = str(path)
    cfg.storage.backend = backend
    cfg.storage.autosave = False
    cfg.storage.compress = False
    cfg.maintenance.background = False
    cfg.policy.decay_enabled = False
    cfg.llm.base_url = ""
    cfg.llm.api_key = ""
    cfg.persistence.enabled = wal
    cfg.persistence.checkpoint_every = 100000
    cfg.persistence.sync_mode = "fsync"
    cfg.extraction.enabled = arm != "raw"
    cfg.extraction.mode = "llm" if arm.startswith("llm") else "rules"
    cfg.factversion.enabled = arm.endswith("_version") and not arm.endswith("no_version")
    if provider is not None:
        cfg.embedding.provider = provider.name
        cfg.embedding.model = provider.model_name
        cfg.embedding.dim = provider.dim
        with patch("sme.engine.build_embedding_provider", return_value=provider):
            e = SpatialMemoryEngine(cfg)
    else:
        e = SpatialMemoryEngine(cfg)
    if arm.startswith("llm"):
        e.extraction.llm = FrozenChat()
    return e


def attach_trace(e):
    log = {"extraction": [], "matching": []}
    original_extract = e.extraction.extract
    def extract_wrapper(text, assistant=False):
        n = len(e.extraction.llm.events) if isinstance(e.extraction.llm, FrozenChat) else 0
        facts = original_extract(text, assistant=assistant)
        events = e.extraction.llm.events[n:] if isinstance(e.extraction.llm, FrozenChat) else []
        log["extraction"].append({"input": text, "facts_before_pipeline_markers": [vars(f).copy() for f in facts], "llm": events,
                                  "rules_fallback": bool(events and (events[-1]["status"] != "ok" or not e.extraction.llm.cache[events[-1]["key"]].get("facts")))})
        return facts
    e.extraction.extract = extract_wrapper
    original_match = e.factversion._find_matches
    def match_wrapper(engine, fact, ns=None):
        old, similarity = original_match(engine, fact, ns)
        log["matching"].append({"fact": vars(fact).copy(), "ns": ns, "best_id": old.id if old else None,
                                "best_text": old.text if old else None, "cosine": similarity,
                                "correction_threshold": e.factversion._threshold(e, "correction_threshold"),
                                "dedup_threshold": e.factversion._threshold(e, "dedup_threshold")})
        return old, similarity
    e.factversion._find_matches = match_wrapper
    return log


def matching_memory(memory, spec):
    return (memory.metadata.get("experiment_turn") == spec["turn"]
            and all(t in memory.text for t in spec["all"])
            and all(t not in memory.text for t in spec.get("absent", []))
            and ("ns" not in spec or memory.metadata.get("ns") == spec["ns"]))


def memories(e):
    return [m.to_dict(include_embedding=False) for m in e.memories.values()]


def assess(e, case):
    query_results = []
    for q in case["queries"]:
        hits = e.search(q["text"], top_k=5, ns=q["ns"], include_history=q["include_history"])
        good = [i+1 for i, h in enumerate(hits) if matching_memory(h.memory, q["gold"]) and
                (q["include_history"] or not h.memory.metadata.get("superseded_by"))]
        query_results.append({**q, "gold_present": any(matching_memory(m, q["gold"]) for m in e.memories.values()),
                              "rank": min(good) if good else None, "top1": bool(good and good[0] == 1), "top5": bool(good),
                              "hits": [{"rank": i+1, "score": h.score, "breakdown": vars(h.breakdown), "memory": h.memory.to_dict(include_embedding=False)} for i,h in enumerate(hits)]})
    links = []
    for link in case["retire"]:
        old = [m for m in e.memories.values() if matching_memory(m, link["old"])]
        newer = [m for m in e.memories.values() if matching_memory(m, link["new"])]
        new_ids = {m.id for m in newer}
        paths = []
        for m in old:
            ids, current = [m.id], m
            while current.metadata.get("superseded_by") in e.memories and len(ids) < 64:
                current = e.memories[current.metadata["superseded_by"]]
                if current.id in ids:
                    break
                ids.append(current.id)
            paths.append(ids)
        correct = bool(old and newer and all(set(p[1:]) & new_ids for p in paths))
        links.append({**link, "old_ids": [m.id for m in old], "new_ids": list(new_ids), "paths": paths, "correct": correct})
    protected = []
    for spec in case["protect"]:
        ms = [m for m in e.memories.values() if matching_memory(m, spec)]
        protected.append({"spec": spec, "ids": [m.id for m in ms], "retired_ids": [m.id for m in ms if m.metadata.get("superseded_by")],
                          "safe": bool(ms and all(not m.metadata.get("superseded_by") for m in ms))})
    timeline = []
    for link in links:
        for mid in link["old_ids"][:1]:
            timeline.append(e.factversion.timeline(mid, e))
    return {"queries": query_results, "links": links, "protected": protected, "timelines": timeline,
            "memories": memories(e)}


def run_case(case, provider_name, provider, arm, suffix="", threshold=None):
    path = TMP / "states" / (provider_name + "_" + arm + suffix) / case["id"] / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    e = new_engine(path, provider, arm)
    if threshold is not None:
        e.config.factversion.correction_threshold = threshold
    trace = attach_trace(e)
    writes = []
    try:
        for i, t in enumerate(case["turns"]):
            before = memories(e)
            result = e.add(t["text"], ns=t["ns"], conversation_id=t["conversation_id"],
                           metadata={"experiment_case": case["id"], "experiment_turn": i,
                                     "input_sha256": digest(t["text"])})
            writes.append({"turn": i, "input": t, "returned_id": result.id,
                           "returned_is_stored": result.id in e.memories, "before": before, "after": memories(e)})
        result = assess(e, case)
        return {"case": case["id"], "category": case["category"], "title": case["title"],
                "provider": provider_name, "arm": arm, "status": "ok", "writes": writes,
                "trace": trace, **result}
    except Exception as exc:
        return {"case": case["id"], "category": case["category"], "provider": provider_name,
                "arm": arm, "status": "error", "error": type(exc).__name__, "traceback": traceback.format_exc(), "writes": writes, "trace": trace}
    finally:
        e.wal.close()


def run():
    plan = read("plan.json")
    provider = frozen_provider()
    rows = []
    for name, p in [("hashing", None), ("qwen", provider)]:
        for arm in plan["scoring"]["arms"]:
            for case in plan["cases"]:
                row = run_case(case, name, p, arm)
                rows.append(row)
            write("pipeline_results.json", rows)
            subset = [r for r in rows if r["provider"] == name and r["arm"] == arm and r["category"] == "explicit"]
            qs = [q for r in subset for q in r.get("queries", [])]
            print(f"run {name}/{arm}: explicit top1={sum(q['top1'] for q in qs)}/{len(qs)} errors={sum(r['status']!='ok' for r in rows)}", flush=True)
    summary(rows)


def summary(rows=None):
    rows = rows or read("pipeline_results.json")
    groups = []
    for provider in sorted({r["provider"] for r in rows}):
        for arm in read("plan.json")["scoring"]["arms"]:
            for category in sorted({r["category"] for r in rows}):
                rs = [r for r in rows if r["provider"] == provider and r["arm"] == arm and r["category"] == category]
                qs = [q for r in rs for q in r.get("queries", [])]
                ls = [l for r in rs for l in r.get("links", [])]
                ps = [p for r in rs for p in r.get("protected", [])]
                groups.append({"provider": provider, "arm": arm, "category": category, "cases": len(rs), "errors": sum(r["status"] != "ok" for r in rs), "queries": len(qs), "gold_present": sum(q["gold_present"] for q in qs), "top1": sum(q["top1"] for q in qs), "top5": sum(q["top5"] for q in qs), "link_targets": len(ls), "correct_links": sum(l["correct"] for l in ls), "protected_targets": len(ps), "protected_safe": sum(p["safe"] for p in ps), "fallback_calls": sum(x["rules_fallback"] for r in rs for x in r.get("trace", {}).get("extraction", []))})
    write("summary.json", groups)


def logical_state(e):
    return {m.id: {"text": m.text, "metadata": m.metadata, "tags": sorted(m.tags), "source": m.source,
                   "created_at": m.created_at, "archived": m.archived} for m in e.memories.values()}


def thresholds():
    """Configuration-only diagnostic; separate from the frozen main comparison."""
    spec = {"hypothesis": "Can a single public correction_threshold improve missed links without extra unrelated retirement?",
            "values": [0.50, 0.60, 0.66, 0.72, 0.85], "provider": "qwen", "arm": "llm_version",
            "note": "Existing provider adaptation is retained unchanged; inspect effective thresholds, not just configured values."}
    write("threshold_plan.json", spec)
    provider = frozen_provider()
    rows, totals = [], []
    for value in spec["values"]:
        rs = [run_case(c, "qwen", provider, "llm_version", suffix=f"_threshold_{value}", threshold=value) for c in read("plan.json")["cases"]]
        rows.extend({"configured_threshold": value, **r} for r in rs)
        explicit = [r for r in rs if r["category"] == "explicit"]
        qs = [q for r in explicit for q in r.get("queries", [])]
        ls = [l for r in explicit for l in r.get("links", [])]
        ps = [p for r in rs if r["category"] == "safety" for p in r.get("protected", [])]
        effective = sorted({t["correction_threshold"] for r in rs for t in r.get("trace", {}).get("matching", [])})
        totals.append({"configured": value, "effective": effective, "explicit_top1": sum(q["top1"] for q in qs),
                       "explicit_queries": len(qs), "strict_links": sum(l["correct"] for l in ls), "link_targets": len(ls),
                       "protected_originals_safe": sum(p["safe"] for p in ps), "protected_original_targets": len(ps)})
    write("threshold_results.json", {"summary": totals, "rows": rows})
    print(json.dumps(totals, ensure_ascii=False), flush=True)


def natural_plan():
    """Exploratory supplement, frozen before extra model calls; main plan unchanged."""
    cases = []
    for cid, title, old, new, query, a, b in [
        ("U01", "第一人称住址更正", "我现在住在北京。", "你记错了，我现在已经搬到上海了。", "我现在住在哪里？", "北京", "上海"),
        ("U02", "第一人称饮料更正", "我每天早上都喜欢喝咖啡。", "更正一下，我现在不喝咖啡了，改喝绿茶。", "我现在喝什么饮料？", "咖啡", "绿茶"),
        ("U03", "第一人称宠物改名", "我养的猫叫团子。", "纠正一下，我的猫现在改名叫豆包了。", "我的猫现在叫什么？", "团子", "豆包"),
        ("U04", "第一人称工作单位更正", "我在海星科技公司上班。", "之前你记错了，我已经跳槽到晨光公司上班了。", "我现在在哪家公司工作？", "海星", "晨光"),
    ]:
        cases.append({"id": cid, "title": title, "category": "natural_explicit", "turns": [turn(old), turn(new, conv="session_after_ten_days")],
                      "queries": [question(query, target(1, b))], "protect": [], "retire": [{"old": target(0, a), "new": target(1, b)}]})
    cases += [
        {"id": "U05", "title": "第一人称澄清朋友与本人", "category": "natural_safety",
         "turns": [turn("我每天喜欢喝咖啡。"), turn("更正一下，不喝咖啡的是我朋友小周，我自己还是喜欢喝咖啡。")],
         "queries": [question("我自己还喜欢喝咖啡吗？", target(0, "咖啡"))], "protect": [target(0, "咖啡")], "retire": []},
        {"id": "U06", "title": "第一人称跨会话省略对象", "category": "natural_context",
         "turns": [turn("我现在住在北京。"), turn("不对，是上海。", conv="session_after_ten_days")],
         "queries": [question("我现在住在哪里？", target(1, "上海"))], "protect": [],
         "retire": [{"old": target(0, "北京"), "new": target(1, "上海")}]}]
    write("natural_language_plan.json", {"purpose": "First-person user-facing phrasing; exploratory supplement, not part of frozen original 32 cases", "time": "session name only, no simulated ten-day event timestamp and no real ten-day wait", "cases": cases})
    print("Frozen six first-person scenarios before additional LLM calls.")


def encode_extra():
    import numpy as np
    import torch
    from sme.modules.extraction import _rules_extract
    from sme.embedding.sentence_transformers_provider import SentenceTransformersProvider
    torch.set_num_threads(4)
    texts = set()
    cases = read("natural_language_plan.json")["cases"]
    if (RESULTS / "targeted_safety_plan.json").exists():
        cases += read("targeted_safety_plan.json")["cases"]
    for c in cases:
        for t in c["turns"]:
            texts.add(t["text"])
            texts.update(f.text for f in _rules_extract(t["text"]))
        texts.update(q["text"] for q in c["queries"])
    for row in read("llm_responses.json").values():
        texts.update(f["text"] for f in row.get("facts", []))
    previous_texts, previous_vectors = [], []
    if (RESULTS / "supplement_vectors.npz").exists():
        previous_texts = read("supplement_embedding_texts.json")
        previous_vectors = list(np.load(RESULTS / "supplement_vectors.npz")["vectors"])
    texts = sorted(texts - set(read("embedding_texts.json")) - set(previous_texts))
    if not texts:
        print("No missing vectors.")
        return
    cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B"
    ref = (cache / "refs/main").read_text().strip()
    print(f"Encoding {len(texts)} extra texts, original 127 vectors remain untouched.", flush=True)
    provider = SentenceTransformersProvider(str(cache / "snapshots" / ref), batch_size=8)
    start = time.perf_counter()
    vectors = []
    for i in range(0, len(texts), 12):
        vectors.extend(provider.embed(texts[i:i+12]))
        print(f"supplement encode {min(i+12,len(texts))}/{len(texts)}", flush=True)
    write("supplement_embedding_texts.json", previous_texts + texts)
    np.savez_compressed(RESULTS / "supplement_vectors.npz", vectors=np.asarray(previous_vectors + vectors))
    write("supplement_embedding_metadata.json", {"texts": len(previous_texts)+len(texts), "new_texts": len(texts), "previous_frozen_vectors_kept": len(previous_texts), "seconds": time.perf_counter()-start, "model_revision": ref, "sha256": hashlib.sha256((RESULTS / "supplement_vectors.npz").read_bytes()).hexdigest(), "main_vectors_unchanged": hashlib.sha256((RESULTS / "qwen_vectors.npz").read_bytes()).hexdigest() == read("embedding_metadata.json")["vectors_sha256"]})


def natural():
    rows = []
    provider = frozen_provider()
    for name, p in [("hashing", None), ("qwen", provider)]:
        for arm in read("plan.json")["scoring"]["arms"]:
            for case in read("natural_language_plan.json")["cases"]:
                rows.append(run_case(case, name, p, arm, suffix="_natural"))
    write("natural_language_results.json", rows)
    for arm in ["rules_version", "llm_no_version", "llm_version"]:
        rs = [r for r in rows if r["provider"] == "qwen" and r["arm"] == arm and r["category"] == "natural_explicit"]
        print(arm, "natural explicit top1", sum(q["top1"] for r in rs for q in r["queries"]), "/4", "links", sum(l["correct"] for r in rs for l in r["links"]), "/4", flush=True)


def targeted_plan():
    cases = []
    definitions = [
        ("E01", "已知本人和朋友，只改朋友", ["我每天喜欢喝咖啡。", "我的朋友小周每天喜欢喝咖啡。", "只纠正我朋友小周的饮料偏好：小周现在改喝绿茶，我自己的偏好没有变化。"],
         target(0, "咖啡"), target(1, "小周", "咖啡"), target(2, "小周", "绿茶"), "我自己平时喜欢喝什么？", "我的朋友小周现在喝什么？"),
        ("E02", "已知猫和狗，只改狗名", ["我养的猫叫团子。", "我另外养了一只狗，名字叫汤圆。", "只纠正狗的名字：我另外养的狗叫豆包，猫的记录不变。"],
         target(0, "猫", "团子"), target(1, "狗", "汤圆"), target(2, "狗", "豆包"), "我的猫叫什么名字？", "我的狗现在叫什么名字？"),
        ("E03", "已知生产和测试，只改测试端口", ["Atlas 生产环境的端口是8080。", "Atlas 测试环境的端口是8081。", "只纠正测试环境：Atlas 测试环境的端口改成9090，生产环境的记录不变。"],
         target(0, "生产", "8080"), target(1, "测试", "8081"), target(2, "测试", "9090"), "Atlas 生产环境端口是多少？", "Atlas 测试环境现在端口是多少？"),
        ("E04", "已知饮料和杯色，只改杯色", ["我每天喜欢喝咖啡。", "我使用的咖啡杯颜色是红色。", "只纠正咖啡杯颜色：我使用的咖啡杯是绿色，喝咖啡的偏好没有变化。"],
         target(0, "喝", "咖啡"), target(1, "咖啡杯", "红色"), target(2, "咖啡杯", "绿色"), "我每天喜欢喝什么？", "我的咖啡杯现在是什么颜色？"),
    ]
    for cid, title, texts, protected, old, new, q_old, q_new in definitions:
        cases.append({"id": cid, "title": title, "category": "targeted_safety", "turns": [turn(t, conv=f"session_{i}") for i,t in enumerate(texts)],
                      "queries": [question(q_old, protected), question(q_new, new)], "protect": [protected], "retire": [{"old": old, "new": new}]})
    write("targeted_safety_plan.json", {"purpose": "Extra explicit-scope tests: both objects already recorded; user says only one changes. Frozen before new model calls. Earlier underspecified guard cases are not treated as definitive bugs without semantic review.", "cases": cases})
    print("Frozen four explicit-scope guard scenarios.")


def targeted():
    provider = frozen_provider()
    rows = []
    for name,p in [("hashing",None),("qwen",provider)]:
        for arm in ["rules_no_version","rules_version","llm_no_version","llm_version"]:
            rows += [run_case(c,name,p,arm,suffix="_targeted") for c in read("targeted_safety_plan.json")["cases"]]
    write("targeted_safety_results.json", rows)
    for r in rows:
        if r["provider"] == "qwen":
            print(r["arm"], r["case"], "rank", [q["rank"] for q in r["queries"]], "original_not_retired", [p["safe"] for p in r["protected"]], "target_link", [l["correct"] for l in r["links"]], flush=True)


def aged():
    """Simulated 15-day record age, no real wait and no changes to saved source."""
    provider = frozen_provider()
    cases = [c for c in read("natural_language_plan.json")["cases"] if c["category"] == "natural_explicit"]
    write("aged_plan.json", {"days": 15, "cases": [c["id"] for c in cases], "arms": ["rules_version","llm_version"], "decay": [False,True], "method": "mock utils.time.time only around isolated add calls; actual created_at/last_hit recorded; base comparison remains untouched"})
    rows = []
    base_time = time.time()
    for decay in [False, True]:
        for arm in ["rules_version", "llm_version"]:
            for c in cases:
                path = TMP / "aged" / (arm+str(decay)) / c["id"] / "state.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                e = new_engine(path,provider,arm)
                e.config.policy.decay_enabled = decay
                trace = attach_trace(e)
                for i,t in enumerate(c["turns"]):
                    stamp = base_time - 15*86400 if i == 0 else base_time
                    with patch("sme.utils.time.time", return_value=stamp):
                        e.add(t["text"],ns=t["ns"],conversation_id=t["conversation_id"],metadata={"experiment_case":c["id"],"experiment_turn":i})
                rows.append({"case": c["id"], "arm":arm,"decay":decay,"trace":trace,**assess(e,c)})
                e.wal.close()
    write("aged_results.json",rows)
    print("Completed sixteen simulated 15-day scenarios.")


def worker(case_id, arm, backend, kind):
    case = next(c for c in read("plan.json")["cases"] if c["id"] == case_id)
    provider = frozen_provider()
    directory = TMP / "recovery" / (case_id + "_" + arm + "_" + backend + "_" + kind)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("state.db" if backend == "sqlite" else "state.json")
    e = new_engine(path, provider, arm, wal=True, backend=backend)
    for i, t in enumerate(case["turns"]):
        e.add(t["text"], ns=t["ns"], conversation_id=t["conversation_id"], metadata={"experiment_case": case_id, "experiment_turn": i})
        if i == 0 and kind == "snapshot_plus_wal":
            e.save()
    e.wal.flush()
    write("expected_" + directory.name + ".json", logical_state(e))
    os._exit(73)


def crash():
    provider = frozen_provider()
    rows = []
    for case_id in ["C01", "C03", "C11", "G01"]:
        for arm in ["rules_version", "llm_version"]:
            for backend in ["json", "sqlite"]:
                for kind in ["wal_only", "snapshot_plus_wal"]:
                    name = case_id + "_" + arm + "_" + backend + "_" + kind
                    cp = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--case", case_id, "--arm", arm, "--backend", backend, "--kind", kind], cwd=EXP, capture_output=True, text=True, encoding="utf-8", timeout=45)
                    path = TMP / "recovery" / name / ("state.db" if backend == "sqlite" else "state.json")
                    row = {"name": name, "exit_code": cp.returncode, "expected_hard_exit": cp.returncode == 73}
                    try:
                        expected = read("expected_" + name + ".json")
                        state_checks = []
                        for _ in range(2):
                            e = new_engine(path, provider, arm, wal=True, backend=backend)
                            row["loaded"] = e.load()
                            state_checks.append(logical_state(e) == expected)
                            e.wal.close()
                        row["two_fresh_recoveries_equal"] = all(state_checks)
                    except Exception:
                        row["error"] = traceback.format_exc()
                    rows.append(row)
                    write("recovery_results.json", rows)
        print(f"recovery {case_id}: {sum(r.get('two_fresh_recoveries_equal',False) and r['expected_hard_exit'] for r in rows)}/{len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["plan", "extract", "encode", "run", "summary", "crash", "worker", "thresholds", "natural_plan", "encode_extra", "natural", "targeted_plan", "targeted", "aged"])
    ap.add_argument("--case")
    ap.add_argument("--arm")
    ap.add_argument("--backend")
    ap.add_argument("--kind")
    args = ap.parse_args()
    if args.task == "plan":
        if (RESULTS / "plan.json").exists():
            raise RuntimeError("Frozen plan already exists")
        write("plan.json", make_plan())
        write("plan_sha256.json", {"sha256": hashlib.sha256((RESULTS / "plan.json").read_bytes()).hexdigest()})
        print("Frozen 32 cases and scoring before execution.")
    elif args.task == "worker":
        worker(args.case, args.arm, args.backend, args.kind)
    else:
        globals()[args.task]()


if __name__ == "__main__":
    main()
