# -*- coding: utf-8 -*-
"""
真实数据集 → runner 格式转换(datasets/synth/{name}.jsonl + .queries.jsonl 同构)。
抽样原则:规模控制在 medium 档可比(检索池 5k-15k,查询 300-500),时间戳铺开模拟记忆年龄。
"""
import gzip
import json
import random
import time
from pathlib import Path

DS = Path(__file__).resolve().parent.parent.parent / "datasets"  # experiment/datasets
SYNTH = DS / "synth"
SYNTH.mkdir(parents=True, exist_ok=True)
NOW = int(time.time())
DAY = 86400


def dump(name, mems, queries):
    with open(SYNTH / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for m in mems:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(SYNTH / f"{name}.queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[{name}] memories={len(mems)} queries={len(queries)}")


def conv_dureader(n_queries=300, seed=11):
    """DuReader-retrieval dev:正例=gold,负例=强干扰,合并入池"""
    rnd = random.Random(seed)
    src = DS / "DuReader" / "dureader_retrieval_dev.jsonl.gz"
    rows = [json.loads(l) for l in gzip.open(src, "rt", encoding="utf-8")][: n_queries * 2]
    rnd.shuffle(rows)
    rows = rows[:n_queries]
    mems, queries = [], []
    pool = {}
    for r in rows:
        gold = []
        for grp, is_gold in (("positive_passages", True), ("negative_passages", False)):
            for p in r.get(grp, [])[:8]:
                key = p["docid"]
                if key not in pool:
                    pool[key] = {
                        "id": f"dr_{key[:24]}", "text": p["text"][:600],
                        "ts": NOW - int(rnd.choice([1, 7, 30, 90, 180, 365]) * DAY),
                        "importance": 0.8 if is_gold else rnd.choice([0.2, 0.5]),
                        "lang": "zh",
                    }
                    mems.append(pool[key])
                if is_gold:
                    gold.append(pool[key]["id"])
        if gold:
            queries.append({"qid": r["query_id"][:24], "text": r["query"], "gold": gold, "type": "direct"})
    dump("dureader", mems, queries)


def conv_cmrc(n_queries=500, seed=12):
    """CMRC2018 dev:段落为检索单元,answer_start 命中段落即 gold"""
    rnd = random.Random(seed)
    src = DS / "CMRC2018" / "cmrc2018_dev.json"
    data = json.load(open(src, encoding="utf-8"))
    arts = data["data"] if isinstance(data, dict) and "data" in data else data
    paras, queries = [], []
    pid = [0]
    for art in arts:
        for para in art["paragraphs"]:
            pid[0] += 1
            p_id = f"cm_{pid[0]:05d}"
            paras.append({"id": p_id, "text": para["context"][:1200],
                          "ts": NOW - int(rnd.choice([1, 30, 90, 365]) * DAY),
                          "importance": 0.5, "lang": "zh"})
            for qa in para["qas"]:
                if not qa["answers"]:
                    continue
                queries.append({"qid": qa["id"][:24], "text": qa["question"], "gold": [p_id], "type": "direct"})
    rnd.shuffle(queries)
    queries = queries[:n_queries]
    queries.sort(key=lambda q: q["qid"])  # 稳定顺序
    dump("cmrc", paras, queries)


def conv_longmemeval(n_samples=25, seed=13):
    """LongMemEval oracle:haystack 会话轮次入池,has_answer 轮为 gold"""
    rnd = random.Random(seed)
    src = DS / "LongMemEval" / "longmemeval_oracle.json"
    rows = json.load(open(src, encoding="utf-8"))
    rnd.shuffle(rows)
    rows = rows[:n_samples]
    mems, queries = [], []
    mid = [0]
    for r in rows:
        gold_ids = []
        sess_ids = r.get("haystack_session_ids") or []
        ans_ids = set(r.get("answer_session_ids") or [])
        for s_i, sess in enumerate(r["haystack_sessions"]):
            is_ans_sess = (sess_ids[s_i] in ans_ids) if s_i < len(sess_ids) else True
            for turn in sess:
                mid[0] += 1
                m_id = f"lm_{mid[0]:06d}"
                age = rnd.randint(1, 365)
                mems.append({"id": m_id,
                             "text": f"{turn['role']}: {turn['content']}"[:600],
                             "ts": NOW - age * DAY, "importance": 0.5, "lang": "en"})
                if turn.get("has_answer") and is_ans_sess:
                    gold_ids.append(m_id)
        if gold_ids:
            queries.append({"qid": r["question_id"][:24], "text": r["question"], "gold": gold_ids, "type": "temporal"})
    dump("longmemeval", mems, queries)


def conv_locomo(n_conv=2, seed=14):
    """LoCoMo:前 n_conv 段对话,轮次入池,evidence dia_id 为 gold"""
    src = DS / "LoCoMo" / "locomo10.json"
    rows = json.load(open(src, encoding="utf-8"))[:n_conv]
    rnd = random.Random(seed)
    mems, queries = [], []
    mid = [0]
    for row in rows:
        dia_map = {}
        for sess_k, turns in row["conversation"].items():
            if not isinstance(turns, list):
                continue  # speaker_a / session_N_date_time 等元数据键
            for turn in turns:
                if "dia_id" not in turn:
                    continue
                mid[0] += 1
                m_id = f"lc_{mid[0]:06d}"
                dia_map[turn["dia_id"]] = m_id
                mems.append({"id": m_id,
                             "text": f"{turn.get('speaker','?')}: {turn.get('text','')}"[:600],
                             "ts": NOW - rnd.randint(1, 365) * DAY,
                             "importance": 0.5, "lang": "en"})
        for qa in row.get("qa", []):
            gold = [dia_map[d] for d in qa.get("evidence", []) if d in dia_map]
            if gold:
                queries.append({"qid": f"lc_q{len(queries)+1:04d}", "text": qa["question"],
                                "gold": gold, "type": "multi" if len(gold) > 1 else "direct"})
    dump("locomo", mems, queries)


if __name__ == "__main__":
    conv_dureader()
    conv_cmrc()
    conv_longmemeval()
    conv_locomo()
