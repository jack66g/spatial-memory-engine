"""从对话脚本自动生成评测任务集（直接回忆 / 长尾细节 / 纠错最新说法）。

判分关键词从期望答案自动提取（数字 + 非停用词 token），
纠错题的 conflict_text = 旧说法（带偏检测用）。

用法::

    python scripts/make_tasks.py --dialogue data/dialogue_300.json \
        --out data/tasks_dialogue.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

STOP = set("我的了在是和一个不就有也这那吗呢吧呀哦嗯哈对吧都还着过被把从向对跟与或以及".strip())

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]+")


def extract_keywords(text: str, max_k: int = 3) -> list[str]:
    """从期望答案提取判分关键词：数字 + 有意义 token。"""
    kws: list[str] = []
    kws += NUM_RE.findall(text)
    for tok in TOKEN_RE.findall(text):
        if len(tok) >= 2 and not all(c in STOP for c in tok):
            kws.append(tok)
        if len(kws) >= max_k + 2:
            break
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
        if len(out) >= max_k:
            break
    return out


def fact_to_question(fact: str) -> str:
    for pattern, repl in [
        ("我的名字叫", "小林叫什么名字"),
        ("我叫", "小林叫什么名字"),
        ("我今年", "小林今年多大"),
        ("我住在", "小林住在哪里"),
        ("我在", "小林在哪里工作"),
        ("我养了一只", "小林养了什么宠物"),
        ("我女朋友叫", "小林女朋友叫什么"),
        ("我有个妹妹", "小林有兄弟姐妹吗"),
        ("我最喜欢的", "小林最喜欢什么"),
        ("我最近在学", "小林最近在学什么"),
        ("我最近在读", "小林最近在读什么书"),
        ("我每周四", "小林每周四做什么"),
        ("我一般十一点半睡觉", "小林一般几点睡觉"),
        ("我早餐一般", "小林早餐吃什么"),
        ("我坐地铁", "小林怎么通勤"),
    ]:
        if pattern in fact:
            return repl
    base = fact.rstrip("。")
    return f"小林{base}吗"


def turn_positions(dialogue: dict) -> dict:
    """fact_id -> 出现轮次列表（用户消息）。"""
    pos: dict[str, list[int]] = {}
    for i, turn in enumerate(dialogue["turns"]):
        fid = turn.get("fact_id")
        if fid and turn["role"] == "user":
            pos.setdefault(fid, []).append(i)
    return pos


def build_tasks(dialogue: dict) -> dict:
    facts = {f["id"]: f for f in dialogue["facts"]}
    pos = turn_positions(dialogue)
    total = len(dialogue["turns"])
    latest: dict[str, str] = {fid: f["text"] for fid, f in facts.items()}
    for turn in dialogue["turns"]:
        if turn.get("kind") == "correction" and turn.get("fact_id"):
            latest[turn["fact_id"]] = turn["new"]

    tasks: list[dict] = []

    # --- 1) 直接回忆（陈述过的事实）---------------------------------- #
    for fid, f in facts.items():
        if len(tasks) >= 15:
            break
        expect = latest.get(fid, f["text"])
        kws = extract_keywords(expect)
        if not kws:
            continue
        tasks.append({
            "id": f"recall_{fid}", "scene": "dialogue",
            "question": fact_to_question(f["text"]),
            "answer_keywords": kws, "expect_text": expect,
            "conflict_text": None, "top_k": 5,
        })

    # --- 2) 长尾细节（只在对话前 1/3 出现过的事实）--------------------- #
    early = [(fid, min(ps)) for fid, ps in pos.items()
             if ps and min(ps) < total / 3]
    early.sort(key=lambda x: x[1])
    for fid, _ in early:
        if len(tasks) >= 30:
            break
        f = facts[fid]
        expect = latest.get(fid, f["text"])
        kws = extract_keywords(expect)
        if not kws:
            continue
        tasks.append({
            "id": f"longtail_{fid}", "scene": "dialogue",
            "question": fact_to_question(f["text"]),
            "answer_keywords": kws, "expect_text": expect,
            "conflict_text": None, "top_k": 5,
        })

    # --- 3) 纠错最新说法（含带偏检测）--------------------------------- #
    corrections = [t for t in dialogue["turns"]
                   if t.get("kind") == "correction" and t.get("fact_id")]
    for turn in corrections:
        if len(tasks) >= 40:
            break
        fid = turn["fact_id"]
        f = facts.get(fid)
        if f is None:
            continue
        old = turn.get("old") or f["text"]
        new = turn["new"]
        kws = extract_keywords(new)
        if not kws:
            continue
        tasks.append({
            "id": f"corr_{fid}", "scene": "dialogue",
            "question": fact_to_question(old),
            "answer_keywords": kws, "expect_text": new,
            "conflict_text": old, "top_k": 5,
        })

    # --- 4) 问答对回放（对话中出现过的问题 + 助手回答）----------------- #
    assistant_after: dict[str, str] = {}
    for i, turn in enumerate(dialogue["turns"]):
        if (turn["role"] == "user" and turn.get("kind") == "question"
                and i + 1 < len(dialogue["turns"])
                and dialogue["turns"][i + 1]["role"] == "assistant"):
            assistant_after[turn["text"]] = dialogue["turns"][i + 1]["text"]
    for q, a in list(assistant_after.items())[:10]:
        kws = extract_keywords(a)
        if not kws:
            continue
        tasks.append({
            "id": f"qa_{len(tasks)}", "scene": "dialogue",
            "question": q, "answer_keywords": kws,
            "expect_text": a, "conflict_text": None, "top_k": 5,
        })

    # --- 5) 画像聚合（"小林有哪些爱好"类，期望命中任一画像事实）--------- #
    profile_sets = [
        ("小林平时有什么兴趣爱好", ["篮球", "钢琴", "摄影", "塞尔达", "胶片"]),
        ("小林早餐一般吃什么", ["全麦面包", "鸡蛋"]),
    ]
    for i, (q, kws) in enumerate(profile_sets):
        tasks.append({
            "id": f"profile_{i}", "scene": "dialogue",
            "question": q, "answer_keywords": kws,
            "expect_text": None, "conflict_text": None, "top_k": 5,
        })

    return {"name": "dialogue_tasks", "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成对话评测任务集")
    parser.add_argument("--dialogue", default="data/dialogue_300.json")
    parser.add_argument("--out", default="data/tasks_dialogue.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dialogue = json.loads((root / args.dialogue).read_text(encoding="utf-8"))
    tasks = build_tasks(dialogue)
    out = root / args.out
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    kinds = {}
    for t in tasks["tasks"]:
        kinds[t["id"].split("_")[0]] = kinds.get(t["id"].split("_")[0], 0) + 1
    print(f"✓ 任务集 {len(tasks['tasks'])} 题 → {out}")
    print(f"  分布: {kinds}")


if __name__ == "__main__":
    main()
