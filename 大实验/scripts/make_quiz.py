"""从对话脚本生成考问集（纯检索级，不调 LLM，零成本）。

对每个事实生成 1-3 个问法；纠错轮的事实以「最新说法」为期望答案。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTION_VARIANTS = [
    "小林{question}？",
    "你还记得小林{question}吗？",
    "{question}，你知道吗？",
]


def fact_to_question(fact: str) -> list[str]:
    """陈述句 -> 问句（覆盖常见事实形态）。"""
    q = fact
    for pattern, repl in [
        ("我的名字叫", "叫什么名字"),
        ("我叫", "叫什么名字"),
        ("我今年", "今年多大"),
        ("我住在", "住在哪里"),
        ("我养了一只", "养了什么宠物"),
        ("我女朋友叫", "女朋友叫什么"),
        ("我有个妹妹", "有没有妹妹"),
        ("我在", "在哪里"),
        ("我最喜欢", "最喜欢什么"),
        ("我最近在读", "最近在读什么书"),
        ("我最近在学", "最近在学什么"),
    ]:
        if pattern in q:
            q = repl
            break
    else:
        if q.endswith("。"):
            q = q[:-1]
        q = q + "吗"
    return q


def build_quiz(dialogue: dict) -> dict:
    # 最新说法：纠错轮的新事实覆盖旧事实
    facts = {f["id"]: f["text"] for f in dialogue["facts"]}
    latest: dict[str, str] = dict(facts)
    for turn in dialogue["turns"]:
        if turn.get("kind") == "correction" and turn.get("fact_id"):
            latest[turn["fact_id"]] = turn["new"]

    queries = []
    for fid, text in facts.items():
        q = fact_to_question(text)
        if q == text:
            continue
        queries.append({
            "fact_id": fid,
            "topic": next((f["topic"] for f in dialogue["facts"]
                           if f["id"] == fid), ""),
            "q": q,
            "expect": latest[fid],
            "top_k": 5,
        })
    return {"persona": dialogue["persona"], "queries": queries}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="从对话生成考问集")
    parser.add_argument("--dialogue", default="data/dialogue_300.json")
    parser.add_argument("--out", default="data/quiz_300.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dialogue = json.loads((root / args.dialogue).read_text(encoding="utf-8"))
    quiz = build_quiz(dialogue)
    out = root / args.out
    out.write_text(json.dumps(quiz, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"✓ 考问集 {len(quiz['queries'])} 题 → {out}")


if __name__ == "__main__":
    main()
