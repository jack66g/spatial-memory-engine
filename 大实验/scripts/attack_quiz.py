"""薄弱点攻击题集：针对各竞品已知弱点设计的对抗性考问。

每道题标注 ``targets``（该题攻击谁的弱点）。攻击题与普通任务集分开，
在对比报告中单独呈现并标注弱点归属。

用法::

    python scripts/attack_quiz.py --dialogue data/dialogue_300.json \
        --out data/tasks_attack.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from make_tasks import extract_keywords, fact_to_question  # noqa: E402

DIALECT_QUESTIONS = {
    # 口语化/同义问法 -> 针对的事实 id
    "f_coffee": "小林平时喝什么饮品呀",
    "f_basketball": "小林周末喜欢啥运动",
    "f_dog": "小林养的那个毛茸茸的家伙叫啥",
    "f_city": "小林现在住哪儿来着",
    "f_age": "小林今年二十几了",
    "f_breakfast": "小林早上都吃些啥",
}


def make_dialect_questions(dialogue: dict, tasks: list[dict]) -> None:
    """攻击1：同义/口语化改写（针对 BM25 纯关键词）。"""
    facts = {f["id"]: f for f in dialogue["facts"]}
    for fid, q in DIALECT_QUESTIONS.items():
        f = facts.get(fid)
        if f is None:
            continue
        tasks.append({
            "id": f"attack_dialect_{fid}", "scene": "dialogue",
            "question": q,
            "answer_keywords": extract_keywords(f["text"]),
            "expect_text": f["text"],
            "conflict_text": None, "top_k": 5,
            "targets": ["bm25", "裸RAG"],
            "note": "同义口语化改写（BM25 关键词不重叠）",
        })


def make_double_correction(dialogue: dict, tasks: list[dict]) -> None:
    """攻击2：双重纠错（同一事实连续两次纠错，旧说法必须失效）。

    对话里同一 fact 只纠错一次，因此用内嵌场景：先回放 3 轮
    （陈述→纠错1→纠错2），再问最新说法。针对无版本管理的 RAG/BM25。
    """
    tasks.append({
        "id": "attack_double_corr", "scene": "dialogue",
        "question": "小林住在哪里",
        "answer_keywords": ["杭州"],
        "expect_text": "我又搬回杭州了，上海太贵了",
        "conflict_text": "我去年搬到上海了",
        "pre_turns": [
            "我住在杭州",
            "其实我去年搬到上海了",
            "不对不对，我又搬回杭州了，上海房租太贵了",
        ],
        "top_k": 5,
        "targets": ["裸RAG", "bm25", "无版本管理"],
        "note": "双重纠错：第二次纠错后，第一次纠错的旧说法仍被检索会带偏回答",
    })


def make_relational(dialogue: dict, tasks: list[dict]) -> None:
    """攻击3：跨话题关联（A 事实引 B 事实，针对无图谱选手）。

    查询用实体锚定（不带高频主语"小林"，否则 BGE 主语霸榜会掩盖
    图谱多跳的价值——这正是待测点之一）。
    """
    pairs = [
        ("f_dog", "f_cat", "可乐怕什么"),              # 金毛->打雷
        ("f_girlfriend", "f_plan_marry", "结婚计划定了吗"),  # 女朋友->明年结婚
        ("f_sister", "f_parents", "妹妹在哪里读书"),   # 妹妹->成都
    ]
    facts = {f["id"]: f["text"] for f in dialogue["facts"]}
    for a_id, b_id, q in pairs:
        if a_id in facts and b_id in facts:
            tasks.append({
                "id": f"attack_rel_{a_id}_{b_id}", "scene": "dialogue",
                "question": q,
                "answer_keywords": extract_keywords(facts[b_id]),
                "expect_text": facts[b_id],
                "conflict_text": None, "top_k": 8,
                "targets": ["无图谱选手", "裸RAG", "bm25", "mem0"],
                "note": f"需关联 {a_id}→{b_id} 两个事实",
            })


NOISE_LINES = ["今天天气不错哦"] * 30 + ["好的好的"] * 20

NOISE_FACTS = [
    ("f_sleep", "小林一般几点睡觉"),
    ("f_blood", "小林是什么血型"),
    ("f_breakfast", "小林早餐一般吃什么"),
]


def make_chitchat_crowd(dialogue: dict, tasks: list[dict]) -> None:
    """攻击4：噪音霸榜（先注入 50 条重复模板句，再问低频真实事实）。"""
    facts = {f["id"]: f for f in dialogue["facts"]}
    for i, (fid, q) in enumerate(NOISE_FACTS):
        f = facts.get(fid)
        if f is None:
            continue
        tasks.append({
            "id": f"attack_noise_{i}", "scene": "dialogue",
            "question": q,
            "answer_keywords": extract_keywords(f["text"]),
            "expect_text": f["text"],
            "conflict_text": None, "top_k": 5,
            "noise_inject": NOISE_LINES if i == 0 else None,
            "targets": ["无噪音抑制", "裸RAG", "bm25"],
            "note": "50 条重复模板句注入后，真实记忆被挤出 top-k",
        })


def build_attack_tasks(dialogue: dict) -> dict:
    tasks: list[dict] = []
    make_dialect_questions(dialogue, tasks)
    make_double_correction(dialogue, tasks)
    make_relational(dialogue, tasks)
    make_chitchat_crowd(dialogue, tasks)
    return {"name": "attack_quiz", "tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成薄弱点攻击题集")
    parser.add_argument("--dialogue", default="data/dialogue_300.json")
    parser.add_argument("--out", default="data/tasks_attack.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dialogue = json.loads((root / args.dialogue).read_text(encoding="utf-8"))
    tasks = build_attack_tasks(dialogue)
    out = root / args.out
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"✓ 攻击题集 {len(tasks['tasks'])} 题 → {out}")
    for t in tasks["tasks"]:
        print(f"  [{t['id']}] {t['question']}  <- {t['note']}")


if __name__ == "__main__":
    main()
