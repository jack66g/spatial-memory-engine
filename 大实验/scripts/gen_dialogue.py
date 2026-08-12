"""对话生成器：用真实 LLM 生成几百轮自然对话。

对话人物「小林」按固定事实剧本陈述信息（含重复、纠错、闲聊、提问），
助手回复由 DeepSeek 真实生成。输出 data/dialogue_*.json。

用法::

    python scripts/gen_dialogue.py --rounds 300 --out data/dialogue_300.json
    python scripts/gen_dialogue.py --rounds 10  --dry-run   # 先小额记账
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient  # noqa: E402

PERSONA = "小林，28 岁，在杭州做前端工程师"

# (fact_id, fact_text, topic)
FACTS = [
    ("f_name", "我的名字叫小林", "personal"),
    ("f_age", "我今年 28 岁", "personal"),
    ("f_city", "我住在杭州", "personal"),
    ("f_job", "我在一家互联网公司做前端工程师", "work"),
    ("f_team", "我们前端组一共六个人", "work"),
    ("f_remote", "我每周四在家远程办公", "work"),
    ("f_stack", "我主要用 Vue 和 TypeScript 写代码", "work"),
    ("f_lang", "我最近在学 Rust", "study"),
    ("f_study", "我每天通勤路上听英语播客", "study"),
    ("f_coffee", "我喜欢喝手冲咖啡，不太喝速溶的", "food"),
    ("f_sweet", "我不太吃甜食，奶茶只喝三分糖", "food"),
    ("f_spicy", "我很能吃辣，最爱重庆火锅", "food"),
    ("f_breakfast", "我早餐一般吃全麦面包和鸡蛋", "food"),
    ("f_basketball", "我周末经常和朋友打篮球", "hobby"),
    ("f_piano", "我小时候学过六年钢琴", "hobby"),
    ("f_read", "我最近在读《三体》第三部", "hobby"),
    ("f_photo", "我喜欢用胶片相机拍照", "hobby"),
    ("f_game", "我偶尔玩塞尔达传说", "hobby"),
    ("f_dog", "我养了一只金毛，叫可乐", "family"),
    ("f_cat", "可乐怕打雷，每次打雷都要躲到床底下", "family"),
    ("f_sister", "我有个妹妹，在成都读研究生", "family"),
    ("f_parents", "我爸妈住在宁波", "family"),
    ("f_girlfriend", "我女朋友叫小雨，是设计师", "family"),
    ("f_anniversary", "我们在一起三年了", "family"),
    ("f_plan_marry", "我们计划明年结婚", "plan"),
    ("f_plan_japan", "下个月我打算去日本玩一周", "plan"),
    ("f_plan_fitness", "我办了张健身卡，想每周去三次", "plan"),
    ("f_sleep", "我一般十一点半睡觉，早上七点起", "health"),
    ("f_glasses", "我近视三百度，平时戴眼镜", "health"),
    ("f_allergy", "我对花粉过敏，春天容易打喷嚏", "health"),
    ("f_blood", "我是 A 型血", "health"),
    ("f_apple", "我比较喜欢苹果，不喜欢香蕉", "opinion"),
    ("f_movie", "我最喜欢的电影是《星际穿越》", "opinion"),
    ("f_music", "我最近在循环听周杰伦的老歌", "opinion"),
    ("f_weather", "我讨厌梅雨天，衣服老是晾不干", "opinion"),
    ("f_book_shop", "我常去的那家书店在西湖边", "daily"),
    ("f_metro", "我坐地铁一号线通勤", "daily"),
    ("f_lunch", "我中午一般在公司食堂吃", "daily"),
    ("f_bike", "我有一辆折叠自行车，周末骑", "daily"),
]

# 陈述模板（把事实自然地说出口）
STATEMENT_TEMPLATES = [
    "{fact}。",
    "对了，{fact}。",
    "跟你说个事，{fact}。",
    "{fact}，这个我之前提过吗？",
    "哦对，{fact}。",
    "{fact}，记住了哈。",
    "顺便说下，{fact}。",
    "嗯，{fact}。",
]

# 纠错模板：旧事实 + 新事实
CORRECTION_TEMPLATES = [
    "其实不是这样的，{new}。",
    "不对不对，我之前说错了，{new}。",
    "更正一下，{new}。",
    "我想起来了，{new}。",
]

# 闲聊填充（不携带新事实）
CHITCHAT = [
    "哈哈，你反应真快",
    "今天天气不错",
    "好的好的",
    "嗯嗯，我明白了",
    "谢谢你呀",
    "对了你吃饭了吗",
    "工作好累啊今天",
    "周末愉快",
    "好嘞",
]

# 提问模板（用于问答对/考问场景）
QUESTION_TEMPLATES = [
    "那{question}？",
    "问你个问题，{question}？",
    "{question}，你知道吗？",
]

ASSISTANT_PROMPT = (
    "你是小林的 AI 助手，性格温和、简短回应。"
    "这是当前对话：\n{history}\n请用一句到两句话回复小林（中文）。"
)

CORRECTION_PAIRS = [
    ("f_coffee", "我喜欢喝手冲咖啡，不太喝速溶的", "我现在改喝速溶的了，手冲太麻烦"),
    ("f_city", "我住在杭州", "其实我去年搬到上海了"),
    ("f_dog", "我养了一只金毛，叫可乐", "可乐其实是只柯基"),
    ("f_spicy", "我很能吃辣，最爱重庆火锅", "我最近不太能吃辣了，胃不太好"),
    ("f_sleep", "我一般十一点半睡觉，早上七点起", "我现在改成十二点睡，八点起"),
]


def build_turn_plan(rng: random.Random, rounds: int) -> list[dict]:
    """生成轮次计划：陈述事实 + 纠错 + 闲聊 + 提问。"""
    plan: list[dict] = []
    facts = list(FACTS)
    rng.shuffle(facts)
    stated: set[str] = set()
    i = 0
    while len(plan) < rounds:
        kind = rng.choices(
            ["statement", "chitchat", "question", "correction"],
            weights=[55, 20, 15, 10],
        )[0]
        if kind == "statement":
            fact = next((f for f in facts if f[0] not in stated), None)
            if fact is None:
                fact = rng.choice(facts)
            stated.add(fact[0])
            plan.append({"kind": "statement", "fact_id": fact[0],
                         "text": rng.choice(STATEMENT_TEMPLATES).format(fact=fact[1])})
        elif kind == "correction":
            old_id, old_text, new_text = rng.choice(CORRECTION_PAIRS)
            plan.append({"kind": "correction", "fact_id": old_id,
                         "old": old_text, "new": new_text,
                         "text": rng.choice(CORRECTION_TEMPLATES).format(new=new_text)})
        elif kind == "question":
            fact = rng.choice(facts)
            plan.append({"kind": "question", "fact_id": fact[0],
                         "text": rng.choice(QUESTION_TEMPLATES).format(
                             question=_fact_to_question(fact[1]))})
        else:
            plan.append({"kind": "chitchat", "text": rng.choice(CHITCHAT)})
        i += 1
    return plan


def _fact_to_question(fact: str) -> str:
    """把陈述句转成问句（简单规则，够用即可）。"""
    for pattern, repl in [
        ("我的名字叫", "我叫什么"),
        ("我最喜欢", "我最喜欢"),
        ("我在", "我在哪"),
        ("我住在", "我住哪"),
    ]:
        if pattern in fact:
            return repl
    if "。" in fact:
        q = fact.split("。")[0] + "吗"
        return q
    return fact + "吗"


def generate(llm: LLMClient, rounds: int, seed: int,
             dry_run: bool = False) -> dict:
    rng = random.Random(seed)
    plan = build_turn_plan(rng, rounds)
    turns: list[dict] = []
    history: list[tuple[str, str]] = []
    max_history = 12  # 只带最近几轮给 LLM，控成本

    for step, item in enumerate(plan):
        user_text = item["text"]
        turn = {"role": "user", "text": user_text,
                "kind": item["kind"],
                "fact_id": item.get("fact_id")}
        if item["kind"] == "correction":
            turn["new"] = item["new"]  # 最新说法（考问期望答案）
        turns.append(turn)
        if dry_run and step >= 10:
            break
        # 助手回复（真实 LLM；dry-run 也照发，用于记账）——必须带上当前
        # 这轮用户消息，否则助手答非所问
        hist_lines = [f"{'小林' if r == 'user' else '助手'}: {t}"
                      for r, t in history[-max_history:]]
        hist_lines.append(f"小林: {user_text}")
        prompt = ASSISTANT_PROMPT.format(history="\n".join(hist_lines) or "（开始）")
        try:
            reply = llm.chat([{"role": "user", "content": prompt}],
                             temperature=0.8, max_tokens=200)
        except Exception as exc:  # noqa: BLE001
            reply = f"（LLM 生成失败：{exc}）"
        turns.append({"role": "assistant", "text": reply})
        history.append(("user", user_text))
        history.append(("assistant", reply))

    return {
        "persona": PERSONA,
        "seed": seed,
        "rounds": len(turns) // 2,
        "facts": [{"id": f[0], "text": f[1], "topic": f[2]} for f in FACTS],
        "turns": turns,
        "llm": llm.summary(),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="生成几百轮真实对话")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--out", default="data/dialogue_300.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="只跑 10 轮记账")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    llm = LLMClient()
    if not llm.configured:
        print("✗ 未设置 SME_LLM_API_KEY（环境变量），无法生成")
        sys.exit(1)
    print(f"LLM: {llm.base_url} / {llm.model}")
    rounds = 10 if args.dry_run else args.rounds
    data = generate(llm, rounds, args.seed, dry_run=args.dry_run)
    out = Path(__file__).resolve().parents[1] / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"✓ 已生成 {data['rounds']} 轮 → {out}")
    print(f"  LLM 调用 {llm.calls} 次，token {llm.total_usage['total_tokens']}，"
          f"估算成本 ¥{llm.cost_estimate_yuan()}")


if __name__ == "__main__":
    main()
