"""生成评测资产（seed 固定）：lab/assets/dialogue.json + quiz.json。

流程（全部 DeepSeek 生成，temperature 分层）：
  A. 记忆点设计（t=0.8）：10 普通记忆点 + 3 纠错链（旧说法→改口）+ 4 闲聊话题
  B. 对话流（t=0.9）：30 轮用户消息，纠错链旧说法在前段、新说法在后段
  C. 考问集（t=0.4）：20 题 = 13 普通 + 3 同义改写 + 2 双重纠错 + 2 噪音干扰

生成后资产冻结落盘，battle 重跑复用同一份（LLM 波动不影响比赛复现性）。
跑完打印前 5 轮对话 + 前 5 题供人工抽检。

Usage: python lab/scripts/gen_assets.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
sys.path.insert(0, str(LAB_ROOT))    # baselines 包
sys.path.insert(0, str(REPO_ROOT))   # sme 包

from baselines.common import llm_json, load_env, u8  # noqa: E402

ASSETS = LAB_ROOT / "assets"
PERSONA = (
    "用户人设：小夏，22 岁，刚毕业半年后端程序员，坐标成都，独居，养宠物，"
    "性格话痨、爱吐槽，习惯每天跟 AI 助手闲聊几句"
)
TOPIC_POOL = [
    "饮食偏好", "运动习惯", "工作项目", "学习计划", "家庭成员",
    "宠物", "旅行计划", "健康作息", "兴趣爱好", "购物消费",
    "居住环境", "社交朋友",
]


def step_a_facts(rng: random.Random) -> dict:
    topics = rng.sample(TOPIC_POOL, 10)
    prompt = (
        "你是中文记忆系统评测的数据设计师。为日常陪伴场景设计记忆点，"
        "要求每条都具体（含名称/数字/时间/品牌等可考细节），口语中能自然说出。\n"
        f"{PERSONA}\n"
        f"普通记忆点主题必须依次来自：{topics}。\n"
        "再设计 3 个纠错链：用户先说一个具体说法，之后（隔多轮）自然改口成另一个具体说法，"
        "两条都要可考问（例如旧说法'我养了只猫叫团子'，新说法'团子其实是我室友的猫，我自己养的是狗叫煤球'）。\n"
        "最后给 4 句纯闲聊话（只有情绪/寒暄，不含任何可存储事实）。\n"
        "输出严格 JSON（不要多余文字）：\n"
        '{"facts": [{"id": "f01", "topic": "饮食偏好", "content": "..."}], '
        '"chains": [{"id": "c01", "topic": "宠物", "old": "...", "new": "..."}], '
        '"chats": ["...", "...", "...", "..."]}'
    )
    data = llm_json([{"role": "user", "content": prompt}], temperature=0.8, max_tokens=3000)
    assert len(data["facts"]) == 10 and len(data["chains"]) == 3, \
        f"step A shape wrong: {len(data['facts'])} facts / {len(data['chains'])} chains"
    return data


def step_b_dialogue(design: dict, rng: random.Random) -> list[dict]:
    facts_json = json.dumps(design["facts"], ensure_ascii=False)
    chains_json = json.dumps(design["chains"], ensure_ascii=False)
    chats_json = json.dumps(design["chats"], ensure_ascii=False)
    prompt = (
        f"{PERSONA}。把下面的记忆点编成连续 {N_DAYS} 天的每日闲聊：用户每天对 AI 助手说一段话（只要用户说的话，"
        "不要 AI 回复），像微信聊天一样口语化、有情绪、有细节。\n\n"
        f"普通记忆点（每个至少出现一次，分散在不同天）：\n{facts_json}\n\n"
        f"纠错链（旧说法必须出现在第 2-12 轮，新说法/改口必须出现在第 16-28 轮，改口要自然，"
        f"像用户随口更正）：\n{chains_json}\n\n"
        f"纯闲聊话（穿插使用，可微调措辞）：\n{chats_json}\n\n"
        "规则：每轮 1~3 句；同一记忆点不要原文重复出现超过 2 次；"
        f"{N_DAYS} 轮整体像真实生活流（有吐槽、有提问、有分享）。\n"
        f'输出严格 JSON：{{"rounds": [{{"round": 1, "text": "..."}}, ... 共 {N_DAYS} 项]}}'
    )
    data = llm_json([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=8000)
    rounds = data["rounds"]
    assert len(rounds) == N_DAYS, f"expected {N_DAYS} rounds, got {len(rounds)}"
    rounds.sort(key=lambda r: r["round"])
    return rounds


def step_c_quiz(design: dict, rng: random.Random) -> list[dict]:
    pool = (
        [{"id": f["id"], "type": "fact", "content": f["content"]} for f in design["facts"]]
        + [{"id": c["id"], "type": "chain", "content": c["new"], "old": c["old"]}
           for c in design["chains"]]
    )
    prompt = (
        "给中文记忆系统出考问题。素材是用户对 AI 助手说过的记忆点（type=chain 表示用户后来改过口，"
        "content 是最终正确说法，old 是过期说法）。\n\n"
        f"{json.dumps(pool, ensure_ascii=False)}\n\n"
        f"出 {N_QUIZ} 题，构成：\n"
        f"1. normal {int(N_QUIZ*13/20)} 题：直接问记忆点内容（覆盖尽量多的记忆点，chain 的题问最终值）；\n"
        f"2. paraphrase {max(3, int(N_QUIZ*3/20))} 题：换完全不同的措辞/角度问某个 fact（不出现 content 的原文关键词）；\n"
        f"3. correction {max(2, int(N_QUIZ*2/20))} 题：针对 chain，问那个曾被改口的点（gold=最终说法；trap=过期说法）；\n"
        f"4. noise {max(2, int(N_QUIZ*2/20))} 题：问题主体问 fact A，但句中故意夹入另一个记忆点 B 的实体词干扰（trap=B 的内容）。\n"
        "题目必须是中文口语问句（像用户自己问'我那只猫叫啥来着？'），答案明确唯一。\n"
        f'输出严格 JSON：{{"questions": [{{"id": 1, "type": "normal|paraphrase|correction|noise", '
        f'"question": "...", "gold": "...", "source": "f01", "trap": "（仅 correction/noise 有）"}} ... 共 {N_QUIZ} 项]}}'
    )
    data = llm_json([{"role": "user", "content": prompt}], temperature=0.4, max_tokens=6000)
    qs = data["questions"]
    assert len(qs) == N_QUIZ, f"expected {N_QUIZ} questions, got {len(qs)}"
    from collections import Counter
    dist = Counter(q["type"] for q in qs)
    expect = {"normal": int(N_QUIZ*13/20), "paraphrase": max(3, int(N_QUIZ*3/20)),
              "correction": max(2, int(N_QUIZ*2/20)), "noise": max(2, int(N_QUIZ*2/20))}
    expect["normal"] = N_QUIZ - expect["paraphrase"] - expect["correction"] - expect["noise"]
    assert dist == expect, f"dist wrong: {dist} != {expect}"
    return qs


N_DAYS = 60       # 对话轮数（模块级常量，供 prompt/断言共用）
N_QUIZ = 40       # 考问题数


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    load_env()  # fail fast：.env 缺失立刻报错
    ASSETS.mkdir(parents=True, exist_ok=True)

    print(u8(f"[gen_assets] seed={args.seed} 设计记忆点 ..."), flush=True)
    design = step_a_facts(rng)
    print(u8(f"  facts={len(design['facts'])} chains={len(design['chains'])} chats={len(design['chats'])}"), flush=True)

    print(u8(f"[gen_assets] 生成 {N_DAYS} 轮对话流 ..."), flush=True)
    rounds = step_b_dialogue(design, rng)

    print(u8(f"[gen_assets] 生成 {N_QUIZ} 题考问 ..."), flush=True)
    qs = step_c_quiz(design, rng)

    (ASSETS / "dialogue.json").write_text(
        json.dumps({"seed": args.seed, "design": design, "rounds": rounds},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (ASSETS / "quiz.json").write_text(
        json.dumps({"seed": args.seed, "questions": qs},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(u8(f"[gen_assets] 落盘 {ASSETS/'dialogue.json'} / quiz.json"), flush=True)

    # ---------------- 人工抽检输出（硬性验证 2） ---------------- #
    print(u8("\n========== 前 5 轮对话 =========="), flush=True)
    for r in rounds[:5]:
        print(u8(f"  第{r['round']:>2}轮 | {r['text']}"), flush=True)
    print(u8("\n========== 记忆点设计（纠错链） =========="), flush=True)
    for c in design["chains"]:
        print(u8(f"  {c['id']} [{c.get('topic','')}] 旧: {c['old']}"), flush=True)
        print(u8(f"        {'':>4}  新: {c['new']}"), flush=True)
    print(u8("\n========== 前 5 题 =========="), flush=True)
    for q in qs[:5]:
        print(u8(f"  Q{q['id']:>2} [{q['type']}] {q['question']}"), flush=True)
        print(u8(f"        gold: {q['gold']}" + (f" | trap: {q.get('trap','')}" if q.get("trap") else "")), flush=True)

    # ---------------- 简单自检 ---------------- #
    # 纠错链的新旧说法都应在对话里出现过（宽松包含检查，防止对话漏编）
    all_text = "".join(r["text"] for r in rounds)
    miss = []
    for c in design["chains"]:
        for key in ("old", "new"):
            frag = c[key]
            probe = frag[: min(len(frag), 8)]
            if probe not in all_text:
                miss.append(f"{c['id']}.{key}:{probe}")
    if miss:
        print(u8(f"\n[warn] 纠错链片段未在对话中原样出现（改写属正常，抽查确认即可）: {miss}"), flush=True)
    print(u8("\n[gen_assets] done"), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(u8(f"[gen_assets] total {time.time()-t0:.1f}s"))
