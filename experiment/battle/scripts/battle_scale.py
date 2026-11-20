# -*- coding: utf-8 -*-
"""补充赛 A：规模干扰赛——考察"库变大后谁不退化"。

协议：同一对话流（默认 seed 1，与正赛同资产）入库后，再注入 N 条无关干扰
记忆，然后照常考问。干扰记忆由 LLM 生成一次后缓存（assets/distractors.json）。
scale=0 档直接复用正赛 battle_seed1.json（同协议同资产）。

    python lab/scripts/battle_scale.py                    # 跑 500/2000 两档
    python lab/scripts/battle_scale.py --scales 500
    python lab/scripts/battle_scale.py --gen-distractors  # 只生成干扰语料

选手：sme_chat / rag_qwen3 / bm25_bigram（mem0/kb_dynamic 逐条 LLM 抽取消耗
过大、graphiti 实体抽取过慢，规模档缺席并记录原因）。
输出：lab/results/battle_scale.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parent))  # 仓库根：baselines 要 import sme
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "scripts"))

from baselines import FACTORIES  # noqa: E402
from baselines.common import load_env, llm_json  # noqa: E402
from run_battle import judge_with_retry  # noqa: E402

load_env()
ASSETS = LAB / "assets"
RESULTS = LAB / "results"
WS = RESULTS / "battle_scale_ws"
DISTRACTORS = ASSETS / "distractors.json"

CONTESTANTS = ["sme_chat", "rag_qwen3", "bm25_bigram"]
DISTRACTOR_TOPICS = [
    "数据库内核", "明代家具", "贝类分类学", "地铁信号系统", "藏传佛教艺术",
    "工业机器人", "咖啡烘焙", "合同法", "航空发动机", "考古地层学",
]


def gen_distractors(n: int = 2000) -> list[str]:
    if DISTRACTORS.exists():
        return json.loads(DISTRACTORS.read_text(encoding="utf-8"))
    out: list[str] = []
    batch = 40
    for i in range(0, n, batch):
        topic = DISTRACTOR_TOPICS[(i // batch) % len(DISTRACTOR_TOPICS)]
        data = llm_json([{"role": "user", "content": (
            f"围绕主题「{topic}」编 {batch} 条互不重复的中文短事实句（每条 10~25 字，"
            "像用户随口提过的专业知识/个人经历，不要出现：成都、猫、狗、直播、"
            "篮球、乐高、日料、火锅、地铁通勤、护士、绵阳、旅游改签相关内容）。"
            f'输出严格 JSON：{{"facts": ["...", ...]}} 共 {batch} 条'
        )}], temperature=0.8, max_tokens=4000)
        out.extend(data["facts"])
        print(f"  干扰语料 {len(out)}/{n}", flush=True)
    out = list(dict.fromkeys(out))[:n]
    DISTRACTORS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def run_scale(scale: int, rounds: list[dict], questions: list[dict],
              distractors: list[str]) -> dict:
    ws_root = WS / f"scale{scale}"
    if ws_root.exists():
        shutil.rmtree(ws_root)
    ws_root.mkdir(parents=True)
    out: dict = {"scale": scale, "results": {}}
    for name in CONTESTANTS:
        if name not in FACTORIES:
            out["results"][name] = {"status": "absent"}
            continue
        print(f"--- [{name}] scale={scale} ---", flush=True)
        rec: dict = {"status": "failed"}
        try:
            c = FACTORIES[name](ws_root / name)
            t0 = time.time()
            for r in rounds:
                c.add(r["text"])
            rec["dialogue_add_s"] = round(time.time() - t0, 1)
            t0 = time.time()
            for d in distractors[:scale]:
                c.add(d)
            rec["distractor_add_s"] = round(time.time() - t0, 1)
            rec["distractor_add_ms_p50"] = round(
                rec["distractor_add_s"] / max(1, scale) * 1000, 1)

            lat, details, score, misled = [], [], 0.0, 0
            for q in questions:
                ts = time.perf_counter()
                hits = c.search(q["question"], top_k=5)
                lat.append((time.perf_counter() - ts) * 1000)
                v = judge_with_retry(q["question"], q["gold"], hits)
                score += v["correct"]
                misled += v["misled"]
                details.append({"qid": q["id"], "type": q["type"], **v})
                print(f"  Q{q['id']:>2} [{q['type']:<10}] correct={v['correct']} "
                      f"({lat[-1]:.0f}ms)", flush=True)
            rec.update({
                "status": "ok",
                "acc": round(score / len(questions), 4),
                "misled": misled,
                "storage": c.count(),
                "retrieval_p50_ms": round(statistics.median(lat), 1),
                "retrieval_p95_ms": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1),
                "details": details,
            })
            print(f"  => acc={rec['acc']*100:.1f}% misled={misled} "
                  f"p50={rec['retrieval_p50_ms']:.0f}ms p95={rec['retrieval_p95_ms']:.0f}ms "
                  f"存储={rec['storage']}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  => 失败: {rec['error'][:120]}", flush=True)
        out["results"][name] = rec
        try:
            c.close()  # noqa
        except Exception:  # noqa: BLE001
            pass
    if WS.exists():
        shutil.rmtree(WS, ignore_errors=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="500,2000")
    ap.add_argument("--gen-distractors", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    scales = [int(x) for x in a.scales.split(",")]

    # 重新生成对应 seed 的资产（注意：不可与正赛并发跑，会互相覆盖 assets）
    os.system(f'"{sys.executable}" "{LAB / "scripts" / "gen_assets.py"}" --seed {a.seed}')
    dialogue = json.loads((ASSETS / "dialogue.json").read_text(encoding="utf-8"))
    quiz = json.loads((ASSETS / "quiz.json").read_text(encoding="utf-8"))
    rounds, questions = dialogue["rounds"], quiz["questions"]

    if a.gen_distractors:
        gen_distractors()
        return
    distractors = gen_distractors()
    print(f"干扰语料 {len(distractors)} 条 | scale 档 {scales} | 选手 {CONTESTANTS}", flush=True)

    runs = []
    for s in scales:
        runs.append(run_scale(s, rounds, questions, distractors))

    out_path = RESULTS / "battle_scale.json"
    base = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    base["meta"] = {"seed": a.seed, "scales_run": scales, "llm": "deepseek-flash",
                    "absent": {"mem0/kb_dynamic": "逐条 LLM 抽取消耗过大",
                               "graphiti": "实体抽取过慢（~20s/条）"},
                    "note": "scale=0 档复用 battle_seed1.json"}
    base.setdefault("runs", []).extend(runs)
    out_path.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
