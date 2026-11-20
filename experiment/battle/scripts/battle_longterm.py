# -*- coding: utf-8 -*-
"""补充赛 B：长程强化赛——考察"重复使用后谁越来越准"。

协议：同一对话流回放 P 遍（模拟多次会话重复提到相同事实），每遍结束后考问
同一套题。SME chat 预设检索命中即强化（Ebbinghaus）且按周期融合去重；
RAG 无任何此类机制，重复回放只会堆积重复向量。

核心指标：第 1 遍 → 第 P 遍的 acc 变化（学习曲线）、存储条数（谁在重复
对话下膨胀）、以及"首遍答错→末遍答对"的翻正率。

    python lab/scripts/battle_longterm.py --passes 3
输出：lab/results/battle_longterm.json
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
from baselines.common import load_env  # noqa: E402
from run_battle import judge_with_retry  # noqa: E402

load_env()
ASSETS = LAB / "assets"
RESULTS = LAB / "results"
WS = RESULTS / "battle_longterm_ws"
CONTESTANTS = ["sme_chat", "rag_qwen3"]


def quiz_pass(c, questions: list[dict]) -> dict:
    lat, details, score, misled = [], [], 0.0, 0
    for q in questions:
        ts = time.perf_counter()
        hits = c.search(q["question"], top_k=5)
        lat.append((time.perf_counter() - ts) * 1000)
        v = judge_with_retry(q["question"], q["gold"], hits)
        score += v["correct"]
        misled += v["misled"]
        details.append({"qid": q["id"], "type": q["type"],
                        "correct": v["correct"], "misled": v["misled"]})
    return {"acc": round(score / len(questions), 4), "misled": misled,
            "retrieval_p50_ms": round(statistics.median(lat), 1),
            "storage": c.count(), "details": details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    os.system(f'"{sys.executable}" "{LAB / "scripts" / "gen_assets.py"}" --seed {a.seed}')
    dialogue = json.loads((ASSETS / "dialogue.json").read_text(encoding="utf-8"))
    quiz = json.loads((ASSETS / "quiz.json").read_text(encoding="utf-8"))
    rounds, questions = dialogue["rounds"], quiz["questions"]

    if WS.exists():
        shutil.rmtree(WS)
    WS.mkdir(parents=True)

    out: dict = {"meta": {"seed": a.seed, "passes": a.passes, "rounds_per_pass": len(rounds),
                          "llm": "deepseek-flash"}, "contestants": {}}
    for name in CONTESTANTS:
        if name not in FACTORIES:
            continue
        print(f"=== [{name}] 回放 {a.passes} 遍 ===", flush=True)
        rec: dict = {"passes": []}
        try:
            c = FACTORIES[name](WS / name)
            first_wrong: set[int] = set()
            for p in range(1, a.passes + 1):
                t0 = time.time()
                for r in rounds:
                    c.add(r["text"])
                res = quiz_pass(c, questions)
                res["add_s"] = round(time.time() - t0, 1)
                rec["passes"].append(res)
                if p == 1:
                    first_wrong = {d["qid"] for d in res["details"] if d["correct"] < 1}
                last = rec["passes"][-1]
                fixed = first_wrong - {d["qid"] for d in last["details"] if d["correct"] < 1}
                print(f"  第{p}遍: acc={res['acc']*100:.1f}% misled={res['misled']} "
                      f"存储={res['storage']} p50={res['retrieval_p50_ms']:.0f}ms "
                      f"（首遍错题已翻正 {len(fixed)}/{len(first_wrong)}）", flush=True)
            final = rec["passes"][-1]["details"]
            rec["recovered_qids"] = sorted(first_wrong - {d["qid"] for d in final if d["correct"] < 1})
            rec["first_wrong_n"] = len(first_wrong)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  失败: {rec['error'][:120]}", flush=True)
        out["contestants"][name] = rec
        try:
            c.close()  # noqa
        except Exception:  # noqa: BLE001
            pass

    (RESULTS / "battle_longterm.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(WS, ignore_errors=True)
    print(f"→ {RESULTS / 'battle_longterm.json'}", flush=True)


if __name__ == "__main__":
    main()
