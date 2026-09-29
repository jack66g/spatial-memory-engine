"""记忆系统擂台赛——冒烟赛（30 轮对话重放 + 20 题考问，1 seed）。

流程（每选手串行）：
  1. reset() 构造（计时 reinit）
  2. 重放同一份对话流：add(round_text) × 30（逐轮计时）
  3. 逐题考问：search(question, top_k=5)（计时）→ DeepSeek judge 0/1 分
  4. close() 释放；单人崩溃则记录错误继续跑其他选手

judge 协议（只看选手返回的记忆）：
  - correct：仅依据检索到的记忆回答问题，答案是否与标准答案一致
  - misled ：记忆中存在与标准答案冲突的过期/干扰信息且正确信息缺失（被带偏）
  判分失败重试 1 次，再失败该题记 0.5 分并标注 judge_failed。

输出 lab/results/battle_smoke.json + 控制台排名表；工作区跑完即删。

Usage: python lab/scripts/run_battle.py [--only sme_chat,mem0] [--keep-ws]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from baselines import ABSENT, FACTORIES  # noqa: E402
from baselines.base import Contestant  # noqa: E402
from baselines.common import llm_json, load_env, u8  # noqa: E402

ASSETS = LAB_ROOT / "assets"
RESULTS = LAB_ROOT / "results"
WS = RESULTS / "run_battle_ws"


# --------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------- #
def judge_one(question: str, gold: str, hits: list[tuple[str, float]]) -> dict:
    """LLM 判分：返回 {correct, misled, reason}；两级失败后给 0.5 分。"""
    if not hits:
        return {"correct": 0, "misled": 0, "reason": "选手未检索到任何记忆", "empty": True}
    mem_lines = "\n".join(f"[{i + 1}] {t}" for i, (t, _s) in enumerate(hits))
    prompt = (
        "你是记忆系统评测裁判。选手系统针对问题检索到以下记忆（按相关性降序，可能包含过期或干扰信息）：\n"
        f"{mem_lines}\n\n"
        f"问题：{question}\n"
        f"标准答案：{gold}\n\n"
        "判定规则：\n"
        "1. correct：假设 AI 只依据上述记忆回答问题，答案是否与标准答案语义一致（1/0）。"
        "记忆中出现与标准答案等价的信息（措辞可不同）即为 1。\n"
        "2. misled：上述记忆是否会把回答带偏——存在与标准答案冲突的过期说法或干扰信息，"
        "且正确信息缺失、无法从记忆中辨认出最新说法（1/0）。若新旧说法并存但最新正确说法清晰在场，correct=1 且 misled=0。\n"
        "只输出 JSON：{\"correct\": 0, \"misled\": 0, \"reason\": \"不超过30字\"}"
    )
    verdict = llm_json(
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=200, timeout=90.0,
    )
    return {
        "correct": int(verdict["correct"]),
        "misled": int(verdict.get("misled", 0)),
        "reason": str(verdict.get("reason", ""))[:60],
    }


def judge_with_retry(question: str, gold: str, hits: list) -> dict:
    last_err = None
    for _ in range(2):  # 首次 + 重试 1 次
        try:
            return judge_one(question, gold, hits)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3)
    return {"correct": 0.5, "misled": 0, "reason": f"judge_failed: {last_err}", "judge_failed": True}


# --------------------------------------------------------------------- #
# 单选手
# --------------------------------------------------------------------- #
def run_contestant(name: str, factory, rounds: list[dict], questions: list[dict]) -> dict:
    rec: dict = {"status": "failed", "error": None}
    c: Contestant | None = None
    try:
        print(u8(f"\n=== [{name}] 构造中 ..."), flush=True)
        c = factory(WS / name)
        rec["reinit_s"] = round(c.reinit_cost_s, 1)

        # --- 重放对话 --- #
        t0 = time.time()
        add_times: list[float] = []
        for r in rounds:
            ta = time.time()
            c.add(r["text"])
            add_times.append(time.time() - ta)
            print(u8(f"  add 第{r['round']:>2}轮 {add_times[-1]*1000:7.0f}ms"), flush=True)
        rec["add_total_s"] = round(time.time() - t0, 1)
        rec["add_p50_ms"] = round(statistics.median(add_times) * 1000, 1)

        # --- 考问 --- #
        t0 = time.time()
        search_ms: list[float] = []
        details = []
        score = 0.0
        misled_n = 0
        for q in questions:
            ts = time.time()
            hits = c.search(q["question"], top_k=5)
            search_ms.append((time.time() - ts) * 1000)
            v = judge_with_retry(q["question"], q["gold"], hits)
            score += v["correct"]
            misled_n += v["misled"]
            details.append({
                "qid": q["id"], "type": q["type"],
                "correct": v["correct"], "misled": v["misled"],
                "reason": v["reason"],
                "hits_n": len(hits),
                "top1": hits[0][0][:80] if hits else "",
                "judge_failed": bool(v.get("judge_failed")),
            })
            print(u8(f"  Q{q['id']:>2} [{q['type']:<10}] correct={v['correct']} "
                     f"misled={v['misled']} ({len(hits)} hits, {search_ms[-1]:.0f}ms) {v['reason']}"),
                  flush=True)
        rec.update({
            "status": "ok",
            "score": score,
            "acc": round(score / len(questions), 3),
            "misled": misled_n,
            "storage": c.count(),
            "retrieval_p50_ms": round(statistics.median(search_ms), 1),
            "quiz_total_s": round(time.time() - t0, 1),
            "total_s": round(rec["reinit_s"] + rec["add_total_s"] + (time.time() - t0), 1),
            "details": details,
        })
    except Exception as e:  # noqa: BLE001 - 单选手崩了记录后继续
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-3:]
        print(u8(f"  !!! [{name}] 崩溃: {rec['error']}"), flush=True)
    finally:
        if c is not None:
            try:
                cnt = c.count()
                rec.setdefault("storage", cnt)
            except Exception:  # noqa: BLE001
                pass
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
    return rec


# --------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="", help="逗号分隔的选手名子集")
    ap.add_argument("--keep-ws", action="store_true", help="保留工作区（调试用）")
    ap.add_argument("--out", type=str, default="battle_smoke.json")
    args = ap.parse_args()

    load_env()
    dialogue = json.loads((ASSETS / "dialogue.json").read_text(encoding="utf-8"))
    quiz = json.loads((ASSETS / "quiz.json").read_text(encoding="utf-8"))
    rounds, questions = dialogue["rounds"], quiz["questions"]

    order = [n for n in FACTORIES if not args.only or n in args.only.split(",")]
    print(u8(f"[battle] 选手 {order} | 缺席 {list(ABSENT)} | 对话 {len(rounds)} 轮 / 考问 {len(questions)} 题"),
          flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    if WS.exists():
        shutil.rmtree(WS)
    WS.mkdir(parents=True)

    env = load_env()
    results: dict = {"contestants": {}}
    t_all = time.time()

    # 增量落盘：每选手跑完立即写盘（防止"跑完全程、写盘一步失败全丢"重演——
    # 2026-09-29 seed1 曾因 --out 路径错误整轮白跑）。最终写盘仍保留。
    def _incremental_write() -> None:
        try:
            _p = RESULTS / f"{args.out}.partial"
            _p.write_text(json.dumps(
                {**results, "meta": {"note": "partial（battle 进行中增量落盘）"}},
                ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as _e:  # noqa: BLE001
            print(u8(f"[battle] 增量落盘失败（不影响比赛继续）: {_e}"), flush=True)

    for name in order:
        results["contestants"][name] = run_contestant(name, FACTORIES[name], rounds, questions)
        _incremental_write()

    results["meta"] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": dialogue["seed"],
        "llm_model": env["LAB_LLM_MODEL"],
        "embed": "Qwen/Qwen3-Embedding-0.6B (1024d, local CPU)",
        "dialogue_rounds": len(rounds),
        "quiz_questions": len(questions),
        "absent": ABSENT,
        "wall_total_s": round(time.time() - t_all, 1),
    }

    # 排名：acc 降序 → misled 升序 → 检索 p50 升序
    ok = [(n, r) for n, r in results["contestants"].items() if r.get("status") == "ok"]
    ranking = [n for n, _ in sorted(
        ok, key=lambda kv: (-kv[1]["acc"], kv[1]["misled"], kv[1]["retrieval_p50_ms"]))]
    results["ranking"] = ranking

    out_path = RESULTS / args.out
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # 最终结果落盘成功，清理增量 partial 文件
    _partial = RESULTS / f"{args.out}.partial"
    if _partial.exists():
        _partial.unlink()

    # ------------------ 排名表 ------------------ #
    print(u8("\n") + u8("=" * 88), flush=True)
    print(u8(f"  冒烟赛排名  （{len(ok)} 位完赛 / {len(order)} 位出场 | judge={env['LAB_LLM_MODEL']}）"),
          flush=True)
    print(u8("=" * 88), flush=True)
    hdr = f"  {'#':<3}{'选手':<16}{'acc':>7}{'被带偏':>7}{'存储':>7}{'检索p50':>10}{'add总s':>9}{'总s':>8}"
    print(u8(hdr), flush=True)
    print(u8("  " + "-" * 84), flush=True)
    for i, n in enumerate(ranking, 1):
        r = results["contestants"][n]
        print(u8(f"  {i:<3}{n:<16}{r['acc']:>7.1%}{r['misled']:>7}"
                 f"{r['storage']:>7}{r['retrieval_p50_ms']:>8.0f}ms"
                 f"{r['add_total_s']:>9.1f}{r['total_s']:>8.1f}"), flush=True)
    for n, r in results["contestants"].items():
        if r.get("status") != "ok":
            print(u8(f"  ✗   {n:<16} 失败: {r.get('error')}"), flush=True)
    for n, why in ABSENT.items():
        print(u8(f"  -   {n:<16} 缺席: {why}"), flush=True)
    print(u8("=" * 88), flush=True)
    print(u8(f"[battle] 结果已写入 {out_path}"), flush=True)

    if not args.keep_ws and WS.exists():
        shutil.rmtree(WS)
        print(u8("[battle] 工作区已清理"), flush=True)


if __name__ == "__main__":
    main()
