# -*- coding: utf-8 -*-
"""Region 消融实验：只关 Region、其余全留，量化 Region 对准确率的净贡献。

消融臂不修改任何引擎源码，继承 ``baselines.sme_presets.SmeContestant``
（chat 预设原样构建），引擎建好后仅改四个纯配置项：

  - ``retrieval.top_regions = 10**9``   区域闸门失效：全体记忆进候选（=全局检索）
  - ``retrieval.region_dampening = 0``  无"区域外记忆"惩罚（闸门失效后不再触发）
  - ``ranking.region = 0`` + 其余七信号等比归一化（保持权重和 = 1.0 的引擎约束）
  - ``region.auto_evolve = False``      无区域动态演化

其余一切（Embedding / BM25 / 八信号其余项 / 命中强化 / 衰减 / 事实版本管理 /
整理压缩节奏）与对照臂逐配置一致；对照臂 = 官方 ``make_sme_chat`` 原样，
同时用于复现已发布擂台战绩、验证装置可信。

协议完全复用 ``run_battle``（同一对话流、40 题、LLM 盲判、逐题明细），
规模赛阶段复刻 ``battle_scale`` 的对话→干扰→考问流程。

用法::

    python lab/scripts/ablation.py --stage plain   # 单 seed 平地赛两臂（装置验证）
    python lab/scripts/ablation.py --stage full    # 3 seed × 2 臂 × (0/+500/+2000)

资产按 seed 重生成（``gen_assets.py --seed N``，只写 gitignore 的 assets）；
结果落 ``lab/results/ablation_*.json``，实验工作区 ``lab/results/ablation_ws``。
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_ROOT.parent
for _p in (str(Path(__file__).resolve().parent), str(LAB_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_battle  # noqa: E402  (模块头自带 baselines/sme 的 sys.path 装配)
from baselines import FACTORIES  # noqa: E402
from baselines.common import u8  # noqa: E402
from baselines.sme_presets import SmeContestant  # noqa: E402

RESULTS = LAB_ROOT / "results"
WS_ROOT = RESULTS / "ablation_ws"
DISTRACTORS = LAB_ROOT / "assets" / "distractors.json"

# 实验工作区重定向：run_contestant 用模块级 WS，勿写擂台自己的 run_battle_ws
run_battle.WS = WS_ROOT


# ------------------------------------------------------------------ #
# 消融选手：chat 预设 + 仅关 Region（构造后原位改配置，不碰源码）
# ------------------------------------------------------------------ #
class SmeRegionOff(SmeContestant):
    def _build_engine(self):
        eng = super()._build_engine()  # 与官方 chat 臂完全一致的构建路径
        eng.config.retrieval.top_regions = 10**9
        eng.config.retrieval.region_dampening = 0.0
        eng.config.region.auto_evolve = False
        rk = eng.config.ranking
        w = rk.region
        rk.region = 0.0
        rest = (rk.semantic + rk.importance + rk.freshness + rk.weight
                + rk.decay + rk.hit_count + rk.recency)
        scale = (rest + w) / rest  # 其余七信号等比放大，保持 sum=1.0
        for f in ("semantic", "importance", "freshness", "weight",
                  "decay", "hit_count", "recency"):
            setattr(rk, f, round(getattr(rk, f) * scale, 4))
        return eng


def make_region_off(workspace: Path) -> SmeRegionOff:
    c = SmeRegionOff.__new__(SmeRegionOff)
    SmeContestant.__init__(c, "chat", workspace)  # preset 仍为 chat；外层用别名记录
    return c


ARMS = {
    "sme_chat": FACTORIES["sme_chat"],   # 对照：官方原样
    "sme_region_off": make_region_off,    # 消融：仅关 Region
}


# ------------------------------------------------------------------ #
# 规模赛跑法（复刻 battle_scale 协议：对话 → 干扰 → 考问）
# ------------------------------------------------------------------ #
def run_scale_arm(name: str, factory, rounds, questions, distractors, scale: int) -> dict:
    import shutil
    import traceback

    rec: dict = {"status": "failed", "scale": scale}
    c = None
    try:
        t0 = time.time()
        ws = WS_ROOT / f"{name}_s{scale}"
        if ws.exists():
            shutil.rmtree(ws)
        c = factory(ws)
        rec["reinit_s"] = round(c.reinit_cost_s, 1)

        t0 = time.time()
        for r in rounds:
            c.add(r["text"])
        rec["dialogue_add_s"] = round(time.time() - t0, 1)

        if scale > 0:
            t0 = time.time()
            add_ms = []
            for d in distractors[:scale]:
                ta = time.time()
                c.add(d)
                add_ms.append((time.time() - ta) * 1000)
            rec["distractor_add_s"] = round(time.time() - t0, 1)
            rec["distractor_add_ms_p50"] = round(statistics.median(add_ms), 1)

        t0 = time.time()
        search_ms, details = [], []
        score, misled_n = 0.0, 0
        for q in questions:
            ts = time.time()
            hits = c.search(q["question"], top_k=5)
            search_ms.append((time.time() - ts) * 1000)
            v = run_battle.judge_with_retry(q["question"], q["gold"], hits)
            score += v["correct"]
            misled_n += v["misled"]
            details.append({"qid": q["id"], "type": q["type"],
                            "correct": v["correct"], "misled": v["misled"],
                            "reason": v["reason"], "top1": hits[0][0][:80] if hits else ""})
            print(u8(f"  [{name}|s{scale}] Q{q['id']:>2} correct={v['correct']} "
                     f"misled={v['misled']} ({search_ms[-1]:.0f}ms)"), flush=True)
        rec.update({"status": "ok", "score": score,
                    "acc": round(score / len(questions), 3), "misled": misled_n,
                    "storage": c.count(),
                    "retrieval_p50_ms": round(statistics.median(search_ms), 1),
                    "quiz_total_s": round(time.time() - t0, 1),
                    "details": details})
    except Exception as e:  # noqa: BLE001 - 单臂崩溃记录后继续
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-3:]
        print(u8(f"  !!! [{name}|s{scale}] 崩溃: {rec['error']}"), flush=True)
    finally:
        if c is not None:
            try:
                rec.setdefault("storage", c.count())
            except Exception:  # noqa: BLE001
                pass
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
    return rec


# ------------------------------------------------------------------ #
def load_assets():
    dialogue = json.loads((LAB_ROOT / "assets" / "dialogue.json").read_text(encoding="utf-8"))
    quiz = json.loads((LAB_ROOT / "assets" / "quiz.json").read_text(encoding="utf-8"))
    return dialogue["rounds"], quiz["questions"], dialogue.get("seed")


def gen_assets(seed: int) -> None:
    """按 seed 重生成实验资产（只写 gitignore 的 lab/assets/*.json）。"""
    subprocess.run([sys.executable, str(LAB_ROOT / "scripts" / "gen_assets.py"),
                    "--seed", str(seed)], check=True, cwd=str(REPO_ROOT))


def out_write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(u8(f"[out] {path}"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["plain", "full"], default="plain")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    WS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stage == "plain":
        rounds, questions, seed = load_assets()
        print(u8(f"[stage plain] seed={seed} 对话 {len(rounds)} 轮 / 考问 {len(questions)} 题"), flush=True)
        data = {"meta": {"seed": seed, "stage": "plain",
                         "note": "对照=官方chat原样；消融=仅关Region(4配置项)"}}
        for name, fac in ARMS.items():
            data[name] = run_battle.run_contestant(name, fac, rounds, questions)
            gc.collect()
        out_write(data, RESULTS / f"ablation_plain_seed{seed}.json")
        return

    # full：3 seed × 2 臂 × (0 / +500 / +2000)
    distractors = json.loads(DISTRACTORS.read_text(encoding="utf-8"))
    if isinstance(distractors, dict):
        distractors = distractors.get("distractors") or distractors.get("items")
    summary = {}
    for seed in (3, 1, 2):
        gen_assets(seed)
        rounds, questions, _ = load_assets()
        for scale in (0, 500, 2000):
            data = {"meta": {"seed": seed, "scale": scale}}
            for name, fac in ARMS.items():
                rec = (run_battle.run_contestant(name, fac, rounds, questions)
                       if scale == 0 else
                       run_scale_arm(name, fac, rounds, questions, distractors, scale))
                data[name] = rec
                summary[f"seed{seed}_s{scale}_{name}"] = {
                    "acc": rec.get("acc"), "misled": rec.get("misled"),
                    "status": rec.get("status")}
                gc.collect()
            out_write(data, RESULTS / f"ablation_seed{seed}_s{scale}.json")
            out_write({"summary": summary}, RESULTS / "_ablation_summary.json")
    print(u8("[full done] " + json.dumps(summary, ensure_ascii=False)), flush=True)


if __name__ == "__main__":
    main()
