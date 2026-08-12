"""第二轮擂台赛：3 seed 对话回放 + 端到端问答 + 公开知识库评测 + 一致性。

选手（9 位，letta 为实验性未完整参赛）：SME×5 / RAG / BM25 / mem0 / langmem。
知识库场景：同一份公开语料（民法典 1381 条款 + 医疗 1875 句）喂给所有选手；
mem0/langmem 知识库导入跳过 LLM 提取（直接存原文，公平）。

用法::

    python scripts/run_battle2.py --seeds 20260809,20260810,20260811
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # SME 插件根
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import ALL_ADAPTERS  # noqa: E402
from baselines.embedder import get_embedder  # noqa: E402
from eval_qa import QAEvaluator  # noqa: E402
from llm_client import LLMClient  # noqa: E402

DIALOGUE_TASKS = "data/tasks_dialogue.json"
ATTACK_TASKS = "data/tasks_attack.json"
KB_TASKS = "data/tasks_kb.json"
LAW_CHUNKS = "data/law/law_chunks.json"
MEDICAL_CHUNKS = "data/medical/medical_chunks.json"


def load(root: Path, rel: str):
    return json.loads((root / rel).read_text(encoding="utf-8"))


def import_kb(adapter, chunks: list[dict], tag: str) -> None:
    """把语料 chunk 导入选手。

    修正（2026-08-09）：SME 选手必须原文直存（import_memories 绕过 v2
    提取/纠错管线）——知识库条款是权威原文，不该被对话记忆的提取/纠错/
    问答对机制处理（旧实现走 engine.add 导致 LLM 提取空转、factversion
    相似条款误合并，43 条条款丢失）。其余选手 store_raw 直存。
    """
    name = adapter.name
    if name.startswith("sme_"):
        items = []
        for c in chunks:
            text = c["text"].strip()
            if not text:
                continue
            meta = {"来源": "民法典" if tag == "law"
                    else c.get("source", "医学")}
            if tag == "law" and c.get("clause"):
                meta["条款号"] = c["clause"]
            items.append({"text": text, "metadata": meta, "tags": ["doc"]})
        if items:
            adapter.engine.import_memories(items)
    else:
        for c in chunks:
            adapter.store_raw(c["text"])


def run_dialogue_seed(llm, tasks, attack, dialogue, adapter, workspace, embedder):
    """回放 + 对话端到端评测。"""
    a = ALL_ADAPTERS[adapter](workspace=str(workspace), embedder=embedder)
    t0 = time.perf_counter()
    for turn in dialogue["turns"]:
        a.store(turn["text"], role=turn["role"])
    replay_s = time.perf_counter() - t0
    ev = QAEvaluator(llm, tasks + attack)
    result = ev.eval_adapter(a, tag=adapter)
    return {"replay_s": round(replay_s, 2), **result}


def run_kb(llm, tasks, chunks_law, chunks_med, adapter, workspace, embedder):
    """知识库场景：导入公开语料 + 条款考问（独立实例）。"""
    a = ALL_ADAPTERS[adapter](workspace=str(workspace / "kb"), embedder=embedder)
    import_kb(a, chunks_law, "law")
    import_kb(a, chunks_med, "med")
    ev = QAEvaluator(llm, tasks)
    result = ev.eval_adapter(a, tag=adapter + "_kb")
    return result


def run_consistency(llm, adapter, workspace, embedder, n=10, repeats=3):
    a = ALL_ADAPTERS[adapter](workspace=str(workspace / "cons"), embedder=embedder)
    for turn in CONSISTENCY_TURNS:
        a.store(turn, role="user")
    ev = QAEvaluator(llm, [])
    out = []
    for q in CONSISTENCY_QS[:n]:
        out.append(ev.consistency(a, q, repeats=repeats))
    return out


CONSISTENCY_TURNS = [
    "我的名字叫小林，我住在杭州，在一家互联网公司做前端工程师",
    "我养了一只金毛，叫可乐，我最近在学 Rust",
    "我周末经常和朋友打篮球，我女朋友叫小雨，是设计师",
]
CONSISTENCY_QS = [
    "小林养了什么宠物", "小林住在哪里", "小林周末喜欢做什么",
    "小林最近在学什么", "小林女朋友是谁",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="第二轮擂台赛")
    parser.add_argument("--seeds", default="20260809,20260810,20260811")
    parser.add_argument("--adapters", default="",
                        help="逗号分隔；空 = 全部可用")
    parser.add_argument("--out", default="results/battle2.json")
    parser.add_argument("--kb-only", action="store_true")
    parser.add_argument("--skip-kb", action="store_true")
    parser.add_argument("--skip-cons", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="增量写入 --out（断点续跑）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    names = [n.strip() for n in args.adapters.split(",") if n.strip()]
    if not names:
        names = list(ALL_ADAPTERS)
    embedder = get_embedder()
    llm = LLMClient()
    if not llm.configured:
        print("✗ 未设置 SME_LLM_API_KEY")
        sys.exit(1)

    tasks_d = load(root, DIALOGUE_TASKS)["tasks"]
    tasks_a = load(root, ATTACK_TASKS)["tasks"]
    tasks_kb = load(root, KB_TASKS)["tasks"]
    chunks_law = load(root, LAW_CHUNKS)["chunks"]
    chunks_med = load(root, MEDICAL_CHUNKS)["chunks"]

    out_path = root / args.out
    if args.resume and out_path.exists():
        report = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"（续跑：已有 {len(report.get('results', {}))} 个选手结果）")
    else:
        report = {"seeds": seeds, "adapters": names, "results": {}}

    def flush() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    # ---------- 对话场景（每 seed 独立实例） ---------- #
    if not args.kb_only:
        for seed in seeds:
            dlg_path = root / "data" / f"dialogue_{seed}.json"
            if not dlg_path.exists():
                print(f"✗ 缺少对话 {dlg_path}（先运行 gen_dialogue.py）")
                continue
            dialogue = json.loads(dlg_path.read_text(encoding="utf-8"))
            ws = root / "results" / f"battle2_ws_{seed}"
            for name in names:
                key = f"d_{seed}"
                if key in report["results"].get(name, {}):
                    print(f"  - {name} {key} 已存在，跳过")
                    continue
                print(f"[{seed}] {name} 回放 {len(dialogue['turns'])} 轮...")
                try:
                    r = run_dialogue_seed(llm, tasks_d, tasks_a, dialogue,
                                          name, ws, embedder)
                    report["results"].setdefault(name, {})[key] = {
                        "accuracy": r["accuracy"], "n": r["n"],
                        "led_astray": r["led_astray"],
                        "replay_s": r["replay_s"],
                        "details": r["details"],
                    }
                except Exception as exc:  # noqa: BLE001
                    print(f"  ✗ {name}: {exc}")
                    report["results"].setdefault(name, {})[key] = {"error": str(exc)}
                flush()

    # ---------- 知识库场景（语料固定，一次） ---------- #
    if not args.skip_kb:
        ws = root / "results" / "battle2_ws_kb"
        for name in names:
            if "kb" in report["results"].get(name, {}):
                print(f"  - {name} kb 已存在，跳过")
                continue
            print(f"[kb] {name} 导入 {len(chunks_law) + len(chunks_med)} 条...")
            try:
                r = run_kb(llm, tasks_kb, chunks_law, chunks_med,
                           name, ws, embedder)
                report["results"].setdefault(name, {})["kb"] = {
                    "accuracy": r["accuracy"], "n": r["n"],
                    "led_astray": r["led_astray"],
                    "details": r["details"],
                }
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ {name}: {exc}")
                report["results"].setdefault(name, {})["kb"] = {"error": str(exc)}
            flush()

    # ---------- 一致性（一次） ---------- #
    if not args.kb_only and not args.skip_cons:
        ws = root / "results" / "battle2_ws_cons"
        for name in names:
            if "consistency" in report["results"].get(name, {}):
                continue
            print(f"[cons] {name} ...")
            try:
                report["results"].setdefault(name, {})["consistency"] = \
                    run_consistency(llm, name, ws, embedder)
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ {name}: {exc}")
            flush()

    # ---------- 汇总表 ---------- #
    print("\n" + "=" * 70)
    print(f"{'基线':<16}{'对话acc':>9}{'对话±':>7}{'kb acc':>8}"
          f"{'被带偏':>6}{'一致率':>7}")
    for name in names:
        r = report["results"].get(name, {})
        d_accs = [r[k]["accuracy"] for k in r if k.startswith("d_") and "accuracy" in r[k]]
        d_acc = sum(d_accs) / len(d_accs) if d_accs else float("nan")
        d_spread = (max(d_accs) - min(d_accs)) if len(d_accs) > 1 else 0.0
        kb = r.get("kb", {})
        kb_acc = kb.get("accuracy", float("nan"))
        astray = max((r[k].get("led_astray", 0) for k in r if k.startswith("d_")), default=0)
        cons = r.get("consistency", [])
        cons_avg = (sum(c["consistency"] for c in cons) / len(cons)
                    if cons else float("nan"))
        print(f"{name:<16}{d_acc:>9.3f}{d_spread:>7.3f}{kb_acc:>8.3f}"
              f"{astray:>6}{cons_avg:>7.3f}")
    print("=" * 70)

    flush()
    print(f"✓ 报告 → {out_path}")


if __name__ == "__main__":
    main()
