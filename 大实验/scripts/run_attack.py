"""攻击题专项评测：9 选手 × 13 攻击题（含双重纠错场景）。

只回放 1 个 seed 的对话 + 攻击题集，轻量快跑。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import ALL_ADAPTERS  # noqa: E402
from baselines.embedder import get_embedder  # noqa: E402
from eval_qa import QAEvaluator  # noqa: E402
from llm_client import LLMClient  # noqa: E402

root = Path(__file__).resolve().parents[1]
SEED = "20260809"
dialogue = json.loads((root / "data" / f"dialogue_{SEED}.json").read_text(encoding="utf-8"))
attack = json.loads((root / "data" / "tasks_attack.json").read_text(encoding="utf-8"))["tasks"]

llm = LLMClient()
if not llm.configured:
    print("✗ 未设置 SME_LLM_API_KEY")
    sys.exit(1)
embedder = get_embedder()
ev = QAEvaluator(llm, attack)

report = {}
for name in ALL_ADAPTERS:
    print(f"[attack] {name} ...")
    ws = root / "results" / f"attack_ws_{SEED}"
    try:
        a = ALL_ADAPTERS[name](workspace=str(ws), embedder=embedder)
        for turn in dialogue["turns"]:
            a.store(turn["text"], role=turn["role"])
        r = ev.eval_adapter(a, tag=name)
        report[name] = {"accuracy": r["accuracy"], "n": r["n"],
                        "led_astray": r["led_astray"],
                        "details": r["details"]}
    except Exception as exc:  # noqa: BLE001
        report[name] = {"error": str(exc)}
        print(f"  ✗ {name}: {exc}")

out = root / "results" / "attack_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"✓ → {out}")

# 按攻击类型汇总
groups = {"dialect": [], "double": [], "rel": [], "noise": []}
for name, r in report.items():
    if "details" not in r:
        continue
    for det in r["details"]:
        tid = det.get("id", "")
        if tid.startswith("attack_"):
            key = tid.split("_")[1]
            if key in groups:
                groups[key].append((name, 1 if det.get("correct") else 0,
                                    det.get("led_astray", False)))
labels = {"dialect": "同义改写", "double": "双重纠错", "rel": "跨话题关联", "noise": "噪音霸榜"}
for key, label in labels.items():
    print(f"\n== {label}（{len(groups[key]) // len(report) if report else 0} 题/选手）==")
    for name in report:
        vals = [v for n, v, _ in groups[key] if n == name]
        astray = any(a for n, _, a in groups[key] if n == name)
        if vals:
            flag = " ⚠带偏" if astray else ""
            print(f"  {name:<16} {sum(vals)}/{len(vals)}{flag}")
