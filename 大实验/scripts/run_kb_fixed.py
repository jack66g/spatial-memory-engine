"""知识库评测（修正版）：SME 原文直存绕过 v2 管线，9 选手同语料。

输出 results/battle2_kb_fixed.json（含每选手记忆条数与条款号完整性）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import ALL_ADAPTERS  # noqa: E402
from baselines.embedder import get_embedder  # noqa: E402
from eval_qa import QAEvaluator  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from run_battle2 import load, import_kb  # noqa: E402

root = Path(__file__).resolve().parents[1]
KB_TASKS = "data/tasks_kb.json"
LAW_CHUNKS = "data/law/law_chunks.json"
MEDICAL_CHUNKS = "data/medical/medical_chunks.json"


def clause_count(engine) -> int:
    """引擎内条款号去重数（验证条款完整性）。"""
    try:
        return len({m.metadata.get("条款号")
                    for m in engine.memories.values()
                    if m.metadata.get("条款号")})
    except Exception:
        return -1


def main() -> None:
    tasks_kb = load(root, KB_TASKS)["tasks"]
    chunks_law = load(root, LAW_CHUNKS)["chunks"]
    chunks_med = load(root, MEDICAL_CHUNKS)["chunks"]
    llm = LLMClient()
    if not llm.configured:
        print("✗ 未设置 SME_LLM_API_KEY")
        sys.exit(1)
    embedder = get_embedder()
    ev = QAEvaluator(llm, tasks_kb)
    ws = root / "results" / "battle2_ws_kb_fixed"

    report = {"law_clauses_total": 1260, "law_chunks": len(chunks_law),
              "medical_chunks": len(chunks_med), "tasks": len(tasks_kb),
              "results": {}}
    for name in ALL_ADAPTERS:
        print(f"[kb-fixed] {name} 导入 {len(chunks_law) + len(chunks_med)} 条...")
        t0 = time.perf_counter()
        try:
            a = ALL_ADAPTERS[name](workspace=str(ws), embedder=embedder)
            import_kb(a, chunks_law, "law")
            import_kb(a, chunks_med, "med")
            import_s = time.perf_counter() - t0
            r = ev.eval_adapter(a, tag=name + "_kb")
            entry = {
                "accuracy": r["accuracy"], "n": r["n"],
                "led_astray": r["led_astray"],
                "details": r["details"],
                "import_seconds": round(import_s, 1),
                "memories": len(a.engine.memories) if name.startswith("sme_")
                else a.store_count,
            }
            if name.startswith("sme_"):
                entry["clause_ids"] = clause_count(a.engine)
            report["results"][name] = entry
            print(f"  acc={r['accuracy']:.3f} mem={entry['memories']} "
                  f"clauses={entry.get('clause_ids', '-')} "
                  f"导入{import_s:.0f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name}: {exc}")
            report["results"][name] = {"error": str(exc)}

    out = root / "results" / "battle2_kb_fixed.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"✓ → {out}")


if __name__ == "__main__":
    main()
