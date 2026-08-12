"""跑批：所有基线回放同一对话流 → 同一考问集 → 统一指标。

用法::

    python scripts/run_battle.py --dialogue data/dialogue_300.json \
        --quiz data/quiz_300.json
    python scripts/run_battle.py --baselines sme_chat,rag,bm25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import ALL_ADAPTERS  # noqa: E402
from baselines.embedder import get_embedder  # noqa: E402


def _contains(a: str, b: str) -> bool:
    """双向包含匹配（存储文本与期望文本可能互为子串）。"""
    if not b:
        return False
    if len(b) < 6:
        return a == b
    return b in a or a in b


def first_hit(retrieved: list, expect: str) -> int:
    for i, r in enumerate(retrieved):
        if _contains(r.text, expect):
            return i
    return -1


def semantic_first_hit(retrieved: list, expect: str, embedder,
                       threshold: float = 0.60) -> int:
    """语义命中：期望文本与命中文本的 BGE 余弦 ≥ 阈值（对 LLM 提取
    归一化鲁棒，如"我叫小林"→"用户叫小林"）。"""
    import numpy as np

    q = np.asarray(embedder.embed_one(expect), dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q), 1e-12, None)
    for i, r in enumerate(retrieved):
        v = np.asarray(embedder.embed_one(r.text), dtype=np.float64)
        v = v / np.clip(np.linalg.norm(v), 1e-12, None)
        if float(v @ q) >= threshold:
            return i
    return -1


def run_baseline(name: str, factory, dialogue: dict, quiz: dict,
                 workspace: str, embedder) -> dict:
    adapter = factory(workspace=workspace, embedder=embedder)
    print(f"  [{adapter.name}] 回放 {len(dialogue['turns'])} 轮...")
    t0 = time.perf_counter()
    for turn in dialogue["turns"]:
        adapter.store(turn["text"], role=turn["role"])
    replay_s = time.perf_counter() - t0

    hits = {1: 0, 3: 0, 5: 0}
    sem_hits = {1: 0, 3: 0, 5: 0}
    misses: list[dict] = []
    queries = quiz["queries"]
    for item in queries:
        q, expect = item["q"], item["expect"]
        top_k = int(item.get("top_k", 5))
        try:
            retrieved = adapter.search(q, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            misses.append({"q": q, "error": str(exc)})
            continue
        pos = first_hit(retrieved, expect)
        spos = semantic_first_hit(retrieved, expect, embedder)
        for k in (1, 3, 5):
            if 0 <= pos < k:
                hits[k] += 1
            if 0 <= spos < k:
                sem_hits[k] += 1
        if pos < 0 and spos < 0:
            misses.append({"q": q, "expect": expect[:30]})

    n = len(queries)
    stats = adapter.stats()
    return {
        "name": name,
        "hit@1": round(hits[1] / n, 4),
        "hit@3": round(hits[3] / n, 4),
        "hit@5": round(hits[5] / n, 4),
        "sem@1": round(sem_hits[1] / n, 4),
        "sem@3": round(sem_hits[3] / n, 4),
        "sem@5": round(sem_hits[5] / n, 4),
        "replay_seconds": round(replay_s, 2),
        "misses": misses[:10],
        **stats,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="记忆系统擂台赛跑批")
    parser.add_argument("--dialogue", default="data/dialogue_300.json")
    parser.add_argument("--quiz", default="data/quiz_300.json")
    parser.add_argument("--baselines", default="",
                        help="逗号分隔；空 = 全部（缺依赖的自动跳过）")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--out", default="results/battle.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dialogue = json.loads((root / args.dialogue).read_text(encoding="utf-8"))
    quiz = json.loads((root / args.quiz).read_text(encoding="utf-8"))
    workspace = root / "results" / f"run_{int(time.time())}"
    workspace.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"对话 {len(dialogue['turns']) // 2} 轮 / 考问 {len(quiz['queries'])} 题")
    print(f"embedding: {args.embedding_model}（所有向量基线共享）")
    print(f"workspace: {workspace}")
    print("=" * 64)

    names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    if not names:
        names = list(ALL_ADAPTERS)
    embedder = get_embedder(args.embedding_model)

    results: list[dict] = []
    for name in names:
        factory = ALL_ADAPTERS.get(name)
        if factory is None:
            print(f"  ✗ 未知基线：{name}（可选：{' '.join(ALL_ADAPTERS)}）")
            continue
        try:
            results.append(run_baseline(name, factory, dialogue, quiz,
                                        str(workspace), embedder))
        except ImportError as exc:
            print(f"  - 跳过 {name}：{exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name} 运行失败：{exc}")

    # 汇总表格
    print()
    print(f"{'基线':<18}{'hit@1':>7}{'sem@1':>7}{'sem@3':>7}{'sem@5':>7}"
          f"{'存储ms':>8}{'检索ms':>8}{'条数':>6}")
    for r in results:
        print(f"{r['name']:<18}{r['hit@1']:>7.3f}{r['sem@1']:>7.3f}"
              f"{r['sem@3']:>7.3f}{r['sem@5']:>7.3f}"
              f"{r['store_avg_ms']:>8.1f}{r['search_avg_ms']:>8.1f}"
              f"{r.get('memories', 0):>6}")

    report = {
        "generated_at": time.time(),
        "dialogue": args.dialogue,
        "quiz": args.quiz,
        "embedding_model": args.embedding_model,
        "llm_gen_cost": dialogue.get("llm", {}),
        "results": results,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n✓ 报告 → {out}")


if __name__ == "__main__":
    main()
