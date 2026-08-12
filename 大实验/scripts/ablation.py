"""SME 本体消融实验：16 配置 × 多场景，双方向交叉验证。

方向 A（基线逐步叠加）：v1 默认 → 逐个开模块
方向 B（全开逐一剔除）：知识库全开 → 逐个关模块

提取统一用 rules 模式（免费、确定性强、内部自洽），检索统一本地 BGE。
场景按任务 id 前缀分组统计：recall(01 主场)/longtail(衰减压缩)/corr(05 主场)
/qa(02 主场)/rel(03 主场)/noise(06 主场)/profile(04 主场)。

用法::

    python scripts/ablation.py --dialogue data/dialogue_300.json \
        --tasks data/tasks_dialogue.json --attack data/tasks_attack.json \
        --out results/ablation_matrix.json
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

from eval_qa import QAEvaluator  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from make_tasks import extract_keywords  # noqa: E402

SCENE_PREFIX = {
    "recall": "直接回忆(01)",
    "longtail": "长尾细节",
    "corr": "纠错版本(05)",
    "qa": "问答回放(02)",
    "profile": "画像聚合(04)",
    "rel": "图谱关联(03)",
    "noise": "噪音抑制(06)",
    "dialect": "同义改写",
}


def make_config(name: str, *, modules: dict, kb_style: bool = False,
                bigram: bool = True) -> dict:
    """构造一个 SME 引擎配置参数集（rules 提取，本地 BGE）。"""
    cfg = {
        "name": name,
        "embedding": {"provider": "sentence-transformers",
                      "model": "BAAI/bge-small-zh-v1.5", "dim": 512},
        "storage": {"autosave": False,
                    "path": f"results/ablation_ws/{name}.json"},
        "extraction": {"enabled": modules.get("01", False), "mode": "rules"},
        "factversion": {"enabled": modules.get("05", False)},
        "qapair": {"enabled": modules.get("02", False)},
        "factgraph": {"enabled": modules.get("03", False),
                      "extract_mode": "rules"},
        "profile": {"enabled": modules.get("04", False)},
        "noise": {"enabled": modules.get("06", False)},
        "persistence": {"enabled": modules.get("07", False)},
        "namespaces": {"enabled": modules.get("12", False)},
        "rerank": {"enabled": False},
        "retrieval": {"cjk_bigram": bigram},
    }
    if kb_style:
        # 知识库风格：无衰减/无演化/无强化（对比方向 B 的基础）
        cfg["policy"] = {"decay_enabled": False, "full_memory": True}
        cfg["region"] = {"auto_evolve": False}
    return cfg


def build_configs() -> list[dict]:
    off = {}
    A = []
    A.append(make_config("A0_全关基线", modules=off))
    for mod, label in [("01", "+01提取"), ("05", "+05纠错"), ("02", "+02问答对"),
                       ("03", "+03图谱"), ("04", "+04画像"), ("06", "+06噪音")]:
        m = dict(off)
        m[mod] = True
        A.append(make_config(f"A_{label}", modules=m))
    A.append(make_config("A_bigram关", modules=off, bigram=False))

    full = {"01": True, "05": True, "02": True, "03": True,
            "04": True, "06": True, "07": True}
    B = [make_config("B0_知识库全开", modules=full, kb_style=True)]
    for mod, label in [("01", "−01提取"), ("05", "−05纠错"), ("02", "−02问答对"),
                       ("03", "−03图谱"), ("04", "−04画像"), ("06", "−06噪音"),
                       ("07", "−07WAL")]:
        m = dict(full)
        m[mod] = False
        B.append(make_config(f"B_{label}", modules=m, kb_style=True))
    return A + B


def build_engine(cfg: dict, workspace: str, model: str):
    from sme.config import SMEConfig
    from sme.engine import SpatialMemoryEngine

    params = dict(cfg)
    params["embedding"]["model"] = model
    params["storage"]["path"] = str(Path(workspace) / f"{cfg['name']}.json")
    return SpatialMemoryEngine(SMEConfig.from_dict(params))


class RulesAdapter:
    """轻量适配器：只做 store/search（消融回放用，绕过 baselines 包）。"""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.search_ms: list[float] = []

    def store(self, text: str, role: str = "user") -> None:
        self.engine.add(text, source=role)

    def search(self, query: str, top_k: int = 5):
        from sme.retrieval import SearchQuery

        t0 = time.perf_counter()
        hits = self.engine.search(SearchQuery(text=query, top_k=top_k))
        self.search_ms.append((time.perf_counter() - t0) * 1000.0)
        return [type("H", (), {"text": h.memory.text, "score": float(h.score)})()
                for h in hits]


def group_stats(results: list[dict]) -> dict:
    """按场景前缀分组统计正确率（attack_xxx → xxx）。"""
    groups: dict[str, dict] = {}
    for r in results:
        if r.get("error"):
            continue
        raw = (r.get("id") or "?").split("_")
        prefix = raw[1] if raw and raw[0] == "attack" and len(raw) > 1 else raw[0]
        if prefix not in SCENE_PREFIX:
            prefix = "other"
        g = groups.setdefault(prefix, {"n": 0, "ok": 0, "astray": 0})
        g["n"] += 1
        g["ok"] += 1 if r.get("correct") else 0
        g["astray"] += 1 if r.get("led_astray") else 0
    out = {}
    for k, g in groups.items():
        out[SCENE_PREFIX.get(k, k)] = {
            "n": g["n"], "acc": round(g["ok"] / g["n"], 3) if g["n"] else 0,
            "astray": g["astray"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SME 消融实验")
    parser.add_argument("--dialogue", default="data/dialogue_300.json")
    parser.add_argument("--tasks", default="data/tasks_dialogue.json")
    parser.add_argument("--attack", default="data/tasks_attack.json")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--out", default="results/ablation_matrix.json")
    parser.add_argument("--skip-replay", action="store_true",
                        help="跳过回放（复用已有工作区）")
    parser.add_argument("--repeat", type=int, default=2,
                        help="每配置评测次数（取均值，消除 LLM 随机噪声）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dialogue = json.loads((root / args.dialogue).read_text(encoding="utf-8"))
    tasks = json.loads((root / args.tasks).read_text(encoding="utf-8"))["tasks"]
    attack = json.loads((root / args.attack).read_text(encoding="utf-8"))["tasks"]
    all_tasks = tasks + attack
    workspace = root / "results" / "ablation_ws"
    workspace.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    if not llm.configured:
        print("✗ 未设置 SME_LLM_API_KEY")
        sys.exit(1)
    ev = QAEvaluator(llm, all_tasks)

    configs = build_configs()
    print(f"消融配置 {len(configs)} 个 × {len(all_tasks)} 题 × {args.repeat} 次"
          f"（rules 提取，本地 BGE）")
    matrix = {}
    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== {name} ===")
        engine = build_engine(cfg, str(workspace), args.embedding_model)
        adapter = RulesAdapter(engine)
        if not args.skip_replay:
            for turn in dialogue["turns"]:
                adapter.store(turn["text"], role=turn["role"])
            engine.save(engine.config.storage.path)  # 落盘，供 --skip-replay 复用
        else:
            if not engine.load(engine.config.storage.path):
                print(f"  ! 无快照 {engine.config.storage.path}，需先回放")
                continue
        # 复跑取均值（LLM 回答有随机性）
        accs, astrays, groups_list = [], [], []
        for _ in range(args.repeat):
            result = ev.eval_adapter(adapter, tag=name)
            accs.append(result["accuracy"])
            astrays.append(result["led_astray"])
            groups_list.append(group_stats(result["details"]))
        acc = sum(accs) / len(accs)
        groups: dict = {}
        for g in groups_list:
            for k, v in g.items():
                entry = groups.setdefault(k, {"n": v["n"], "accs": []})
                entry["accs"].append(v["acc"])
        group_out = {}
        for k, v in groups.items():
            group_out[k] = {
                "n": v["n"],
                "acc": round(sum(v["accs"]) / len(v["accs"]), 3),
                "spread": round(max(v["accs"]) - min(v["accs"]), 3),
                "astray": max(astrays),
            }
        matrix[name] = {
            "accuracy": round(acc, 3),
            "spread": round(max(accs) - min(accs), 3),
            "n": len(all_tasks),
            "led_astray": max(astrays),
            "groups": group_out,
            "llm_calls": result["llm_calls"] * args.repeat,
            "memories": len(engine.memories),
        }
        line = "  ".join(f"{k}={v['acc']:.2f}" for k, v in group_out.items())
        print(f"  acc={acc:.3f}±{matrix[name]['spread']:.3f} "
              f"astray={max(astrays)} mem={len(engine.memories)} | {line}")

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n✓ 消融矩阵 → {out}")


if __name__ == "__main__":
    main()
