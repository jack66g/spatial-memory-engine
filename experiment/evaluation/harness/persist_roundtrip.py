# -*- coding: utf-8 -*-
"""
T4a 持久化往返完整性:save → load → 逐字段核对 + 检索行为一致性。
覆盖:json 后端 × compress 开/关、sqlite 后端(若实现)、归档记忆、图边、metadata/tags。
任何不一致 = bug 证据。
"""
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ← {detail}" if detail and not cond else ""))


def build_state(engine, n=300):
    """构造有代表性状态:普通/高importance/带metadata/tags/图边/归档/强化过"""
    import random
    rnd = random.Random(7)
    ids = []
    for i in range(n):
        m = engine.add(f"测试记忆{i}:用户{rnd.choice(['喜欢','讨厌'])}{rnd.choice(['苹果','跑步','Python','火锅'])}且编号{i}",
                       metadata={"idx": i, "tag_str": f"t{i%7}", "tag_list": [i % 3, "x"]},
                       tags=[f"tag{i%5}"], importance=[0.2, 0.5, 0.9][i % 3])
        ids.append(m.id)
    # 图边
    for a, b in zip(ids[:50], ids[1:51]):
        engine.link(a, b, kind="reference", weight=1.0)
    # 强化 + 归档一部分
    for mid in ids[:20]:
        engine.reinforce(mid)
    for mid in ids[60:70]:
        engine.archive(mid)
    return ids


def snapshot(engine):
    snap = {}
    for mid, m in engine.memories.items():
        snap[mid] = {
            "text": m.text, "metadata": m.metadata, "tags": list(m.tags or []),
            "importance": m.importance, "weight": round(m.weight, 6),
            "hit_count": m.hit_count, "archived": m.archived,
            "created_at": m.created_at, "version": getattr(m, "version", None),
        }
    return snap


def compare(sa, sb, label):
    keys_a, keys_b = set(sa), set(sb)
    check(f"{label}: 记忆数一致({len(sa)})", len(sa) == len(sb), f"save={len(sa)} load={len(sb)}")
    missing = keys_a - keys_b
    check(f"{label}: 无记忆丢失", not missing, f"丢失{len(missing)}条: {list(missing)[:5]}")
    extra = keys_b - keys_a
    check(f"{label}: 无幽灵记忆", not extra, f"多出{len(extra)}条")
    diffs = []
    for k in keys_a & keys_b:
        for field in sa[k]:
            va, vb = sa[k][field], sb[k].get(field)
            if field in ("weight",):
                if abs(va - (vb or 0)) > 1e-6:
                    diffs.append((k, field, va, vb))
            elif va != vb:
                diffs.append((k, field, va, vb))
    check(f"{label}: 字段逐一相等", not diffs, f"{len(diffs)}处差异, 前3: {diffs[:3]}")
    return not diffs


def run_case(case_name, overrides, use_sqlite=False):
    print(f"\n=== {case_name} ===")
    from sme.config import SMEConfig
    from sme import SpatialMemoryEngine
    issues = []
    with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
        state = str(Path(td) / "state.json")
        cfg = SMEConfig.from_dict(overrides) if overrides else SMEConfig()
        cfg.storage.autosave = False
        cfg.storage.path = state
        if use_sqlite:
            cfg.storage.backend = "sqlite"
        e1 = SpatialMemoryEngine(cfg)
        ids = build_state(e1)
        s1 = snapshot(e1)
        edges1 = sorted((ed.source, ed.target, ed.kind) for ed in e1.graph_edges()) if e1.graph_edges() else []
        n_arch1 = e1.archived_count()
        q = "用户喜欢什么"
        r1 = [(h.memory.id, round(h.score, 4)) for h in e1.search(q, top_k=10)]
        p = e1.save(state)

        e2 = SpatialMemoryEngine()
        ok = e2.load(p)
        check(f"{case_name}: load() 返回 True", ok is True or ok, f"返回 {ok!r}")
        s2 = snapshot(e2)
        compare(s1, s2, case_name)
        edges2 = sorted((ed.source, ed.target, ed.kind) for ed in e2.graph_edges()) if e2.graph_edges() else []
        check(f"{case_name}: 图边保留({len(edges1)})", edges1 == edges2,
              f"save={len(edges1)} load={len(edges2)}")
        check(f"{case_name}: 归档数保留", n_arch1 == e2.archived_count(), f"{n_arch1} vs {e2.archived_count()}")
        r2 = [(h.memory.id, round(h.score, 4)) for h in e2.search(q, top_k=10)]
        ids1 = [x[0] for x in r1]; ids2 = [x[0] for x in r2]
        check(f"{case_name}: 检索结果序一致", ids1 == ids2, f"{ids1[:5]} vs {ids2[:5]}")
        if r1 and r2:
            maxdiff = max(abs(a[1] - b[1]) for a, b in zip(r1, r2)) if len(r1) == len(r2) else 999
            check(f"{case_name}: 分数一致(±1e-3)", maxdiff <= 1e-3, f"最大分差 {maxdiff}")
        # 二次往返(链式稳定性)
        p2 = e2.save(str(Path(td) / "state2.json"))
        e3 = SpatialMemoryEngine()
        e3.load(p2)
        compare(s2, snapshot(e3), case_name + "-二跳")
    return issues


if __name__ == "__main__":
    t0 = time.time()
    run_case("json+compress", {})
    run_case("json+nocompress", {"storage": {"compress": False}})
    try:
        run_case("sqlite", {}, use_sqlite=True)
    except Exception as ex:
        print(f"\n[CASE-CRASH] sqlite: {ex}")
        FAIL.append(f"sqlite 后端崩溃: {ex}")
    print(f"\n===== 持久化往返: PASS={len(PASS)} FAIL={len(FAIL)} ({time.time()-t0:.0f}s) =====")
    if FAIL:
        print("失败项:")
        for f in FAIL:
            print("  -", f)
    (HERE.parent / "results" / "t4a_persist.json").write_text(
        json.dumps({"pass": PASS, "fail": FAIL}, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(1 if FAIL else 0)
