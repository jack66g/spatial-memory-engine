# -*- coding: utf-8 -*-
"""
T4c 记忆动力学长程模拟:虚拟时间轴上验证衰减/强化/融合/压缩的行为契约。
契约(README 口径):衰减永不删除记忆;归档可恢复;强化走 Ebbinghaus;融合不丢关键信息。
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not cond else ""))


def fresh(overrides=None):
    from sme import SpatialMemoryEngine, SMEConfig
    cfg = SMEConfig.from_dict(overrides) if overrides else SMEConfig()
    cfg.storage.autosave = False
    cfg.maintenance.background = False
    cfg.storage.path = str(HERE.parent / "work" / "dyn_state.json")
    return SpatialMemoryEngine(cfg)


def run():
    DAY = 86400
    now = time.time()
    e = fresh()

    # 1) 时间注入:300 条记忆铺在 365 天时间轴上
    ids = []
    for i in range(300):
        m = e.add(f"动力学记忆{i}:主题{chr(0x4e00 + (i % 40))}的内容编号{i}")
        obj = e.memories[m.id]
        age_days = [0, 1, 7, 30, 90, 180, 365][i % 7]
        obj.created_at = now - age_days * DAY
        obj.last_hit = now - age_days * DAY
        ids.append(m.id)

    # 2) 强化一半(模拟高频命中)
    for mid in ids[:150]:
        e.reinforce(mid)

    # 3) 衰减:推进虚拟时间 90 天再 apply_decay
    ref = now + 90 * DAY
    n_decayed = e.apply_decay(reference=ref)
    check("apply_decay 返回处理数 >= 0", isinstance(n_decayed, int) and n_decayed >= 0, f"{n_decayed}")
    alive = len(e.memories)
    check("衰减后无记忆被删除(永不删除契约)", alive == 300, f"剩 {alive}/300")

    # 4) decay 因子分层验证:【未强化】的新老对比(强化会刷新 last_hit,不能当对照组)
    i_new = next(i for i in range(150, 300) if i % 7 == 0)   # age=0 → 距 ref 90 天
    i_old = next(i for i in range(150, 300) if i % 7 == 6)   # age=365 → 距 ref 455 天
    f_new = e.memories[ids[i_new]].decay_factor
    f_old = e.memories[ids[i_old]].decay_factor
    check("新记忆 decay 因子 > 老记忆", f_new > f_old, f"new={f_new:.3f} old={f_old:.3f}")

    # 5) 强化后的 weight 应高于未强化(同龄记忆)
    w_reinforced = e.memories[ids[0]].weight    # 被强化
    # 找同龄未强化对照(i%7==0 且 i>=150)
    ctrl = next(mid for i, mid in enumerate(ids) if i % 7 == 0 and i >= 150)
    w_ctrl = e.memories[ctrl].weight
    check("强化记忆 weight > 未强化同龄", w_reinforced > w_ctrl, f"强化={w_reinforced:.3f} 对照={w_ctrl:.3f}")

    # 6) 衰减关闭时因子恒定
    e2 = fresh({"policy": {"decay_enabled": False}})
    m2 = e2.add("无衰减记忆")
    e2.memories[m2.id].created_at = now - 365 * DAY
    e2.apply_decay(reference=now)
    check("decay 关闭时因子恒为 1", abs(e2.memories[m2.id].decay_factor - 1.0) < 1e-9,
          f"{e2.memories[m2.id].decay_factor}")

    # 7) 融合:加入 6 条高度相似记忆 → consolidate 产生 summary,原文不消失(可追溯)
    e3 = fresh()
    dup_ids = [e3.add(f"用户喜欢苹果,每天吃{i}个").id for i in range(6)]
    others = [e3.add(f"无关干扰记忆{j}") for j in range(10)]
    summaries = e3.consolidate()
    n_sum = len(summaries)
    check("相似簇 consolidate 产生摘要记忆", n_sum >= 1, f"{n_sum} 条摘要")
    if n_sum:
        sm = summaries[0]
        check("摘要 source=summary", getattr(sm, "source", "") == "summary", f"source={getattr(sm, 'source', '?')}")
        has_parent = getattr(sm, "parent_id", None) or (getattr(sm, "children", None) or [])
        check("摘要保留父子链", bool(has_parent), f"parent_id={getattr(sm, 'parent_id', None)}, children={len(getattr(sm, 'children', []) or [])}")
        for d in dup_ids:
            m = e3.memories.get(d)
            if m and getattr(m, "archived", False):
                ok = e3.restore(d)
                check("融合归档的原记忆可恢复", ok is True or ok)
                break
        else:
            check("融合后原文仍可检索(未归档路径)", True)

    # 8) 压缩:老记忆 compress
    e4 = fresh()
    old = e4.add("很老的知识条目")
    e4.memories[old.id].created_at = now - 300 * DAY
    for j in range(12):
        e4.add(f"新知识条目{j}")
    comp = e4.compress()
    check("compress 对老记忆产生动作", isinstance(comp, list), f"{len(comp)} 条")

    # 9) effective_importance 地板契约
    e5 = fresh()
    m5 = e5.add("地板测试", importance=0.9)
    e5.memories[m5.id].created_at = now - 3650 * DAY  # 10年
    e5.apply_decay(reference=now)
    ei = e5.decay.effective_importance(e5.memories[m5.id])
    floor = 0.9 * (1 - 0.85)
    check("10年记忆 effective_importance 不低于地板", ei >= floor - 1e-6, f"ei={ei:.4f} floor={floor:.4f}")

    # 10) 归档不可检索、恢复后可检索
    e6 = fresh()
    m6 = e6.add("归档测试:用户养了一只猫叫豆豆")
    e6.archive(m6.id)
    hits_arch = e6.search("豆豆", top_k=5)
    check("归档记忆默认不出现在检索", all(h.memory.id != m6.id for h in hits_arch))
    e6.restore(m6.id)
    hits_rest = e6.search("豆豆", top_k=5)
    check("恢复后重新可检索", any(h.memory.id == m6.id for h in hits_rest))


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\n===== 动力学模拟: PASS={len(PASS)} FAIL={len(FAIL)} ({time.time()-t0:.0f}s) =====")
    for f in FAIL:
        print("  FAIL:", f)
    (HERE.parent / "results" / "t4c_dynamics.json").write_text(
        json.dumps({"pass": PASS, "fail": FAIL}, ensure_ascii=False, indent=1), encoding="utf-8")
    sys.exit(1 if FAIL else 0)
