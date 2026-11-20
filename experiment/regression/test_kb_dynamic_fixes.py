"""kb_dynamic 擂台"被带偏"修复的回归测试（iteration 2.6 归因）。

覆盖四个修复点（全部默认关=零漂移，见各 test）：
  1. bridge._pool_widen：模块全关不动 query（零漂移护栏）
  2. 检索候选池扩宽 + search_post 恢复 top_k（v1 在 search_post 之前
     就截断 top_k，stale 降权/噪音重排只能在预切列表里洗牌）
  3. search_post 冗余折叠：同义变体降位不删、带新数字的候选保护
  4. factversion 多目标 stale 传播（同批兄弟 + 跨批高相似变体）
  5. extraction keep_raw 双写（getattr 开关，默认关）
"""

from __future__ import annotations

import numpy as np
import pytest

from sme.models import Memory, SearchHit
from sme.modules.bridge import V2Bridge
from sme.modules.factversion import ADD_BATCH_KEY, STALE_TAG


def _hit(vec, text, score):
    mem = Memory(text=text, embedding=np.array(vec, dtype=float))
    return SearchHit(memory=mem, score=score)


# ----------------------- 1. pool widen 零漂移 ------------------------------- #
def test_pool_widen_off_when_modules_off(fresh_engine):
    from sme.retrieval.retriever import SearchQuery

    q = SearchQuery(text="咖啡", top_k=5)
    fresh_engine._v2.search_pre(q)
    assert q.top_k == 5  # all post modules off -> query untouched


def test_pool_widen_and_restore(fresh_engine):
    from sme.retrieval.retriever import SearchQuery

    e = fresh_engine
    e.config.noise.enabled = True
    e.add("用户喜欢喝咖啡")
    e.add("用户住在成都")

    q = SearchQuery(text="咖啡", top_k=5)
    e._v2.search_pre(q)
    assert q.top_k >= 20  # widened for the post-rerank pool

    hits = e.retriever.search(e, q)
    out = e._v2.search_post(q, hits)
    assert len(out) <= 5
    assert q.top_k == 5          # caller-owned query restored
    assert not getattr(q, "_v2_pooled", False)


# ----------------------- 2. 冗余折叠 ---------------------------------------- #
def test_collapse_demotes_variants_keeps_diverse():
    e1 = [1.0, 0.0]
    e2 = [0.99, 0.02]   # ~0.9996 to e1: paraphrase variant
    e3 = [0.0, 1.0]     # distinct fact
    hits = [
        _hit(e1, "用户养的狗叫煤球", 0.9),
        _hit(e2, "用户养了一只名叫煤球的狗", 0.85),
        _hit(e3, "用户囤了云南咖啡豆", 0.8),
    ]
    out = V2Bridge._collapse_redundant(hits)
    texts = [h.memory.text for h in out]
    assert texts[0] == "用户养的狗叫煤球"        # best variant first
    assert texts[1] == "用户囤了云南咖啡豆"      # diverse fact pulled up
    assert texts[2] == "用户养了一只名叫煤球的狗"  # variant demoted, not dropped
    assert len(out) == 3


def test_collapse_digit_guard_keeps_value_variants():
    e1 = [1.0, 0.0]
    e2 = [0.999, 0.01]  # ~1.0: same schema, different value
    hits = [
        _hit(e1, "用户月租1900", 0.9),   # newer value kept first
        _hit(e2, "用户月租2300", 0.85),  # old value: near-dup BUT new digits
    ]
    out = V2Bridge._collapse_redundant(hits)
    assert [h.memory.text for h in out][:2] == ["用户月租1900", "用户月租2300"]


def test_collapse_far_pairs_untouched():
    hits = [
        _hit([1.0, 0.0], "用户的项目叫星链", 0.9),
        _hit([0.0, 1.0], "用户合租室友的猫叫年糕", 0.8),
    ]
    out = V2Bridge._collapse_redundant(hits)
    assert [h.memory.text for h in out] == [
        "用户的项目叫星链", "用户合租室友的猫叫年糕",
    ]


def test_collapse_superset_swaps_in_richer_text():
    """同义簇里信息更全的成员（原文吞并其碎片）占位，碎片降位。"""
    e1 = [1.0, 0.0]
    e2 = [0.995, 0.03]  # ~0.9995: same fact
    hits = [
        _hit(e1, "用户每周二四晚上8点去府河绿道慢跑", 0.9),      # fragment
        _hit(e2, "用户每周二四晚上8点去小区楼下沿府河绿道慢跑一般3.5公里", 0.85),  # raw superset
    ]
    out = V2Bridge._collapse_redundant(hits)
    assert out[0].memory.text.startswith("用户每周二四晚上8点去小区楼下")
    assert out[1].memory.text == "用户每周二四晚上8点去府河绿道慢跑"


def test_search_returns_full_topk_despite_collapse(fresh_engine):
    """折叠降位不删：top_k=5 仍返回 5 条（变体垫底）。"""
    e = fresh_engine
    e.config.noise.enabled = True
    base = "用户养了一只柯基叫煤球"
    variants = [
        base,
        "用户养的柯基名叫煤球",
        "用户家的狗是柯基煤球",
        "用户养了柯基，名字叫煤球",
        "用户有一条柯基犬叫煤球",
        "用户的柯基狗狗叫煤球",
    ]
    for v in variants:
        e.add(v)
    e.add("用户喜欢喝手冲咖啡")
    out = e.search("用户的狗是什么品种叫什么", top_k=5)
    assert len(out) == 5


# ----------------------- 3. factversion stale 传播 -------------------------- #
def test_correction_marks_batch_siblings_stale(fresh_engine):
    """改口压制整条旧说法：同批（同一次 add）的兄弟记忆一起降权。"""
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.factversion.enabled = True
    e.add("用户喜欢喝咖啡")                  # old statement (fact)
    old = next(iter(e.memories.values()))

    # 手工放一条"同批"兄弟（同一 add 批次的其它碎片/原文）
    sib = e.memory_manager.add_memory(
        text="用户天天都要喝咖啡",
        metadata={"fact_kind": "fact", ADD_BATCH_KEY: old.metadata[ADD_BATCH_KEY]},
        tags=["fact", "extracted"], source="user",
    )
    # 跨批高相似变体（verbatim restatement）
    variant = e.memory_manager.add_memory(
        text="用户喜欢喝咖啡了", tags=["fact"], source="user",
    )
    # 无关记忆
    far = e.memory_manager.add_memory(
        text="用户在公司上班", tags=["fact"], source="user",
    )

    e.add("其实用户不喜欢喝咖啡了")           # correction (matches `old`)

    stale = [m for m in e.memories.values() if m.metadata.get(STALE_TAG)]
    stale_ids = {m.id for m in stale}
    assert old.id in stale_ids
    assert sib.id in stale_ids        # batch sibling propagated
    assert variant.id in stale_ids    # cross-batch verbatim variant propagated
    assert far.id not in stale_ids    # unrelated untouched


# ----------------------- 4. keep_raw 双写 ----------------------------------- #
def test_keep_raw_off_by_default(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["likes_coffee"])
    texts = [m.text for m in e.memories.values()]
    assert texts == [_strip(zh["likes_coffee"])]  # only the fact, no raw copy


def test_keep_raw_dual_write(fresh_engine):
    """LLM 抽取把长句改写成碎片时，原文双写保住场景上下文。

    rules 模式下抽取结果≈原句（cos=1.0，raw 会被去重跳过——短单事实轮
    本就没有上下文可丢，属预期行为），所以这里桩掉 extract 模拟 LLM 的
    改写行为来测双写链路。
    """
    from sme.models import Fact

    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.extraction.keep_raw = True  # getattr flag (lab kb_dynamic)
    raw_text = "为了对付成都春天的雾霾，我在京东买了个米家空气净化器4 Pro，花了899"
    e.extraction.extract = lambda text, assistant=False: [
        Fact(text="用户买了一个米家空气净化器", kind="fact", subject="用户"),
    ]
    e.add(raw_text)
    texts = sorted(m.text for m in e.memories.values())
    assert texts == sorted(["用户买了一个米家空气净化器", raw_text])
    raw = next(m for m in e.memories.values() if m.text == raw_text)
    assert raw.metadata.get("kind") == "raw"
    assert "raw" in raw.tags
    assert raw.metadata.get(ADD_BATCH_KEY) is not None


def test_keep_raw_still_drops_noise(fresh_engine, zh):
    """keep_raw 只在抽取出了事实时双写：纯寒暄轮依旧全丢。"""
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.extraction.keep_raw = True
    e.add(zh["haha"])
    assert len(e.memories) == 0


def test_keep_raw_dedups_repeated_raw(fresh_engine, zh):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.extraction.keep_raw = True
    e.add(zh["likes_coffee"])
    e.add(zh["likes_coffee"])  # verbatim repeat: raw copy must not duplicate
    raws = [m for m in e.memories.values() if m.text == zh["likes_coffee"]]
    assert len(raws) == 1


def _strip(text: str) -> str:
    from sme.modules.extraction import _strip_particles

    return _strip_particles(text)
