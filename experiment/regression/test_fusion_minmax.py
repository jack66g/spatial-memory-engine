"""retrieval.fusion 混合归一化（iteration 3.x）：weighted 兼容档 + minmax 新档。

三个已知缺陷的单元级验证：
  1. BM25 峰值归一化漂移 —— weighted 模式下候选集一变（去掉最强匹配），
     池内其余文档分数全体膨胀；弱匹配单独成池拿虚假 1.0。
     minmax 模式：无峰值锚定（单独成池 0.5），匹配区间下段漂移显著减小、
     最弱匹配零漂移。
  2. vector 通道 (cos+1)/2 压缩 —— 正交文档白拿 0.5，量程被压扁。
     minmax 模式：正交文档≈池内最低=0，量程拉开到全 [0,1]。
  3. metadata 死信号 —— 候选已被预过滤，meta_score 恒 1.0，加权后只是
     常数偏移。minmax 模式删除该通道，权重按比例并入 vector/keyword。

weighted（默认）代码路径必须与旧实现逐字节等价：见
test_weighted_mode_matches_legacy_formula。
"""

from __future__ import annotations

import math
import statistics

import numpy as np
import pytest

from sme.config import RetrievalConfig, RankingConfig, SMEConfig
from sme.config_items import ITEM_BY_PATH, defaults_config, parse_value
from sme.retrieval.ranking import MemoryRanker
from sme.retrieval.retriever import TwoStageRetriever


def _retriever(fusion: str) -> TwoStageRetriever:
    cfg = RetrievalConfig()
    cfg.fusion = fusion
    return TwoStageRetriever(cfg, MemoryRanker(RankingConfig()))


class _FakeMem:
    """Minimal memory stub for channel-level scoring (id + optional text)."""

    _n = 0

    def __init__(self, mid: str | None = None, text: str = "",
                 embedding=None) -> None:
        self.id = mid or f"m{_FakeMem._n}"
        self.text = text
        self.embedding = embedding
        _FakeMem._n += 1


# --------------------------------------------------------------------- #
# 配置注册
# --------------------------------------------------------------------- #
def test_fusion_config_item_registered():
    item = ITEM_BY_PATH["retrieval.fusion"]
    assert item.kind == "enum"
    assert item.choices == ("weighted", "minmax")
    # A/B 达标（sme_chat +2.5pp / sme_minimal +4.2pp）后默认档 = minmax
    assert item.default == "minmax"
    assert item.group == "检索与排序"
    # defaults_config 自带该项（Web 配置中心自动展示）
    assert defaults_config()["retrieval"]["fusion"] == "minmax"
    # 解析与校验
    assert parse_value(item, "weighted") == "weighted"
    with pytest.raises(ValueError):
        parse_value(item, "cosine")
    # SMEConfig 字段 + from_dict 往返
    assert RetrievalConfig().fusion == "minmax"
    cfg = SMEConfig.from_dict({"retrieval": {"fusion": "weighted"}})
    assert cfg.retrieval.fusion == "weighted"
    assert cfg.to_dict()["retrieval"]["fusion"] == "weighted"


# --------------------------------------------------------------------- #
# weighted 兼容档：与旧公式逐字节等价
# --------------------------------------------------------------------- #
def test_weighted_mode_matches_legacy_formula():
    rtr = _retriever("weighted")
    v, k, m = 0.7312, 0.4428, 1.0
    c = rtr.config
    legacy = c.vector_weight * v + c.keyword_weight * k + c.metadata_weight * m
    assert rtr._hybrid(v, k, m) == legacy  # 精确相等（同一条 IEEE 运算链）
    assert rtr._channel_weights() == (0.60, 0.30, 0.10)


def test_weighted_vector_is_legacy_linear_map():
    rtr = _retriever("weighted")
    cosines = np.array([1.0, 0.3, 0.0])
    embs = [np.array([1.0, 0.0]), np.array([0.3, math.sqrt(1 - 0.09)]),
            np.array([0.0, 1.0])]
    cands = [_FakeMem(f"m{i}", embedding=e) for i, e in enumerate(embs)]
    out = rtr._vector_scores(np.array([1.0, 0.0]), cands)
    for i, cos in enumerate(cosines):
        assert out[f"m{i}"] == pytest.approx((cos + 1.0) / 2.0)


# --------------------------------------------------------------------- #
# 缺陷 1：BM25 峰值归一化漂移
# --------------------------------------------------------------------- #
_DRIFT_TEXTS = [
    # d0：明显最强匹配（多词高频命中）
    "python pytest coverage python pytest coverage report tool",
    # d1-d7：弱匹配（各命中 1 个查询词）
    "python scripting basics for beginners",
    "pytest fixtures introduction guide",
    "coverage measurement notes and tips",
    "python list comprehension examples",
    "pytest marks and parametrize examples",
    "coverage threshold configuration file",
    "python virtualenv setup steps",
    # d8-d19：完全不相关
] + [f"topic {i} unrelated chatter about weather and coffee" for i in range(12)]

_DRIFT_QUERY = "python pytest coverage"


def _drift_scores(fresh_engine, fusion, candidates):
    fresh_engine.config.retrieval.fusion = fusion
    return fresh_engine.retriever._keyword_scores(_DRIFT_QUERY, candidates)


def test_bm25_peak_drift_weighted_vs_minmax(fresh_engine):
    for t in _DRIFT_TEXTS:
        fresh_engine.add(t)
    mems = sorted(fresh_engine.memories.values(), key=lambda m: m.created_at)
    strongest = mems[0]
    reduced = [m for m in mems if m.id != strongest.id]
    matched = [m.id for m in mems[1:8]]

    drift = {}
    for mode in ("weighted", "minmax"):
        full = _drift_scores(fresh_engine, mode, mems)
        cut = _drift_scores(fresh_engine, mode, reduced)
        drift[mode] = {mid: abs(cut[mid] - full[mid]) for mid in matched}

    # 旧缺陷（weighted）：去掉最强匹配后池内所有匹配文档分数全体膨胀
    assert all(d > 0.3 for d in drift["weighted"].values())

    # minmax：匹配区间上的平均漂移显著减小，最弱匹配（= min 锚点）零漂移
    mean_w = statistics.mean(drift["weighted"].values())
    mean_m = statistics.mean(drift["minmax"].values())
    assert mean_m < mean_w
    minmax_full = _drift_scores(fresh_engine, "minmax", mems)
    weakest = min(matched, key=lambda mid: minmax_full[mid])
    assert drift["minmax"][weakest] == 0.0
    assert drift["weighted"][weakest] > 0.3


def test_bm25_weak_match_alone_no_false_perfect_score(fresh_engine):
    """弱匹配恰好是池内最强：weighted 拿虚假 1.0，minmax 给 0.5（无峰值锚定）。"""
    fresh_engine.add(_DRIFT_TEXTS[1])  # 单个弱匹配成池
    mems = list(fresh_engine.memories.values())
    w = _drift_scores(fresh_engine, "weighted", mems)
    m = _drift_scores(fresh_engine, "minmax", mems)
    assert list(w.values())[0] == 1.0  # 旧缺陷：池内最强恒 1.0
    assert list(m.values())[0] == 0.5  # 全等池 -> 0.5


def test_minmax_keyword_unmatched_stay_zero():
    rtr = _retriever("minmax")
    docs = {
        "d1": "python pytest unit test framework",
        "d2": "python pytest fixture smoke",
        "d3": "python script automation tooling basics",
        "d4": "coffee brewing recipes",
        "d5": "gardening tomatoes soil",
    }
    for k, v in docs.items():
        rtr.bm25.add_document(k, v)
    cands = [_FakeMem(k) for k in docs]
    out = rtr._keyword_scores("python pytest", cands)
    assert out["d4"] == 0.0 and out["d5"] == 0.0  # 不匹配保持 0
    assert out["d2"] == pytest.approx(1.0)         # 最强匹配 -> 1
    assert out["d3"] == pytest.approx(0.0)         # 最弱匹配 -> min 锚点 0


# --------------------------------------------------------------------- #
# 缺陷 2：vector 通道 (cos+1)/2 压缩
# --------------------------------------------------------------------- #
def test_minmax_vector_no_free_half_point():
    embs = [np.array([0.9, math.sqrt(1 - 0.81)]),         # cos 0.9
            np.array([0.6, math.sqrt(1 - 0.36)]),         # cos 0.6
            np.array([0.3, math.sqrt(1 - 0.09)]),         # cos 0.3（弱相关）
            np.array([0.0, 1.0])]                         # 正交
    cands = [_FakeMem(f"m{i}", embedding=e) for i, e in enumerate(embs)]
    q = np.array([1.0, 0.0])

    w = _retriever("weighted")._vector_scores(q, cands)
    assert w["m3"] == pytest.approx(0.5)          # 旧缺陷：正交白拿 0.5
    assert max(w.values()) - min(w.values()) == pytest.approx(0.45)  # 量程被压扁

    m = _retriever("minmax")._vector_scores(q, cands)
    assert m["m3"] == pytest.approx(0.0)          # 正交 = 池内最低 -> 0
    assert m["m0"] == pytest.approx(1.0)
    assert max(m.values()) - min(m.values()) == pytest.approx(1.0)   # 拉满量程


def test_minmax_vector_all_equal_pool_gets_half():
    embs = [np.array([0.6, math.sqrt(1 - 0.36)]),
            np.array([0.6, -math.sqrt(1 - 0.36)])]
    cands = [_FakeMem(f"m{i}", embedding=e) for i, e in enumerate(embs)]
    q = np.array([1.0, 0.0])
    out = _retriever("minmax")._vector_scores(q, cands)
    assert all(v == 0.5 for v in out.values())  # max == min -> 全 0.5


# --------------------------------------------------------------------- #
# 缺陷 3：metadata 死信号删除 + 权重按比例并入
# --------------------------------------------------------------------- #
def test_minmax_drops_dead_metadata_channel():
    rtr = _retriever("minmax")
    vw, kw, mw = rtr._channel_weights()
    assert mw == 0.0
    assert vw == pytest.approx(2 / 3)   # 0.6 / (0.6+0.3)
    assert kw == pytest.approx(1 / 3)
    # hybrid 量程保持 [0, 1]（Ranker semantic 权重的量纲不变）
    assert rtr._hybrid(1.0, 1.0, 1.0) == pytest.approx(1.0)
    assert rtr._hybrid(0.0, 0.0, 1.0) == pytest.approx(0.0)  # meta 不再偏移
    # weighted 档保留常数偏移（兼容）
    w = _retriever("weighted")
    assert w._hybrid(0.0, 0.0, 1.0) == pytest.approx(0.10)


def test_minmax_engine_search_with_metadata_filters(fresh_engine, zh):
    e = fresh_engine
    e.config.retrieval.fusion = "minmax"
    e.add(zh["likes_coffee"], metadata={"user": "alice"})
    e.add(zh["lives_beijing"], metadata={"user": "bob"})
    hits = e.search(zh["q_name"], top_k=5)  # 无过滤
    assert len(hits) == 2
    hits = e.search(zh["q_name"], top_k=5, metadata_filters={"user": "alice"})
    assert [h.memory.metadata["user"] for h in hits] == ["alice"]
    for h in hits:
        assert 0.0 <= h.breakdown.semantic <= 1.0


# --------------------------------------------------------------------- #
# 端到端：minmax 检索质量 + graph_expand / Ranker 下游兼容
# --------------------------------------------------------------------- #
def test_minmax_engine_end_to_end_quality(fresh_engine, zh):
    """多关键词命中场景：minmax 排序正确且区分度不劣于 weighted（top1-top2 间距）。"""
    e = fresh_engine
    cat = zh["cat_tuanzi"]        # 小明的猫叫团子（最强匹配）
    coffee = zh["xiaoming_coffee"]
    company = zh["xiaoming_company"]
    weather = zh["weather_fine"]
    e.config.retrieval.fusion = "weighted"
    for t in (cat, coffee, company, weather):
        e.add(t)
    q = zh["q_cat_name"]
    w_hits = e.search(q, top_k=4)
    assert w_hits[0].memory.text == cat
    w_gap = w_hits[0].score - w_hits[1].score

    # 换 minmax 引擎重建同一语料
    e2_cfg = SMEConfig()
    e2_cfg.storage.autosave = False
    e2_cfg.retrieval.fusion = "minmax"
    from sme.engine import SpatialMemoryEngine
    e2 = SpatialMemoryEngine(e2_cfg)
    for t in (cat, coffee, company, weather):
        e2.add(t)
    m_hits = e2.search(q, top_k=4)
    assert m_hits[0].memory.text == cat
    m_gap = m_hits[0].score - m_hits[1].score
    assert m_gap >= w_gap  # minmax 拉开首尾区分度
    for h in m_hits:
        bd = h.breakdown.to_dict()
        assert 0.0 <= bd["semantic"] <= 1.0
        assert 0.0 <= bd["final"] <= 1.0


def test_minmax_graph_expand_compatible(fresh_engine):
    """"graph_expand 的 merged 排序消费同一 hybrid 管线，minmax 下不炸不越界。"""
    e = fresh_engine
    e.config.retrieval.fusion = "minmax"
    a = e.add("python decorator usage guide advanced")
    b = e.add("python functools wraps implementation detail")
    e.link(a.id, b.id, kind="reference")
    hits = e.search("python decorator usage guide advanced", top_k=1,
                    graph_expand=2)
    assert 1 <= len(hits) <= 1
    assert hits[0].score > 0
    for h in hits:
        assert 0.0 <= h.breakdown.semantic <= 1.0


def test_weighted_default_results_unchanged(fresh_engine, zh):
    """weighted 兼容档搜索结果与旧行为一致：top1 命中、量程含常数偏移。"""
    e = fresh_engine
    e.config.retrieval.fusion = "weighted"  # 默认已切 minmax，这里显式回兼容档
    e.add(zh["a_name"])
    e.add(zh["likes_coffee"])
    hits = e.search(zh["q_name"], top_k=2)
    assert hits[0].memory.text == zh["a_name"]
    # weighted 档 semantic 下限包含 metadata 常数偏移（0.1）+ (cos+1)/2 下限
    assert all(h.breakdown.semantic >= 0.1 for h in hits)
