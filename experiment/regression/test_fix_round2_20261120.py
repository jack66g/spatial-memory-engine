# -*- coding: utf-8 -*-
"""2026-11-20 修复轮·第二批(战报 §6 留存清项)回归。

P2-1  混合拉丁+CJK 分词:拉丁整词从 BM25 索引消失
P2-5  演化候选饥饿:超额候选被 bulk clear,大 Region 永不拆分
P3-6  merge_regions 不清理被吸收 Region 的第三方邻边(悬空边)
P3-14 热重建把 persistence True→False 时丢弃未 checkpoint 的 WAL 窗口
P4    rerank 分数不同步 / kb 预设权重和 0.95 / 孤立代理字符 / top_k 类型 /
      ghost id 裸 KeyError / _is_blank 重复定义 / QUESTION_WORDS 重复项
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sme import SpatialMemoryEngine, SMEConfig
from sme.config_items import PRESET_BY_KEY
from sme.utils import tokenize


# ------------------------------------------------------------------ #
# P2-1: 混合分词
# ------------------------------------------------------------------ #
def test_tokenize_mixed_run_keeps_latin_words():
    toks = tokenize("GPT模型的推理能力很强", cjk_bigram=True)
    assert "gpt" in toks, f"拉丁整词丢失: {toks}"
    assert "模型" in toks          # CJK bigram 保留
    toks2 = tokenize("iPhone手机的价格", cjk_bigram=True)
    assert "iphone" in toks2
    # 纯 CJK / 纯拉丁行为不变
    assert tokenize("违约金", cjk_bigram=True) == ["违", "约", "金", "违约", "约金"]
    assert tokenize("hello world", cjk_bigram=True) == ["hello", "world"]


def test_mixed_word_bm25_recall(fresh_engine):
    fresh_engine.add("GPT模型的推理能力很强")
    fresh_engine.add("iPhone手机的价格很贵")
    fresh_engine.add("Python语言的生态很好")
    # 纯拉丁查询必须能通过关键词通道命中混合词文档
    for q, expect in [("GPT", "GPT模型"), ("iPhone", "iPhone手机"), ("Python", "Python语言")]:
        hits = fresh_engine.search(q, top_k=3)
        assert hits and expect in hits[0].memory.text, f"search({q!r}) top1={hits[0].memory.text if hits else None}"


# ------------------------------------------------------------------ #
# P2-5: 候选饥饿
# ------------------------------------------------------------------ #
def test_split_candidates_carry_over_when_capped(monkeypatch, tmp_path):
    import sme.space.region_manager as rm

    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "s.json")
    cfg.region.split_threshold = 4
    cfg.region.min_region_size = 2
    cfg.region.evolve_interval = 1
    cfg.region.max_density = 0.0          # 密度门全开,只看规模
    cfg.region.auto_evolve = False        # 灌数据期间不演化,让超阈候选堆起来
    e = SpatialMemoryEngine(cfg)
    import numpy as np
    dim = e.config.embedding.dim
    # 12 个正交簇(独热主方向 + 每成员微小独立扰动保证可分),每簇 6 条
    # → 12 个必然超阈(size>=4)的 Region(全同向量会让 split 的种子选取
    # seed_a==seed_b 而失败,故必须加扰动)
    for c in range(12):
        v = np.zeros(dim)
        v[c % dim] = 1.0
        for i in range(6):
            vi = v.copy()
            vi[(c * 7 + i * 3 + 13) % dim] += 0.05
            e.add(f"簇{c}的记忆{i}", embedding=vi)
    n_candidates = len(e.space._split_candidates)
    assert n_candidates >= 10, f"前置条件不成立: 仅 {n_candidates} 个超阈候选"

    capped = rm.MAX_SPLITS_PER_PASS
    monkeypatch.setattr(rm, "MAX_SPLITS_PER_PASS", 2)
    e.region_manager.evolution_pass(e.space)  # 单轮:最多拆 2 个
    leftover = len(e.space._split_candidates)
    assert leftover >= n_candidates - 2, (
        f"超额候选被丢弃(候选饥饿): {n_candidates} 候选单轮拆 2 后应剩 >={n_candidates - 2}, 实剩 {leftover}"
    )
    monkeypatch.setattr(rm, "MAX_SPLITS_PER_PASS", capped)
    # 多轮演化后全部拆完(候选不再永久滞留)
    for _ in range(10):
        e.region_manager.evolution_pass(e.space)
    over = [r for r in e.space.regions.values() if r.size >= cfg.region.split_threshold * 2]
    assert not over, f"仍有 {len(over)} 个未拆分的大 Region: {[r.size for r in over]}"


# ------------------------------------------------------------------ #
# P3-6: merge 悬空邻边
# ------------------------------------------------------------------ #
def test_merge_prunes_absorbed_region_edges(tmp_path):
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "m.json")
    e = SpatialMemoryEngine(cfg)
    ids = []
    for c in range(4):
        for i in range(3):
            ids.append(e.add(f"簇{c}记忆{i}内容{c}"))
    e.region_manager.evolution_pass(e.space)
    rids = list(e.space.regions)
    if len(rids) >= 2:
        a, b = rids[0], rids[1]
        e.region_manager.merge_regions(e.space, a, b)
        dangling = [k for k, ed in e.space.region_edges.items()
                    if ed.source not in e.space.regions or ed.target not in e.space.regions]
        assert not dangling, f"merge 后残留悬空边: {dangling}"


# ------------------------------------------------------------------ #
# P3-14: 热重建 persistence off → 立即 checkpoint
# ------------------------------------------------------------------ #
def test_rebuild_persistence_off_checkpoints_to_disk(tmp_path):
    from fastapi.testclient import TestClient
    from sme.api.server import create_app

    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "wal.json")
    cfg.persistence.enabled = True
    engine = SpatialMemoryEngine(cfg)
    client = TestClient(create_app(engine, config_file=str(tmp_path / "c.json")))
    for i in range(10):
        assert client.post("/memories", json={"text": f"重建前记忆{i}内容{i}"}).status_code == 200
    r = client.put("/config", json={"values": {"persistence.enabled": False}})
    assert r.status_code == 200 and r.json().get("ok"), r.text[:200]
    # 磁盘上的恢复存储必须已包含全部记忆(重启不回退)
    checker = SpatialMemoryEngine(SMEConfig())
    checker.config.storage.path = str(tmp_path / "wal.json")
    e2 = SpatialMemoryEngine(SMEConfig())
    e2.config.storage.path = str(tmp_path / "wal.json")
    assert e2.load(str(tmp_path / "wal.json"))
    texts = [m.text for m in e2.memories.values()]
    assert sum(1 for t in texts if t.startswith("重建前记忆")) == 10, f"回退!只剩 {len(texts)} 条"


# ------------------------------------------------------------------ #
# P4: rerank 分数同步(伪打分器,免 sentence-transformers 依赖)
# ------------------------------------------------------------------ #
def test_rerank_syncs_scores(tmp_path):
    from sme.retrieval.rerank import Reranker
    from sme.models import Memory, ScoreBreakdown

    class FakeHit:
        def __init__(self, mid, text):
            self.memory = Memory(id=mid, text=text)
            self.score = 0.0
            self.breakdown = ScoreBreakdown()

    rr = Reranker(SMEConfig().rerank)
    rr.config.enabled = True

    class FakeModel:
        def predict(self, pairs, convert_to_numpy=True):
            return [0.1, 0.9, 0.5]

    monkey_model = FakeModel()
    rr.config.model = "fake"
    rr._model = monkey_model
    rr._name = "fake"
    hits = [FakeHit("a", "文本A"), FakeHit("b", "文本B"), FakeHit("c", "文本C")]
    out = rr.rerank("查询", hits, top_n=3)
    assert [h.memory.id for h in out] == ["b", "c", "a"]
    assert out[0].score == 0.9 and out[1].score == 0.5 and out[2].score == 0.1
    assert out[0].breakdown.final == 0.9, "breakdown.final 未与 score 同步"


# ------------------------------------------------------------------ #
# P4: kb 预设权重和 = 1.0
# ------------------------------------------------------------------ #
def test_kb_presets_ranking_sum_is_one():
    for key in ("kb_dynamic", "kb_static"):
        preset = PRESET_BY_KEY[key]
        values = dict(preset["values"])
        # 未列出的 ranking 键取 dataclass 默认
        from sme.config import RankingConfig
        rc = RankingConfig()
        total = 0.0
        for f in ("semantic", "importance", "freshness", "weight",
                  "decay", "hit_count", "recency", "region"):
            total += values.get(f"ranking.{f}", getattr(rc, f))
        assert abs(total - 1.0) < 1e-6, f"{key} ranking 权重和 = {total}"


# ------------------------------------------------------------------ #
# P4: 孤立代理字符 / top_k 类型 / ghost id
# ------------------------------------------------------------------ #
def test_lone_surrogate_rejected(fresh_engine):
    with pytest.raises(ValueError, match="invalid unicode"):
        fresh_engine.add("坏字符\ud800在中间")
    m = fresh_engine.add("正常记忆")
    with pytest.raises(ValueError, match="invalid unicode"):
        fresh_engine.update(m.id, text="更新\udfff坏")


def test_search_top_k_type_guard(fresh_engine):
    fresh_engine.add("类型守卫记忆")
    with pytest.raises(ValueError, match="top_k"):
        fresh_engine.search("守卫", top_k=3.7)
    assert len(fresh_engine.search("守卫", top_k=3.0)) >= 1   # 整数 float 放行


def test_ghost_id_value_error(fresh_engine):
    with pytest.raises(ValueError, match="not found"):
        fresh_engine.update("ghost_id", text="x")


# ------------------------------------------------------------------ #
# P4: 源级卫生
# ------------------------------------------------------------------ #
def test_no_duplicate_defs_and_words():
    import inspect
    import sme.config_items as ci
    src = inspect.getsource(ci)
    assert src.count("def _is_blank") == 1, "_is_blank 重复定义回归"
    from sme.modules.extraction import QUESTION_WORDS
    assert len(QUESTION_WORDS) == len(set(QUESTION_WORDS)), "QUESTION_WORDS 含重复项"
