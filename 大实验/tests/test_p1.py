"""P1 迭代测试：噪音参数（2.1）/ 候选窗口（2.2）/ rerank（2.3）/ kb 预设（2.4）
/ rules 语气词（2.5）/ 加载并行（2.6）。"""

from __future__ import annotations

import time

import pytest

from sme.config_items import PRESET_BY_KEY, apply_preset, defaults_config


# ------------------------- 2.1 noise tuning -------------------------------- #
def test_noise_new_defaults(fresh_engine):
    from sme.config import NoiseConfig

    cfg = NoiseConfig()
    assert cfg.min_density == 0.10
    assert cfg.template_penalty == 0.3


def test_noise_configurable_parameters(fresh_engine, zh):
    e = fresh_engine
    e.config.noise.enabled = True
    e.config.noise.min_density = 0.9   # everything looks low-density -> penalized
    e.config.noise.template_penalty = 0.1
    e.add("用户周末喜欢打篮球")
    hits = e.search("篮球", top_k=10)
    n = e.noise.factor(e, next(iter(e.memories.values())))
    assert 0 < n <= 1.0
    assert hits


def test_noise_extracted_fact_whitelisted(fresh_engine, zh):
    e = fresh_engine
    e.config.noise.enabled = True
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["likes_coffee"])
    mem = next(iter(e.memories.values()))
    assert mem.metadata.get("fact_kind") == "fact"
    assert e.noise.factor(e, mem) == 1.0  # whitelisted, never penalized


# ------------------------- 2.2 candidate window ---------------------------- #
def test_candidate_window_configurable(fresh_engine):
    e = fresh_engine
    e.config.retrieval.candidate_window = 4   # shrink the pool
    e.add("user likes apples one")
    hits = e.search("apples", top_k=5)
    assert hits


def test_candidate_window_default_keeps_topk_behavior(fresh_engine):
    e = fresh_engine
    for i in range(30):
        e.add(f"user likes apple fruit day {i}")
        e.add(f"user plays basketball sport day {i}")
    e.config.retrieval.candidate_window = 20  # default
    hits = e.search("apple fruit", top_k=10)
    assert len(hits) <= 10
    assert any("apple" in h.memory.text for h in hits)


# ------------------------- 2.3 rerank adapter ------------------------------ #
def test_rerank_disabled_by_default(fresh_engine):
    e = fresh_engine
    assert e.config.rerank.enabled is False
    assert e._rerank_enabled is False
    e.add("user likes tea")
    hits = e.search("tea", top_k=3)
    assert hits


def test_rerank_mock_scorer(fresh_engine, monkeypatch):
    """Wire check: enabled rerank re-orders hits by the model scores."""
    e = fresh_engine
    for i in range(10):
        e.add(f"user likes tea variant {i}")
    e.add("user likes coffee beans")
    e.config.rerank.enabled = True
    e.config.rerank.top_n = 5

    class _FakeModel:
        def predict(self, pairs, convert_to_numpy=True):
            # score coffee-related docs higher
            return [1.0 if "coffee" in doc else 0.1 for _, doc in pairs]

    monkeypatch.setattr(
        "sme.rerank.Reranker._get_model", lambda self: _FakeModel()
    )
    hits = e.search("coffee", top_k=10)
    assert hits
    assert hits[0].memory.text == "user likes coffee beans"


def test_rerank_fallback_on_model_error(fresh_engine, monkeypatch):
    e = fresh_engine
    e.add("user likes tea")
    e.config.rerank.enabled = True

    def boom(self):
        raise RuntimeError("model load failed")

    monkeypatch.setattr("sme.rerank.Reranker._get_model", boom)
    hits = e.search("tea", top_k=3)  # must not raise
    assert hits


# ------------------------- 2.4 kb preset ranking --------------------------- #
def test_kb_preset_ranking_tuning():
    cfg = defaults_config()
    for key in ("kb_dynamic", "kb_static"):
        changes = apply_preset(cfg, PRESET_BY_KEY[key])
        from sme.config_items import get_value, ITEM_BY_PATH

        assert get_value(cfg, ITEM_BY_PATH["ranking.semantic"]) == 0.45
        assert get_value(cfg, ITEM_BY_PATH["ranking.freshness"]) == 0.04
        assert get_value(cfg, ITEM_BY_PATH["ranking.recency"]) == 0.02
        assert changes


def test_chat_preset_ranking_untouched():
    from sme.config_items import get_value, ITEM_BY_PATH

    cfg = defaults_config()
    apply_preset(cfg, PRESET_BY_KEY["chat"])
    assert get_value(cfg, ITEM_BY_PATH["ranking.semantic"]) == 0.40
    assert get_value(cfg, ITEM_BY_PATH["ranking.freshness"]) == 0.10
    assert get_value(cfg, ITEM_BY_PATH["ranking.recency"]) == 0.06


# ------------------------- 2.5 rules particles ----------------------------- #
def test_rules_strip_trailing_particles(fresh_engine, zh):
    from sme.extraction import _rules_extract, _strip_particles

    assert _strip_particles("用户不喜欢喝咖啡了") == "用户不喜欢喝咖啡"
    assert _strip_particles("用户喜欢喝咖啡呢") == "用户喜欢喝咖啡"
    assert _strip_particles("用户住在北京吧") == "用户住在北京"
    facts = _rules_extract("用户不喜欢喝咖啡了")
    assert facts and facts[0].text == "用户不喜欢喝咖啡"


def test_rules_particle_versions_dedup(fresh_engine, zh):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.factversion.enabled = True
    e.add(zh["likes_coffee"])
    e.add("用户喜欢喝咖啡了")   # particle version of the same fact
    assert len(e.memories) == 1, "particle statement must dedup onto the base fact"


# ------------------------- 2.6 parallel load ------------------------------- #
def test_load_100k_under_budget(tmp_path):
    import numpy as np

    from sme.config import SMEConfig
    from sme.engine import SpatialMemoryEngine
    from sme.models import Memory, Region
    from sme.storage import EngineSnapshot, save_snapshot

    rng = np.random.default_rng(42)
    n = 100_000
    mems, vecs = [], {}
    cent = np.zeros(64)
    for i in range(n):
        if i % 2000 == 0:
            cent = rng.standard_normal(64)
            cent /= np.linalg.norm(cent)
        v = cent + rng.standard_normal(64) * 0.1
        v /= np.linalg.norm(v)
        vecs[f"m{i:05d}"] = v
        mems.append(Memory(id=f"m{i:05d}",
                           text=f"user likes topic {i % 50} about things {i}",
                           embedding=v))
    regions = []
    for r in range(50):
        ids = {f"m{i:05d}" for i in range(r * 2000, (r + 1) * 2000)}
        reg = Region(id=f"r{r:02d}", member_ids=ids)
        reg.update_geometry(vecs, 64)
        regions.append(reg)
    path = str(tmp_path / "big100.json.gz")
    save_snapshot(path, EngineSnapshot(
        memories=mems, regions=regions, region_edges=[], memory_edges=[],
        counters={}, config=SMEConfig()))
    e = SpatialMemoryEngine()
    e.config.storage.autosave = False
    t0 = time.perf_counter()
    assert e.load(path) is True
    elapsed = time.perf_counter() - t0
    assert len(e.memories) == n
    # target: <= 30s (measured ~3s)
    assert elapsed < 30, f"100k load took {elapsed:.1f}s"
    assert e.region_stats().count == 50
