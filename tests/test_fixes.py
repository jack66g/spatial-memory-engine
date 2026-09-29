"""阶段 A 修复项的回归测试：B1-B4、P1 几何、D3/D8/D9。"""

from __future__ import annotations

import numpy as np
import pytest

from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
from sme.models import Memory


# --------------------------------------------------------------------------- #
# B1: QAPair must work standalone (module 02 without module 01)
# --------------------------------------------------------------------------- #
def test_b1_qapair_standalone_captures_pairs(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.qapair.enabled = True
    e.add(zh["q_name"])
    e.add(zh["a_name"], source="assistant")
    assert e.qapair.count() == 1
    pair = e.qapair.pairs[0]
    assert pair.answer_text == zh["a_name"]


def test_b1_qapair_standalone_replay(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.qapair.enabled = True
    e.config.qapair.similarity_threshold = 0.80
    e.add(zh["q_name"])
    e.add(zh["a_name"], source="assistant")
    hits = e.search(zh["q_name"] + "？", top_k=5)
    assert hits
    assert any("小明" in h.memory.text for h in hits)


def test_b1_qapair_with_extraction_still_works(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.qapair.enabled = True
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["q_name"])
    e.add(zh["a_name"], source="assistant")
    assert e.qapair.count() == 1


# --------------------------------------------------------------------------- #
# B2: namespace isolation for qapair replay + factgraph expansion
# --------------------------------------------------------------------------- #
def test_b2_qapair_replay_isolated_by_ns(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.namespaces.enabled = True
    e.config.qapair.enabled = True
    e.config.qapair.similarity_threshold = 0.80
    e.add(zh["q_name"], ns="user_a")
    e.add(zh["a_name"], source="assistant", ns="user_a")
    # user_a sees the replay
    hits = e.search(zh["q_name"] + "？", top_k=5, ns="user_a")
    assert any(h.memory.metadata.get("kind") == "qapair_replay" for h in hits)
    # user_b must NOT see it
    hits_b = e.search(zh["q_name"] + "？", top_k=5, ns="user_b")
    assert not any(h.memory.metadata.get("kind") == "qapair_replay" for h in hits_b)


def test_b2_qapair_ns_survives_save_load(fresh_engine, zh, tmp_path, new_engine):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "ns.json")
    e.config.namespaces.enabled = True
    e.config.qapair.enabled = True
    e.config.qapair.similarity_threshold = 0.80
    e.add(zh["q_name"], ns="user_a")
    e.add(zh["a_name"], source="assistant", ns="user_a")
    e.save(e.config.storage.path)

    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    e2.config.namespaces.enabled = True
    e2.config.qapair.enabled = True
    assert e2.load(e.config.storage.path) is True
    assert e2.qapair.pairs[0].ns == "user_a"
    hits_b = e2.search(zh["q_name"] + "？", top_k=5, ns="user_b")
    assert not any(h.memory.metadata.get("kind") == "qapair_replay" for h in hits_b)


def test_b2_factgraph_expansion_isolated_by_ns(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.namespaces.enabled = True
    e.config.factgraph.enabled = True
    e.config.factgraph.extract_mode = "rules"
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["colleague_zhang"], ns="user_a")   # 用户的同事是张三
    # a query mentioning 张三 must not pull user_a's memory into user_b
    hits = e.search("张三", top_k=5, ns="user_b")
    assert all(h.memory.metadata.get("ns") == "user_b" or not h.memory.metadata.get("ns") for h in hits)


# --------------------------------------------------------------------------- #
# B3: WAL crash recovery correctness
# --------------------------------------------------------------------------- #
def test_b3_wal_does_not_resurrect_dropped(fresh_engine, zh, tmp_path, new_engine):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "wal.json")
    e.config.persistence.enabled = True
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["likes_coffee"])
    e.save(e.config.storage.path)          # checkpoint
    e.add(zh["haha"])                      # dropped by extraction (chat)
    assert e.wal.ops == 0, "dropped utterance must not enter the WAL"
    assert len(e.memories) == 1

    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    e2.config.persistence.enabled = True
    e2.config.extraction.enabled = True
    e2.config.extraction.mode = "rules"
    assert e2.load(e.config.storage.path) is True
    assert len(e2.memories) == 1
    assert not any(m.text == zh["haha"] for m in e2.memories.values())


def test_b3_wal_preserves_metadata_ns(fresh_engine, zh, tmp_path, new_engine):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "wal2.json")
    e.config.persistence.enabled = True
    e.config.namespaces.enabled = True
    e.add(zh["likes_coffee"], ns="user_a", tags=["fact"], metadata={"k": 1})
    e.save(e.config.storage.path)          # checkpoint
    e.add(zh["lives_beijing"], ns="user_a", tags=["fact"], metadata={"k": 2})
    assert e.wal.ops == 1

    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    e2.config.persistence.enabled = True
    assert e2.load(e.config.storage.path) is True
    assert len(e2.memories) == 2
    mem = next(m for m in e2.memories.values() if m.text == zh["lives_beijing"])
    assert mem.metadata.get("ns") == "user_a"
    assert mem.metadata.get("k") == 2
    assert "fact" in mem.tags


def test_b3_wal_replay_ids_stable(fresh_engine, tmp_path, new_engine):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "wal3.json")
    e.config.persistence.enabled = True
    e.add("stable id memory")
    e.save(e.config.storage.path)
    mid = next(iter(e.memories))
    e.add("another one")
    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    e2.config.persistence.enabled = True
    e2.load(e.config.storage.path)
    assert mid in e2.memories  # replay preserved the original id


# --------------------------------------------------------------------------- #
# B4: REST batch honors the v2 pipeline
# --------------------------------------------------------------------------- #
def test_b4_batch_route_via_pipeline(fresh_engine, zh, new_engine):
    from fastapi.testclient import TestClient

    from sme.api.server import create_app

    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    client = TestClient(create_app(e))
    r = client.post("/memories/batch", json={
        "memories": [
            {"text": zh["haha"]},                # chat -> dropped
            {"text": zh["likes_coffee"]},        # fact -> stored
        ]
    })
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 1
    assert len(e.memories) == 1


def test_b4_batch_v1_behavior_unchanged(fresh_engine, new_engine):
    from fastapi.testclient import TestClient

    from sme.api.server import create_app

    e = fresh_engine
    client = TestClient(create_app(e))
    r = client.post("/memories/batch", json={
        "memories": [{"text": "one"}, {"text": "two"}]
    })
    assert r.status_code == 200
    assert r.json()["added"] == 2
    assert len(e.memories) == 2


# --------------------------------------------------------------------------- #
# P1: incremental geometry is exact
# --------------------------------------------------------------------------- #
def test_p1_incremental_centroid_matches_full_recompute(fresh_engine, new_engine):
    e = fresh_engine
    for i in range(200):
        e.add(f"user likes topic {i % 7} about things {i}")
    space = e.space
    for rid, region in space.regions.items():
        members = [space.vectors[mid] for mid in region.member_ids]
        exact = np.mean(np.stack(members), axis=0)
        err = float(np.abs(region.centroid - exact).max())
        assert err < 1e-9, f"region {rid} centroid drift {err}"
        # deferred geometry is exact after a sync
        region.update_geometry(space.vectors, space.dim)
        assert region._geometry_stale is False


def test_p1_evolution_still_runs_and_splits(fresh_engine, new_engine):
    e = fresh_engine
    # force a dense single topic to grow past the split threshold
    e.config.region.split_threshold = 16
    e.config.region.max_density = 5.0
    e.config.region.evolve_interval = 5
    for i in range(200):
        e.add(f"user likes apple fruit juice day {i}")
    assert e.region_manager.split_count >= 1


def test_p1_stats_and_save_geometry_consistent(fresh_engine, tmp_path, new_engine):
    e = fresh_engine
    for i in range(60):
        e.add(f"user likes topic {i % 4} stuff {i}")
    stats = e.region_stats()
    assert stats.count >= 1
    path = str(tmp_path / "geo.json")
    e.save(path)
    e2 = new_engine()
    e2.load(path)
    s2 = e2.region_stats()
    assert s2.count == stats.count
    assert abs(s2.avg_density - stats.avg_density) < 1e-6


def test_p1_remove_keeps_geometry_consistent(fresh_engine, new_engine):
    e = fresh_engine
    mems = [e.add(f"user likes stable topic {i}") for i in range(30)]
    for m in mems[::2]:
        e.delete(m.id)
    for rid, region in e.space.regions.items():
        members = [e.space.vectors[mid] for mid in region.member_ids]
        if members:
            exact = np.mean(np.stack(members), axis=0)
            assert float(np.abs(region.centroid - exact).max()) < 1e-9


def test_p1_write_throughput_10k(fresh_engine, new_engine):
    """10k writes must finish well under the old O(N^2) wall (was ~118s)."""
    import time

    e = fresh_engine
    texts = [f"user likes topic {i % 50} about things number {i}" for i in range(10000)]
    t0 = time.perf_counter()
    e.add_many(texts)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30, f"10k writes took {elapsed:.1f}s (regression!)"
    assert len(e.memories) == 10000


# --------------------------------------------------------------------------- #
# D3: freshness signal is alive
# --------------------------------------------------------------------------- #
def test_d3_freshness_decays(fresh_engine, new_engine):
    e = fresh_engine
    old = e.add("old memory")
    old.last_hit = 1000000000.0  # ancient last hit
    fresh = e.add("new memory")
    f_old = e.ranker.score(old, e.embeddings.embed_one("x"), 0.0, e, detailed=False)[0]
    f_new = e.ranker.score(fresh, e.embeddings.embed_one("x"), 0.0, e, detailed=False)[0]
    # with identical semantic inputs, the fresh memory scores higher
    assert f_new > f_old


# --------------------------------------------------------------------------- #
# D8: engine.search exposes include_archived
# --------------------------------------------------------------------------- #
def test_d8_search_include_archived(fresh_engine, new_engine):
    e = fresh_engine
    m = e.add("archivable keyword xyz")
    e.archive(m.id)
    assert not e.search("xyz", top_k=5)
    hits = e.search("xyz", top_k=5, include_archived=True)
    assert any(h.memory.id == m.id for h in hits)


# --------------------------------------------------------------------------- #
# D9: provider dim mismatch is detectable
# --------------------------------------------------------------------------- #
def test_d9_sentence_transformer_dim_detected():
    from sme.embedding.factory import build_embedding_provider
    from sme.config import EmbeddingConfig

    cfg = EmbeddingConfig(provider="sentence-transformers",
                          model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                          dim=64)
    try:
        prov = build_embedding_provider(cfg)
    except ImportError:
        pytest.skip("sentence-transformers not installed")
    assert prov.dim != cfg.dim  # the model dimension differs from the config
