"""v2 模块专项测试：12 个模块逐个开启的行为断言。"""

from __future__ import annotations

import os

import numpy as np
import pytest


# ------------------------------ module 01 ---------------------------------- #
def test_extraction_rules_drops_noise(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["likes_coffee"])
    e.add(zh["haha"])
    e.add(zh["lives_beijing"])
    texts = [m.text for m in e.memories.values()]
    assert len(texts) == 2
    assert zh["likes_coffee"] in texts
    assert zh["haha"] not in texts
    # stored facts carry the extraction metadata
    mem = next(m for m in e.memories.values() if m.text == zh["likes_coffee"])
    assert mem.metadata.get("fact_kind") == "fact"


def test_extraction_assistant_not_stored(fresh_engine, new_engine):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add("用户喜欢苹果")
    e.add("好的我知道了", source="assistant")
    assert len(e.memories) == 1


def test_extraction_rules_question_detection(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add("今天天气怎么样？")
    assert len(e.memories) == 0  # bare question w/o qapair -> dropped


# ------------------------------ module 05 ---------------------------------- #
def test_factversion_correction_supersedes(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.factversion.enabled = True
    e.add(zh["likes_coffee"])
    e.add(zh["corr_no_coffee"])
    superseded = [m for m in e.memories.values() if m.metadata.get("superseded_by")]
    superseding = [m for m in e.memories.values() if m.metadata.get("supersedes")]
    assert superseded and superseding
    # retrieval penalizes the stale version
    hits = e.search("咖啡", top_k=10)
    for h in hits:
        if h.memory.id == superseded[0].id:
            assert h.score < 1.0


def test_factversion_dedup_collapses_repeats(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.factversion.enabled = True
    e.add(zh["likes_coffee"])
    e.add(zh["likes_coffee"])  # same statement again
    assert len(e.memories) == 1


# ------------------------------ module 06 ---------------------------------- #
def test_noise_factor_and_rerank(fresh_engine, new_engine):
    e = fresh_engine
    e.config.noise.enabled = True
    for _ in range(10):
        e.add("what is the weather today")
    e.add("用户周末喜欢打篮球")
    hits = e.search("what is the weather today", top_k=10)
    n = e.noise.factor(e, e.memories[list(e.memories)[0]])
    assert 0 < n <= 1.0
    assert hits


# ------------------------------ module 02 ---------------------------------- #
def test_qapair_sidecar_roundtrip(fresh_engine, zh, tmp_path, new_engine):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "side.json")
    e.config.qapair.enabled = True
    e.add(zh["q_name"])
    e.add(zh["a_name"], source="assistant")
    e.save(e.config.storage.path)
    assert os.path.exists(e._sidecar_path("qapairs"))

    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    assert e2.load(e.config.storage.path) is True
    assert e2.qapair.count() == 1
    assert e2.qapair.pairs[0].answer_text == zh["a_name"]


# ------------------------------ module 03 ---------------------------------- #
def test_factgraph_rules_entities(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.factgraph.enabled = True
    e.config.factgraph.extract_mode = "rules"
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["likes_coffee"])
    e.add(zh["colleague_zhang"])
    assert e.factgraph.entities
    found = e.factgraph.find_entities("张三")
    assert found


def test_factgraph_sidecar_roundtrip(fresh_engine, zh, tmp_path, new_engine):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "fg.json")
    e.config.factgraph.enabled = True
    e.config.factgraph.extract_mode = "rules"
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["colleague_zhang"])
    e.save(e.config.storage.path)
    assert os.path.exists(e._sidecar_path("factgraph"))
    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    e2.load(e.config.storage.path)
    assert e2.factgraph.entities


# ------------------------------ module 04 ---------------------------------- #
def test_profile_upsert_and_snapshot(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.profile.enabled = True
    e.config.profile.snapshot_every = 2
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["likes_coffee"])
    e.add(zh["works_company"])
    assert len(e.profile.profile_facts) == 2
    assert e.profile.snapshots.get("default") is not None
    # boost during retrieval
    hits = e.search("用户喜欢什么", top_k=10)
    tagged = [h for h in hits if "profile_fact" in h.memory.tags]
    assert tagged


# ------------------------------ module 07 ---------------------------------- #
def test_wal_basic_flow(fresh_engine, tmp_path, new_engine):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "w.json")
    e.config.persistence.enabled = True
    for i in range(5):
        e.add(f"wal memory {i}")
    e.save(e.config.storage.path)
    for i in range(3):
        e.add(f"wal new {i}")
    assert e.wal.ops == 3

    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    e2.config.persistence.enabled = True
    assert e2.load(e.config.storage.path) is True
    assert len(e2.memories) == 8
    assert e2.wal.ops == 0  # replay truncated the log


# ------------------------------ module 09 ---------------------------------- #
def test_rest_api_crud_and_search(fresh_engine, zh, new_engine):
    from fastapi.testclient import TestClient

    from sme.api.server import create_app

    client = TestClient(create_app(fresh_engine))
    r = client.post("/memories", json={"text": zh["likes_coffee"], "tags": ["fact"]})
    assert r.status_code == 200
    mid = r.json()["id"]
    assert client.get(f"/memories/{mid}").json()["text"] == zh["likes_coffee"]
    assert client.get("/memories/does_not_exist").status_code == 404
    assert client.patch(f"/memories/{mid}", json={"text": "updated"}).status_code == 200
    assert client.post("/memories/search", json={"text": "updated"}).json()["count"] >= 1
    assert client.delete(f"/memories/{mid}").status_code == 200
    assert client.delete(f"/memories/{mid}").status_code == 404


def test_rest_auth_token():
    from fastapi.testclient import TestClient

    from sme.api.server import create_app
    from sme.engine import SpatialMemoryEngine

    engine = SpatialMemoryEngine()
    engine.config.storage.autosave = False
    engine.config.api.auth_token = "sekret"
    client = TestClient(create_app(engine))
    assert client.get("/health").status_code == 200  # open
    assert client.get("/memories/abc").status_code == 401
    r = client.get("/memories/abc", headers={"Authorization": "Bearer sekret"})
    assert r.status_code == 404


# ------------------------------ module 10 ---------------------------------- #
def test_telemetry_summary_and_export(fresh_engine, tmp_path, new_engine):
    e = fresh_engine
    e.config.observability.enabled = True
    e.add("telemetry test")
    e.search("telemetry", top_k=3)
    e.search("nothing matches this", top_k=3)
    s = e.telemetry.summary()
    assert s["searches"] == 2
    assert s["adds"] == 1
    p = str(tmp_path / "report.json")
    assert e.telemetry.export_json(p) == p
    assert os.path.exists(p)


# ------------------------------ module 11 ---------------------------------- #
def test_context_build_layers(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.context.enabled = True
    e.config.profile.enabled = True
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.add(zh["likes_coffee"])
    msgs = e.context.build(e, [("user", "hi"), ("assistant", "hello")],
                           "你叫什么", [], window_rounds=3)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "用户画像" in msgs[1]["content"] or "（暂无相关记忆）" in msgs[1]["content"]


# ------------------------------ module 12 ---------------------------------- #
def test_namespaces_isolation(fresh_engine, zh, new_engine):
    e = fresh_engine
    e.config.namespaces.enabled = True
    e.add(zh["likes_coffee"], ns="user_a")
    e.add(zh["lives_beijing"], ns="user_b")
    hits = e.search("咖啡", top_k=5, ns="user_a")
    assert hits
    assert all(h.memory.metadata.get("ns") == "user_a" for h in hits)
    assert len(e.namespaces.view(e, "user_a").memories()) == 1


# ------------------------------ module 08 ---------------------------------- #
def test_sqlite_backend_roundtrip(fresh_engine, tmp_path):
    e = fresh_engine
    e.config.storage.backend = "sqlite"
    path = str(tmp_path / "state.db")
    for i in range(20):
        e.add(f"sqlite memory {i} about dogs")
    e.save(path)
    e2 = fresh_engine
    e2.config.storage.backend = "sqlite"
    assert e2.load(path) is True
    assert len(e2.memories) == 20
    assert all(m.embedding is not None for m in e2.memories.values())
    hits = e2.search("dogs 5", top_k=3)
    assert hits


def test_sqlite_backend_query_vectors(fresh_engine, tmp_path):
    """SqliteBackend.query_vectors runs a cosine search over the stored DB."""
    from sme.storage_backends import SqliteBackend

    e = fresh_engine
    e.config.storage.backend = "sqlite"
    path = str(tmp_path / "qv.db")
    for i in range(30):
        e.add(f"user likes topic {i % 5} about dogs {i}")
    e.save(path)
    backend = SqliteBackend()
    q = e.embeddings.embed_one("dogs about topic 3")
    rows = backend.query_vectors(path, q, top_k=5)
    assert len(rows) == 5
    assert rows[0][1] is not None  # (cosine, memory_id)
    assert all(0.0 <= r[0] <= 1.0 for r in rows)


def test_sqlite_wal_incremental_channel(fresh_engine, tmp_path):
    """WAL ops live inside the sqlite db (iteration 3.3)."""
    e = fresh_engine
    e.config.storage.backend = "sqlite"
    e.config.persistence.enabled = True
    path = str(tmp_path / "wal.db")
    e.config.storage.path = path
    for i in range(4):
        e.add(f"sqlite wal memory {i}")
    e.save(path)                       # checkpoint
    e.add("sqlite wal new one")        # post-checkpoint op -> wal_ops table
    assert e.wal.sqlite is True
    assert e.wal.ops == 1

    e2 = fresh_engine
    e2.config.storage.backend = "sqlite"
    e2.config.storage.path = path
    e2.config.persistence.enabled = True
    assert e2.load(path) is True
    assert len(e2.memories) == 5, "sqlite WAL replay failed"
    assert e2.wal.ops == 0


# ------------------------------ shared infra ------------------------------- #
def test_ann_index_fallback_and_hnsw():
    from sme.index.ann import ANNIndex

    idx = ANNIndex(4, metric="cosine", use_hnsw=None)
    for i in range(50):
        v = np.array([1.0, i % 5, (i * 7) % 11, i / 10.0])
        idx.add(f"k{i}", v)
    out = idx.query(np.array([1.0, 2.0, 3.0, 0.5]), 5)
    assert len(out) == 5
    idx.remove("k10")
    assert "k10" not in idx
    idx.rebuild({f"n{i}": np.ones(4) * i for i in range(10)})
    assert len(idx) == 10


def test_tokenize_cjk(fresh_engine, new_engine):
    from sme.utils import tokenize

    toks = tokenize("用户喜欢喝咖啡")
    assert toks == list("用户喜欢喝咖啡")
    toks = tokenize("user likes tea")
    assert toks == ["user", "likes", "tea"]


def test_visualize(fresh_engine, tmp_path, new_engine):
    for i in range(20):
        fresh_engine.add(f"visual topic {i % 3} group")
    out = str(tmp_path / "space.png")
    p = fresh_engine.visualize(out)
    assert os.path.exists(p)
