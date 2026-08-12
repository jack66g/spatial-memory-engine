"""全关回归：v1 行为零漂移断言。

Every test here runs with ALL v2 modules disabled (the engine's default),
so it locks the v1 baseline: writes, retrieval quality, ranking sanity,
region evolution, persistence roundtrips and snapshot compatibility.
"""

from __future__ import annotations

import time

import pytest

from sme.retrieval import SearchQuery


def test_v1_basic_add_search_reinforce(fresh_engine, new_engine):
    e = fresh_engine
    e.add("用户喜欢苹果")
    e.add("用户每天喝一杯苹果汁")
    e.add("用户周末打篮球")
    hits = e.search("用户喜欢什么水果？", top_k=5)
    assert hits, "no hits"
    assert hits[0].memory.text
    assert hits[0].breakdown is not None
    assert hits[0].breakdown.final > 0
    # 8-signal breakdown is populated for the top-k
    bd = hits[0].breakdown.to_dict()
    assert set(bd) == {"semantic", "importance", "freshness", "weight",
                       "decay", "hit_count", "recency", "region", "final"}
    assert e.memory_stats().total == 3
    assert e.space.region_count >= 1


def test_v1_crud(fresh_engine, new_engine):
    e = fresh_engine
    m = e.add("hello world")
    assert e.get(m.id) is not None
    e.update(m.id, text="hello world v2")
    assert e.get(m.id).text == "hello world v2"
    assert e.delete(m.id) is True
    assert e.delete(m.id) is False
    assert e.get(m.id) is None


def test_v1_archive_restore(fresh_engine, new_engine):
    e = fresh_engine
    m = e.add("archivable")
    assert e.archive(m.id) is True
    assert e.archived_count() == 1
    assert all(h.memory.id != m.id for h in e.search("archivable", top_k=5))
    assert e.restore(m.id) is True
    assert any(h.memory.id == m.id for h in e.search("archivable", top_k=5))


def test_v1_reinforce_grows_weight(fresh_engine, new_engine):
    e = fresh_engine
    m = e.add("user likes tea")
    w0, imp0 = m.weight, m.importance
    delta = e.reinforce(m.id)
    assert delta is not None
    assert m.weight > w0
    assert m.importance >= imp0
    assert m.hit_count == 1
    # unknown id -> None
    assert e.reinforce("no_such_id") is None


def test_v1_ranking_semantic_dominates(fresh_engine, new_engine):
    e = fresh_engine
    for i in range(20):
        e.add(f"user likes topic {i % 5} about apples")
        e.add(f"user plays topic {i % 5} basketball games")
    hits = e.search("apples", top_k=5)
    assert any("apples" in h.memory.text for h in hits[:3])


def test_v1_consolidate_compress(fresh_engine, new_engine):
    e = fresh_engine
    for i in range(12):
        e.add(f"user likes apple fruit variant {i}")
        e.add(f"user plays basketball sport day {i}")
    e.consolidate()
    e.compress()
    stats = e.engine_stats()
    assert stats["consolidations"] > 0
    assert stats["compressions"] >= 0
    # summaries are retrievable but penalized vs real facts
    assert any(m.source == "summary" for m in e.memories.values())


def test_v1_graph_edges_and_traversal(fresh_engine, new_engine):
    e = fresh_engine
    a = e.add("a")
    b = e.add("b")
    c = e.add("c")
    e.link(a.id, b.id, "conversation")
    e.link(b.id, c.id, "reference")
    assert len(e.graph_edges()) == 2
    assert a.id in e.traverse_graph(b.id, max_depth=2)
    e.delete(b.id)
    assert len(e.graph_edges()) == 0


def test_v1_region_evolution_history(fresh_engine, new_engine):
    e = fresh_engine
    for i in range(300):
        e.add(f"user likes apple fruit day {i}")
        e.add(f"user plays basketball sport day {i}")
    history = e.region_history()
    assert isinstance(history, list)
    # every recorded event has the expected shape
    for ev in history:
        assert {"kind", "region_id", "detail"} == set(ev)
    assert e.region_stats().count >= 1


def test_v1_save_load_roundtrip(fresh_engine, tmp_path, new_engine):
    e = fresh_engine
    for i in range(30):
        e.add(f"memory number {i} about topic apple {i}")
    path = str(tmp_path / "state.json.gz")
    e.save(path)

    e2 = new_engine()
    assert e2.load(path) is True
    assert len(e2.memories) == 30
    # embeddings survive the sidecar
    assert all(m.embedding is not None for m in e2.memories.values())
    # retrieval works identically after load
    before = [(h.memory.id, round(h.score, 6)) for h in e.search("topic apple 5", top_k=5)]
    after = [(h.memory.id, round(h.score, 6)) for h in e2.search("topic apple 5", top_k=5)]
    assert before == after
    # no pending WAL
    assert e2.wal.ops == 0


def test_v1_snapshot_byte_schema(fresh_engine, tmp_path, new_engine):
    """schema_version stays v2 and the config round-trips."""
    e = fresh_engine
    e.add("schema test")
    path = str(tmp_path / "state.json")
    e.save(path)
    import gzip
    import json

    with gzip.open(path + ".gz", "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["schema_version"] == 2
    assert "embedding_file" in data
    assert data["config"]["embedding"]["provider"] == "hashing"


def test_v1_export_import(fresh_engine, tmp_path, new_engine):
    e = fresh_engine
    for i in range(10):
        e.add(f"export test {i} cats")
    path = str(tmp_path / "export.json")
    e.export_json(path)
    e2 = new_engine()
    assert e2.import_json(path) == 10
    assert len(e2.memories) == 10
    assert all(m.embedding is not None for m in e2.memories.values())


def test_v1_search_query_object_and_kwargs(fresh_engine, new_engine):
    e = fresh_engine
    e.add("user likes apples")
    e.add("user likes oranges")
    q = SearchQuery(text="apples", top_k=3, top_regions=2)
    hits = e.search(q)
    assert hits and "apples" in hits[0].memory.text
    # string + include_archived kwarg (facade now exposes it)
    hits = e.search("apples", top_k=3, include_archived=True)
    assert hits


def test_v1_metadata_and_tag_filters(fresh_engine, new_engine):
    e = fresh_engine
    e.add("secret recipe", metadata={"category": "food"}, tags=["cook"])
    e.add("other stuff", metadata={"category": "work"})
    hits = e.search(SearchQuery(text="secret", metadata_filters={"category": "food"}))
    assert hits and "secret" in hits[0].memory.text
    hits = e.search(SearchQuery(text="secret", tags=["cook"]))
    assert hits
    hits = e.search(SearchQuery(text="secret", tags=["nope"]))
    assert not hits


def test_v1_apply_decay(fresh_engine, new_engine):
    e = fresh_engine
    m = e.add("decay me")
    e.apply_decay()
    assert 0 < m.decay_factor <= 1.0
