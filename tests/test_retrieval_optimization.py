"""Behavioral regressions for recall, rerank, rejection and inference locks."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sme.config import SMEConfig
from sme.embedding.base import EmbeddingProvider
from sme.engine import SpatialMemoryEngine
from sme.models import SearchHit
from sme.retrieval import SearchQuery


def make_engine(tmp_path):
    cfg = SMEConfig()
    cfg.storage.path = str(tmp_path / "state.json")
    cfg.storage.autosave = False
    cfg.policy.decay_enabled = False
    cfg.region.auto_evolve = False
    return SpatialMemoryEngine(cfg)


def vector(x, y):
    v = np.zeros(64)
    v[:2] = (x, y)
    return v


def test_bm25_independent_recall_crosses_region_gate_and_filters_first(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    e.config.retrieval.top_regions = 1
    e.config.retrieval.candidate_window = 2
    e.config.retrieval.keyword_candidates = 1
    e.config.retrieval.vector_weight = 0
    e.config.retrieval.keyword_weight = 1
    e.config.ranking.semantic = 1
    for key in ("importance", "freshness", "weight", "decay", "hit_count", "recency", "region"):
        setattr(e.config.ranking, key, 0)
    for i in range(5):
        e.add(f"nearby unrelated topic {i}", ns="alice", embedding=vector(1, 0))
    target = e.add("ERR_XQ917 restore instructions", ns="alice", embedding=vector(0, 1))
    e.add("ERR_XQ917", ns="bob", embedding=vector(0, 1))
    monkeypatch.setattr(e.embeddings, "embed_one", lambda text: vector(1, 0))
    hits = e.search("ERR_XQ917", top_k=1, ns="alice")
    assert [h.memory.id for h in hits] == [target.id]


def test_rerank_can_recover_an_answer_below_requested_topk(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    memories = [e.add(f"candidate number {i}") for i in range(8)]
    e.config.rerank.enabled = True
    e.config.rerank.top_n = 8
    def retrieve(engine, query, **kwargs):
        return [SearchHit(memory=m, score=1-i/10) for i, m in enumerate(memories[:query.top_k])]
    monkeypatch.setattr(e.retriever, "search", retrieve)
    class Model:
        def predict(self, pairs, **kwargs):
            return [10 if text.endswith("7") else 0 for _, text in pairs]
    monkeypatch.setattr("sme.retrieval.rerank.Reranker._get_model", lambda self: Model())
    query = SearchQuery("question", top_k=2)
    hits = e.search(query)
    assert len(hits) == 2
    assert hits[0].memory.id == memories[7].id
    assert query.top_k == 2


def test_query_instruction_never_changes_document_embeddings():
    class Provider(EmbeddingProvider):
        def __init__(self):
            self.inputs = []
        def embed(self, texts):
            self.inputs.extend(texts)
            return [vector(1, 0) for _ in texts]
    p = Provider()
    p.query_prefix = "Instruct: retrieve facts\nQuery: "
    p.embed_one("document")
    p.embed_query("question")
    assert p.inputs == ["document", "Instruct: retrieve facts\nQuery: question"]


def test_query_cache_does_not_cache_results_and_is_bounded(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    e.config.retrieval.query_cache_size = 2
    e.add("first", embedding=vector(0, 1))
    calls = []
    def encode(text):
        calls.append(text)
        return vector(1, 0)
    monkeypatch.setattr(e.embeddings, "embed_one", encode)
    e.search("question")
    added = e.add("second", embedding=vector(1, 0))
    assert e.search("question", top_k=1)[0].memory.id == added.id
    assert calls == ["question"]
    for query in ("q2", "q3", "question"):
        e.search(query)
    assert calls == ["question", "q2", "q3", "question"]
    assert len(e.retriever._query_cache) == 2


def test_query_cache_keys_include_query_prefix(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    calls = []
    monkeypatch.setattr(e.embeddings, "embed_one", lambda text: calls.append(text) or vector(1, 0))
    e.search("same question")
    e.embeddings.query_prefix = "new prefix "
    e.search("same question")
    assert calls == ["same question", "new prefix same question"]


def test_optional_rejection_uses_raw_cosine_not_relative_ranking(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    e.add("unrelated", embedding=vector(0, 1))
    monkeypatch.setattr(e.embeddings, "embed_one", lambda text: vector(1, 0))
    assert e.search("unknown question")  # compatibility default
    e.config.retrieval.min_similarity = 0.5
    assert e.search("unknown question") == []
    relevant = e.add("relevant answer", embedding=vector(1, 0))
    assert [h.memory.id for h in e.search("unknown question")] == [relevant.id]


def test_query_inference_does_not_block_an_independent_write(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    e.add("existing", embedding=vector(1, 0))
    entered, release = threading.Event(), threading.Event()
    def slow_query(text):
        entered.set()
        assert release.wait(5)
        return vector(1, 0)
    monkeypatch.setattr(e.embeddings, "embed_one", slow_query)
    with ThreadPoolExecutor(max_workers=2) as pool:
        search = pool.submit(e.search, "slow query")
        assert entered.wait(2)
        try:
            write = pool.submit(e.add, "independent", embedding=vector(0, 1))
            added = write.result(timeout=1)
        finally:
            release.set()
        assert added.id in e.memories
        search.result(timeout=2)


def test_provider_rebind_retries_query_in_the_new_space(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    target = e.add("new-space target", embedding=vector(0, 1))
    old = e.embeddings
    class NewProvider:
        query_prefix = ""
        def embed_one(self, text):
            return vector(0, 1)
    new = NewProvider()
    def old_encode(text):
        e.embeddings = new
        return vector(1, 0)
    monkeypatch.setattr(old, "embed_one", old_encode)
    e.config.retrieval.min_similarity = 0.9
    assert e.search("question")[0].memory.id == target.id


def test_unhashable_provider_and_runtime_cache_resize(tmp_path):
    from dataclasses import dataclass
    e = make_engine(tmp_path)
    @dataclass
    class Provider:
        query_prefix: str = ""
        def embed_one(self, text):
            return vector(1, 0)
    e.embeddings = Provider()
    for query in ("q1", "q2", "q3"):
        e.search(query)
    e.config.retrieval.query_cache_size = 1
    e.search("q3")
    assert len(e.retriever._query_cache) == 1
    e.config.retrieval.query_cache_size = 0
    e.search("q3")
    assert not e.retriever._query_cache


def test_new_settings_are_serialized_and_validated():
    from sme.config_items import validate_config
    from sme.config import EmbeddingConfig
    legacy = EmbeddingConfig("hashing", "model", 64, "", "", 32, False)
    assert legacy.normalize is False and legacy.query_prefix == ""
    cfg = SMEConfig.from_dict({"embedding": {"query_prefix": "query: "},
                              "retrieval": {"keyword_candidates": 20, "min_similarity": 0.4,
                                            "query_cache_size": 64}})
    cfg.llm.reasoning_effort = "none"
    assert SMEConfig.from_dict(cfg.to_dict()).embedding.query_prefix == "query: "
    assert cfg.retrieval.keyword_candidates == 20
    # Validation should reject unsafe budgets/floors before hot rebuilding.
    assert not validate_config(cfg.to_dict())
    cfg.retrieval.keyword_candidates = -2
    cfg.retrieval.min_similarity = 2
    result = validate_config(cfg.to_dict())
    assert any("keyword_candidates" in error for error in result)
    assert any("min_similarity" in error for error in result)


def test_cached_vectors_do_not_pin_an_obsolete_model(tmp_path):
    import gc
    import weakref
    e = make_engine(tmp_path)
    class Provider:
        def embed_one(self, text):
            return vector(1, 0)
    provider = Provider()
    reference = weakref.ref(provider)
    e.retriever.prepare_query(provider, "question")
    del provider
    gc.collect()
    assert reference() is None


def test_load_rebinds_write_provider_before_wal_replay(tmp_path):
    cfg = SMEConfig()
    cfg.embedding.dim = 32
    cfg.storage.path = str(tmp_path/"dim.json")
    cfg.storage.autosave = False
    cfg.persistence.enabled = True
    a = SpatialMemoryEngine(cfg)
    a.add("base in new space")
    a.save()
    tail = a.add("tail in new space")
    a.wal.close()
    b = make_engine(tmp_path)  # starts with the default 64-dimensional provider
    assert b.load(cfg.storage.path)
    assert b.memories[tail.id].embedding.shape == (32,)
    assert b.add("new write after recovery").embedding.shape == (32,)
    b.wal.close()
