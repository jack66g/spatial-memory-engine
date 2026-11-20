"""Regression contracts from the 2026-10-23 health check (isolated stores)."""
from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from sme.api.client import MemoryClient
from sme.api.server import create_app
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
from sme.storage.wal import WALRecoveryError


@pytest.fixture
def make(tmp_path):
    engines = []
    def factory(backend="json", mode="fsync", wal=False, qa=False, profile=False):
        cfg = SMEConfig()
        cfg.storage.path = str(tmp_path / ("state.db" if backend == "sqlite" else "state.json.gz"))
        cfg.storage.backend = backend
        cfg.storage.autosave = False
        cfg.persistence.enabled = wal
        cfg.persistence.sync_mode = mode
        cfg.region.auto_evolve = False
        cfg.qapair.enabled = qa
        cfg.profile.enabled = profile
        e = SpatialMemoryEngine(cfg)
        engines.append(e)
        return e
    yield factory
    for e in engines:
        e.wal.close()
        e._maint_stop.set()


def replay_texts(e, ns=None, **kwargs):
    return [h.memory.text for h in e.search("What is the access code?", ns=ns, **kwargs)
            if h.memory.metadata.get("kind") == "qapair_replay"]


def test_cross_user_capture_is_not_consumed(make):
    e = make(qa=True)
    e.add("What is the access code?", ns="alice")
    e.add("BOB-SECRET", ns="bob", source="assistant")
    assert replay_texts(e, "alice") == []
    e.add("ALICE-SECRET", ns="alice", source="assistant")
    assert replay_texts(e, "alice") == ["ALICE-SECRET"]


def test_failed_answer_capture_preserves_pending_and_requires_recovery(make, monkeypatch):
    e = make(qa=True, wal=True)
    e.add("What is the access code?", ns="alice", conversation_id="one")
    def fail(**kwargs):
        raise OSError("simulated QA persistence failure")
    monkeypatch.setattr(e.qapair, "put", fail)
    with pytest.raises(OSError, match="simulated QA"):
        e.add("ALICE-SECRET", ns="alice", conversation_id="one", source="assistant")
    assert ("alice", "one") in e._v2.pending_questions
    assert e.wal.recovery_error
    with pytest.raises(WALRecoveryError):
        e.save()


def test_mixed_concurrent_writes_queries_mutations_and_checkpoints(make):
    e = make(wal=True, qa=True)
    def writer(worker):
        ids = []
        for index in range(30):
            m = e.add(f"project {worker} record {index}", ns=str(worker))
            e.reinforce(m.id)
            if ids:
                e.link(ids[-1], m.id, kind="next")
            ids.append(m.id)
        for mid in ids[:5]:
            e.delete(mid)
        return ids[5:]
    def reader(worker):
        for _ in range(60):
            assert all(h.memory.metadata.get("ns") == str(worker)
                       for h in e.search("project record", ns=str(worker)))
    def saver():
        for _ in range(12):
            e.save()
    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = [pool.submit(writer, w) for w in range(4)]
        readers = [pool.submit(reader, w) for w in range(4)]
        saves = pool.submit(saver)
        retained = [mid for future in futures for mid in future.result()]
        for f in readers:
            f.result()
        saves.result()
    expected = {mid: m.to_dict() for mid, m in e.memories.items()}
    edges = e.graph.to_dict()
    assert set(retained) == set(expected)
    e.wal.flush(); e.wal.close()
    r = make(wal=True, qa=True)
    assert r.load()
    assert {mid: r.get(mid).to_dict() for mid in retained} == expected
    assert r.graph.to_dict() == edges


@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("mode", ["fsync", "grouped"])
def test_export_does_not_checkpoint_configured_store(make, tmp_path, backend, mode):
    e = make(wal=True, backend=backend, mode=mode)
    first = e.add("original record")
    e.save()
    second = e.add("must survive export and restart")
    e.reinforce(first.id)
    expected_hits = first.hit_count
    export = tmp_path / ("export.db" if backend == "sqlite" else "export.json.gz")
    e.save(str(export))
    assert e.wal.has_pending()
    e.wal.close()
    restored = make(wal=True, backend=backend, mode=mode)
    assert restored.load()
    assert restored.get(second.id).text == second.text
    assert restored.get(first.id).hit_count == expected_hits


@pytest.mark.parametrize("restart", [False, True])
def test_conversation_pairing_and_identical_questions(make, restart):
    e = make(qa=True, wal=True)
    for ns, conv in [("alice", "one"), ("bob", "one"), ("alice", "two")]:
        e.add("What is the access code?", ns=ns, conversation_id=conv)
    if restart:
        e.wal.close()
        e = make(qa=True, wal=True)
        assert e.load()
    for ns, conv in [("alice", "two"), ("bob", "one"), ("alice", "one")]:
        e.add(ns + conv, ns=ns, conversation_id=conv, source="assistant")
    assert len(e.qapair.pairs) == 3
    assert {(p.ns, p.conversation_id, p.answer_text) for p in e.qapair.pairs} == {
        ("alice", "one", "aliceone"), ("alice", "two", "alicetwo"), ("bob", "one", "bobone")}
    assert replay_texts(e, "alice", conversation_id="one") == ["aliceone"]


@pytest.mark.parametrize("which", ["question", "answer"])
@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_delete_invalidates_qa_and_recovers(make, which, backend):
    e = make(qa=True, wal=True, backend=backend)
    e.config.extraction.store_assistant = True
    q = e.add("What is the access code?", ns="alice")
    a = e.add("ALICE-SECRET", ns="alice", source="assistant")
    assert replay_texts(e, "alice")
    e.save()
    assert e.delete(q.id if which == "question" else a.id)
    assert replay_texts(e, "alice") == []
    e.wal.close()
    r = make(qa=True, wal=True, backend=backend)
    assert r.load()
    assert replay_texts(r, "alice") == []


def test_qa_filters_and_archive_restore(make):
    e = make(qa=True)
    q = e.add("What is the access code?", ns="alice", tags=["private"], metadata={"project": "alpha"})
    e.add("ALICE-SECRET", ns="alice", source="assistant")
    assert not replay_texts(e, "alice", tags=["missing"])
    assert not replay_texts(e, "alice", metadata_filters={"project": "missing"})
    assert replay_texts(e, "alice", tags=["private"], metadata_filters={"project": "alpha"})
    e.archive(q.id)
    assert not replay_texts(e, "alice")
    assert replay_texts(e, "alice", include_archived=True)
    e.restore(q.id)
    assert replay_texts(e, "alice")


@pytest.mark.parametrize("wal", [False, True])
def test_sdk_batch_namespace_and_single_encoding(make, wal):
    e = make(wal=wal)
    count = []
    original = e.embeddings.embed
    e.embeddings.embed = lambda texts: (count.append(len(texts)) or original(texts))
    memories = e.add_many(["alpha", "beta"], metadata={"ns": "bob"}, ns="alice")
    assert all(m.metadata["ns"] == "alice" for m in memories)
    assert count == [2]


def test_rest_and_client_namespace_and_conversation(make):
    e = make(qa=True)
    client = TestClient(create_app(e))
    assert client.post("/memories/batch", json={"memories": [
        {"text": "What is the access code?", "ns": "alice", "conversation_id": "c"},
        {"text": "SECRET", "source": "assistant", "ns": "alice", "conversation_id": "c"}]}).status_code == 200
    assert e.qapair.pairs[0].ns == "alice"
    assert e.qapair.pairs[0].conversation_id == "c"
    api = MemoryClient("http://testserver")
    api._client = client
    result = api.add("standalone", ns="bob", conversation_id="other")
    assert result["metadata"]["ns"] == "bob"


@pytest.mark.parametrize("item", [{"text": " "}, {"text": "bad", "metadata": [1]},
                                     {"text": "bad", "tags": "tag"}, 3, None])
def test_batch_prevalidates_all_entries(make, item):
    e = make()
    c = TestClient(create_app(e), raise_server_exceptions=False)
    r = c.post("/memories/batch", json={"memories": [{"text": "good"}, item]})
    assert r.status_code in (400, 422), r.text
    assert not e.memories


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_blank_request_is_client_error(make, text):
    c = TestClient(create_app(make()), raise_server_exceptions=False)
    assert c.post("/memories", json={"text": text}).status_code in (400, 422)


def test_batch_runtime_failure_reports_committed_prefix(make, monkeypatch):
    e = make(wal=True)
    c = TestClient(create_app(e), raise_server_exceptions=False)
    original = e.add
    def add(text, **kwargs):
        if text == "fail":
            raise RuntimeError("embedding unavailable")
        return original(text, **kwargs)
    monkeypatch.setattr(e, "add", add)
    r = c.post("/memories/batch", json={"memories": [{"text": "good"}, {"text": "fail"}]})
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail["committed_ids"] == list(e.memories)
    assert detail["failed_index"] == 1 and detail["committed_count"] == 1


def test_scoped_profile_and_context(make):
    e = make(profile=True)
    for ns, text in [("alice", "用户喜欢茉莉茶。"), ("bob", "用户喜欢BOB-SECRET。")]:
        mem = e.add(text, ns=ns, metadata={"fact_kind": "fact"})
        e.profile.upsert(mem)
    e.profile.snapshot()
    msgs = e.context.build(e, [], "我喜欢什么？", [], ns="alice")
    assert "茉莉茶" in str(msgs) and "BOB-SECRET" not in str(msgs)
    scoped = TestClient(create_app(e)).get("/profile", params={"ns": "alice"}).json()
    assert "BOB-SECRET" not in json.dumps(scoped, ensure_ascii=False)


def test_context_total_budget_and_required_content(make):
    e = make()
    e.config.context.budget_tokens = 180
    for i in range(6):
        e.add("长记忆" * 300 + "金额是19，不是91。")
    msgs = e.context.build(e, [("user", "历史" * 200)], "金额是多少？", e.search("金额"))
    assert e.context.budget_used(msgs) <= 180
    assert "金额是多少？" in msgs[-1]["content"]
    e.config.context.budget_tokens = 16
    with pytest.raises(ValueError, match="budget"):
        e.context.build(e, [], "不能丢失的问题" * 100, [])


@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("mode", ["fsync", "grouped"])
def test_every_mutation_recovers_exact_state(make, backend, mode):
    e = make(backend=backend, mode=mode, wal=True)
    a = e.add("alpha original", ns="alice")
    b = e.add("beta original", ns="alice")
    e.save()
    e.reinforce(a.id)
    e.link(a.id, b.id, kind="cause", weight=0.7, note="manual")
    e.update(b.id, text="beta changed")
    c = e.add("new after checkpoint")
    expected = {mid: copy.deepcopy(m.to_dict()) for mid, m in e.memories.items()}
    edges = e.graph.to_dict()
    e.wal.close()
    r = make(backend=backend, mode=mode, wal=True)
    assert r.load()
    for mid, record in expected.items():
        actual = r.get(mid).to_dict()
        for key in record.keys() - {"region_id"}:
            assert actual[key] == record[key], (mid, key)
    assert r.graph.to_dict() == edges
    assert r.get(c.id).created_at == c.created_at


@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("operation", ["consolidate", "compress"])
def test_summary_and_parent_fields_recover(make, backend, operation):
    e = make(backend=backend, wal=True)
    e.config.region.min_join_cosine = 0
    e.config.consolidation.min_group_size = 2
    e.config.consolidation.similarity_threshold = 0
    e.config.compression.min_region_compact = 2
    e.config.compression.age_days_threshold = 0
    e.add_many(["alpha likes apples", "alpha likes apples daily", "alpha likes apples often"])
    e.save()
    summaries = getattr(e, operation)()
    assert summaries
    before = {mid: m.to_dict() for mid, m in e.memories.items()}
    e.wal.close()
    r = make(backend=backend, wal=True)
    assert r.load()
    for mid, record in before.items():
        actual = r.get(mid).to_dict()
        for key in record.keys() - {"region_id"}:
            assert actual[key] == record[key], (mid, key)
    assert r.graph.to_dict() == e.graph.to_dict()
    assert r.consolidation.consolidation_count == e.consolidation.consolidation_count
    assert r.compression.compression_count == e.compression.compression_count


def test_failed_write_blocks_other_mutations_and_save(make, monkeypatch):
    e = make(wal=True)
    a = e.add("durable")
    e.save()
    monkeypatch.setattr(e.wal, "append", lambda op: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        e.reinforce(a.id)
    for action in [lambda: e.update(a.id, importance=0.9), lambda: e.delete(a.id),
                   lambda: e.archive(a.id), lambda: e.link(a.id, a.id), e.save]:
        with pytest.raises(WALRecoveryError):
            action()


@pytest.mark.parametrize("token,legacy,expected", [("mcp", "api", "mcp"), (None, "api", "api"), (None, None, None)])
def test_mcp_auth_headers_without_restart(make, monkeypatch, token, legacy, expected):
    from sme import mcp_server as m
    for key, value in [("SME_MCP_AUTH_TOKEN", token), ("SME_API_AUTH_TOKEN", legacy)]:
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    calls = []
    monkeypatch.setattr(m, "_ensure_service", lambda: None)
    def request(method, url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json={}, request=httpx.Request(method, url))
    monkeypatch.setattr(m.httpx, "request", request)
    m._api("GET", "/stats")
    assert calls[0].get("headers", {}).get("Authorization") == ("Bearer " + expected if expected else None)


def test_legacy_pending_migrates_without_invented_scope(make):
    e = make(qa=True)
    q = e.add("What is the access code?", ns="alice")
    e._restore_module_state({"pending_question": {"question": q.text,
                            "question_memory_id": q.id, "ns": "alice"}})
    e.add("wrong conversation", ns="alice", source="assistant", conversation_id="other")
    assert not e.qapair.pairs
    e.add("legacy answer", ns="alice", source="assistant")
    assert e.qapair.pairs[0].answer_text == "legacy answer"


def test_qa_concurrent_scopes(make):
    e = make(qa=True, wal=True)
    def ask(i):
        e.add("What is the access code?", ns="alice", conversation_id=str(i))
    def answer(i):
        e.add(f"answer-{i}", ns="alice", conversation_id=str(i), source="assistant")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(ask, range(30)))
        list(pool.map(answer, reversed(range(30))))
    assert len(e.qapair.pairs) == 30
    assert all(p.answer_text == f"answer-{p.conversation_id}" for p in e.qapair.pairs)


@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("mode", ["fsync", "grouped"])
def test_archiving_and_deleting_cold_entries_recover(make, backend, mode):
    e = make(backend=backend, mode=mode, wal=True)
    a, b = e.add_many(["archive me", "delete me"])
    e.save()
    e.archive(a.id)
    e.archive(b.id)
    e.delete(b.id)
    e.wal.close()
    r = make(backend=backend, mode=mode, wal=True)
    assert r.load()
    assert r.get(a.id).archived and set(r.archive_manager.cold) == {a.id}
    r.restore(a.id)
    r.wal.close()
    again = make(backend=backend, mode=mode, wal=True)
    assert again.load()
    assert not again.get(a.id).archived and not again.archive_manager.cold


def test_reinforcement_log_does_not_duplicate_vector(make):
    e = make(wal=True)
    a = e.add("record")
    e.save()
    e.reinforce(a.id)
    from pathlib import Path
    log = json.loads(Path(e.wal.path).read_text(encoding="utf-8").splitlines()[-1])
    assert not log["memories"] and "embedding" not in log["patches"][0]


@pytest.mark.parametrize("operation", ["reinforce", "delete", "link"])
def test_failed_commit_keeps_checkpoint_and_requires_recovery(make, monkeypatch, operation):
    e = make(wal=True, qa=True)
    e.config.extraction.store_assistant = True
    a = e.add("What is the access code?", ns="alice")
    b = e.add("answer", ns="alice", source="assistant")
    e.save()
    with monkeypatch.context() as m:
        m.setattr(e.wal, "append", lambda op: (_ for _ in ()).throw(OSError("full")))
        with pytest.raises(OSError):
            if operation == "link":
                e.link(a.id, b.id)
            else:
                getattr(e, operation)(a.id)
        with pytest.raises(WALRecoveryError):
            e.save()
    r = make(wal=True, qa=True)
    assert r.load()
    assert r.get(a.id).hit_count == 0
    assert r.get(a.id) is not None and r.qapair.pairs


def test_failed_embedding_update_leaves_old_record(make, monkeypatch):
    e = make(wal=True)
    a = e.add("original")
    before = a.to_dict()
    monkeypatch.setattr(e.embeddings, "embed_one", lambda text: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError):
        e.update(a.id, text="changed")
    assert a.to_dict() == before


def test_profile_deleted_while_disabled_does_not_reappear(make):
    e = make(profile=True, wal=True)
    a = e.add("用户喜欢茉莉茶", metadata={"fact_kind": "fact"}, ns="alice")
    e.profile.upsert(a)
    e.profile.snapshot(ns="alice")
    e.config.profile.enabled = False
    e.delete(a.id)
    e.save()
    e.config.profile.enabled = True
    assert not e.profile.profile_memories(e.memories, ns="alice")
    assert not e.profile.snapshots


def test_context_exact_counter_and_whole_negation(make):
    e = make()
    e.config.context.budget_tokens = 200
    counter = len
    # The required prompt itself is allowed to reject an insufficient budget.
    with pytest.raises(ValueError, match="budget"):
        e.context.build(e, [], "问题" * 200, [], token_counter=counter)
    e.config.context.budget_tokens = 1000
    e.add("预算金额是19，不是91。")
    messages = e.context.build(e, [("user", "请确认"), ("assistant", "不能删除。")],
                               "预算多少？", e.search("预算"), token_counter=counter)
    assert e.context.budget_used(messages, counter) <= 1000
    assert "预算金额是19，不是91。" in messages[-1]["content"]


def test_graph_shared_source_cleanup(make):
    from sme.models import Entity, Relation
    e = make()
    a, b = e.add_many(["shared source one", "shared source two"])
    graph = e.factgraph
    graph.load_dict({"entities": [Entity(id="x", name="X", memory_ids=[a.id, b.id]).to_dict(),
                                  Entity(id="y", name="Y", memory_ids=[a.id, b.id]).to_dict()],
                     "relations": [Relation(source="x", target="y", predicate="p",
                                             memory_ids=[a.id, b.id]).to_dict()]})
    e.delete(a.id)
    assert graph.entities["x"].memory_ids == [b.id]
    assert graph.relations[0].memory_ids == [b.id]
    e.delete(b.id)
    assert not graph.entities and not graph.relations


@pytest.mark.parametrize("seed", [177, 288, 399])
def test_confirmed_versions_prefer_current_and_keep_controls(make, seed):
    import random
    from sme.models import SearchHit
    from sme.retrieval import SearchQuery
    rng = random.Random(seed)
    e = make()
    e.config.factversion.enabled = True
    for index in range(20):
        subject = f"subject_{seed}_{index}"
        old = e.add(f"{subject} lives in Beijing", ns="alice")
        new = e.add(f"{subject} lives in Shanghai", ns="alice")
        old.metadata["superseded_by"] = new.id
        new.metadata["supersedes"] = old.id
        controls = [e.add(f"{subject} {attribute}", ns="alice") for attribute in
                    ("birthplace Beijing", "age 29", "likes coffee")]
        bob = e.add(f"{subject} lives in Paris", ns="bob")
        old_score, new_score = rng.uniform(.8, 1), rng.uniform(.1, .5)
        hits = [SearchHit(old, old_score), SearchHit(new, new_score)] + [
            SearchHit(m, rng.uniform(.01, .05)) for m in controls + [bob]]
        ranked = e.factversion.prioritize_current(SearchQuery("Where does the subject live now?"), hits, e)
        assert ranked[0].memory.id == new.id
        assert {h.memory.id for h in ranked} == {h.memory.id for h in hits}
        assert all("superseded_by" not in m.metadata for m in controls + [bob])
        historical = [SearchHit(old, old_score), SearchHit(new, new_score)]
        assert e.factversion.prioritize_current(SearchQuery("Where before?"), historical, e) == historical


def test_reranker_cannot_promote_confirmed_stale_ancestor(make, monkeypatch):
    from sme.models import SearchHit
    e = make()
    e.config.factversion.enabled = True
    old, new = e.add_many(["old name", "new name"])
    old.metadata["superseded_by"] = new.id
    new.metadata["supersedes"] = old.id
    e.config.rerank.enabled = True
    monkeypatch.setattr(e.retriever, "search", lambda *a, **kw: [SearchHit(old, .9), SearchHit(new, .1)])
    class Rerank:
        def rerank(self, q, hits):
            return sorted(hits, key=lambda h: h.score, reverse=True)
    e.reranker = Rerank()
    assert e.search("current name", top_k=1)[0].memory.id == new.id
    assert e.search("name history", top_k=1)[0].memory.id == old.id


def test_caller_supplied_vector_survives_wal_without_reencoding(make, monkeypatch):
    import numpy as np
    e = make(wal=True)
    vector = np.arange(64, dtype=float)
    a = e.add("custom vector", embedding=vector)
    e.wal.close()
    r = make(wal=True)
    monkeypatch.setattr(r.embeddings, "embed_one", lambda text: (_ for _ in ()).throw(RuntimeError("offline")))
    assert r.load()
    np.testing.assert_array_equal(r.get(a.id).embedding, vector)


def test_decay_and_reset_survive_without_recalculating_time(make):
    e = make(wal=True)
    e.config.policy.decay_enabled = True
    e.decay.enabled = True
    a = e.add("old fact")
    e.save()
    e.apply_decay(reference=a.last_hit+86400*30)
    expected = a.decay_factor
    assert expected < 1
    e.wal.close()
    r = make(wal=True)
    assert r.load()
    assert r.get(a.id).decay_factor == expected


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_delete_invalidates_generated_summary_but_keeps_other_sources(make, backend):
    e = make(wal=True, backend=backend)
    e.config.region.min_join_cosine = 0
    e.config.consolidation.min_group_size = 2
    e.config.consolidation.similarity_threshold = 0
    a, b = e.add_many(["alpha password SECRET", "alpha connection notes"])
    summaries = e.consolidate()
    assert summaries
    e.save()
    e.delete(a.id)
    assert e.get(b.id) is not None
    assert not any("SECRET" in m.text for m in e.memories.values())
    e.wal.close()
    r = make(wal=True, backend=backend)
    assert r.load()
    assert r.get(b.id) is not None
    assert not any("SECRET" in m.text for m in r.memories.values())


def test_independent_graph_facts_survive_unrelated_delete(make):
    e = make()
    a = e.add("unrelated source")
    x = e.factgraph.upsert_entity("manual X")
    y = e.factgraph.upsert_entity("manual Y")
    rel = e.factgraph.add_relation(x.id, y.id, "manual")
    e.delete(a.id)
    assert x.id in e.factgraph.entities and rel in e.factgraph.relations


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_explicit_grouped_flush_confirms_sequence(make, backend):
    e = make(wal=True, backend=backend, mode="grouped")
    e.add("accepted operation")
    assert e.wal.flush() == e.wal.sequence == 1
    assert not e.wal._buf
    # Flush does not truncate the recovery log.
    assert e.wal.has_pending()
