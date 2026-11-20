"""Fault injection at recovery/commit boundaries; all state stays in tmp_path."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sme.api.server import _restore_snapshot
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
from sme.models import Fact
from sme.storage import snapshot as snapshots


def engine_at(tmp_path, backend="json", mode="fsync", **modules):
    cfg = SMEConfig()
    cfg.storage.path = str(tmp_path / ("state.db" if backend == "sqlite" else "state.json"))
    cfg.storage.backend = backend
    cfg.storage.autosave = False
    cfg.persistence.enabled = True
    cfg.persistence.sync_mode = mode
    for name, enabled in modules.items():
        getattr(cfg, name).enabled = enabled
    cfg.extraction.mode = "rules"
    return SpatialMemoryEngine(cfg)


@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("mode", ["fsync", "grouped"])
def test_recovery_survives_two_restarts(tmp_path, backend, mode):
    a = engine_at(tmp_path, backend, mode)
    a.add("snapshot base")
    a.save()
    tail = a.add("confirmed WAL tail")
    a.wal.close()
    b = engine_at(tmp_path, backend, mode)
    assert b.load()
    assert tail.id in b.memories
    b.wal.close()
    c = engine_at(tmp_path, backend, mode)
    assert c.load()
    assert tail.id in c.memories
    c.wal.close()


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_failed_recovery_checkpoint_keeps_log_and_snapshot(tmp_path, monkeypatch, backend):
    a = engine_at(tmp_path, backend)
    a.add("base")
    a.save()
    tail = a.add("recover me")
    a.wal.close()
    b = engine_at(tmp_path, backend)
    with monkeypatch.context() as m:
        def fail(*args, **kwargs):
            raise OSError("injected checkpoint failure")
        m.setattr(type(b.storage_backend), "save", fail)
        with pytest.raises(Exception, match="checkpoint|injected"):
            _restore_snapshot(b)
    assert b.wal.has_pending()
    assert not list(tmp_path.glob("*.corrupt-*"))
    b.wal.close()
    c = engine_at(tmp_path, backend)
    assert _restore_snapshot(c)
    assert tail.id in c.memories
    c.wal.close()


def test_apply_failure_retains_wal_and_prevents_partial_checkpoint(tmp_path, monkeypatch):
    a = engine_at(tmp_path)
    a.add("base")
    a.save()
    # Legacy WAL needs re-embedding: a transient provider error must be visible.
    a.wal.append({"op": "add", "mid": "legacy_tail", "text": "recover me"})
    a.wal.close()
    wal_path = Path(a.wal.path)
    before = wal_path.read_bytes()
    b = engine_at(tmp_path)
    def fail(self, text):
        raise RuntimeError("injected embedding failure")
    with monkeypatch.context() as m:
        # load may rebuild the provider from snapshot configuration.
        m.setattr(type(b.embeddings), "embed_one", fail)
        with pytest.raises(Exception, match="WAL|replay|embedding"):
            b.load()
        assert wal_path.read_bytes() == before
        with pytest.raises(Exception, match="recovery|WAL|replay"):
            b.save()
        assert wal_path.read_bytes() == before
    b.wal.close()
    c = engine_at(tmp_path)
    assert c.load()
    assert "legacy_tail" in c.memories
    c.wal.close()


def test_invalid_wal_is_retained_and_reported(tmp_path):
    a = engine_at(tmp_path)
    a.add("base")
    a.save()
    Path(a.wal.path).write_text('{"op": "add", broken\n', encoding="utf-8")
    b = engine_at(tmp_path)
    with pytest.raises(Exception, match="WAL|record|JSON"):
        b.load()
    assert Path(a.wal.path).exists()
    b.wal.close()


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_all_extracted_facts_raw_and_modules_survive_crash(tmp_path, monkeypatch, backend):
    a = engine_at(tmp_path, backend, extraction=True, profile=True, factgraph=True)
    a.config.extraction.keep_raw = True
    a.config.factgraph.extract_mode = "rules"
    a.save()
    facts = [Fact(text="user likes coffee beans", subject="user"),
             Fact(text="user works at Example Labs", subject="user")]
    monkeypatch.setattr(a.extraction, "extract", lambda *args, **kwargs: facts)
    raw = "Original message with both preferences and employer, plus useful context."
    a.add(raw, ns="alice", metadata={"origin": "fixture"})
    expected = {m.id: m.to_dict() for m in a.memories.values()}
    modules = {name: getattr(a, name).to_dict() for name in ("profile", "factgraph")}
    assert len(expected) == 3
    a.wal.close()
    b = engine_at(tmp_path, backend)
    assert b.load()
    actual = {m.id: m.to_dict() for m in b.memories.values()}
    # Regions are a derived index and may be rebuilt during recovery.
    for records in (expected, actual):
        for value in records.values():
            value.pop("region_id", None)
    assert actual == expected
    assert {name: getattr(b, name).to_dict() for name in modules} == modules
    b.wal.close()


def test_correction_and_all_stale_siblings_survive_crash(tmp_path, monkeypatch):
    a = engine_at(tmp_path, extraction=True, factversion=True)
    old = a.add("user likes coffee beans", ns="alice")
    a.save()
    monkeypatch.setattr(a.extraction, "extract", lambda *args, **kwargs: [
        Fact(text="user likes coffee tea", kind="correction", subject="user")])
    new = a.add("correction message", ns="alice")
    assert old.metadata.get("superseded_by") == new.id
    a.wal.close()
    b = engine_at(tmp_path)
    assert b.load()
    assert b.memories[old.id].metadata["superseded_by"] == new.id
    assert b.memories[new.id].metadata["ns"] == "alice"
    assert b.memories[old.id].metadata["invalid_at"] == b.memories[new.id].metadata["valid_at"]
    b.wal.close()


def test_pending_qa_and_answer_survive_separate_restarts(tmp_path):
    a = engine_at(tmp_path, qapair=True)
    a.save()
    a.add("What is your name?", ns="alice")
    a.wal.close()
    b = engine_at(tmp_path)
    assert b.load()
    b.add("My name is Robin", source="assistant", ns="alice")
    assert b.qapair.count() == 1
    expected = b.qapair.to_dict()
    b.wal.close()
    c = engine_at(tmp_path)
    assert c.load()
    assert c.qapair.to_dict() == expected
    c.wal.close()


def test_json_commit_failure_preserves_old_text_and_vector(tmp_path, monkeypatch):
    a = engine_at(tmp_path)
    a.config.persistence.enabled = False
    memory = a.add("original unrelated text")
    old_vector = memory.embedding.copy()
    path = a.save()
    a.update(memory.id, text="completely different new content")
    with monkeypatch.context() as m:
        def fail(*args, **kwargs):
            raise OSError("injected manifest failure")
        m.setattr(snapshots, "_write_json_atomic", fail)
        with pytest.raises(OSError, match="injected"):
            a.save()
    old = snapshots.load_snapshot(path)
    assert old.memories[0].text == "original unrelated text"
    np.testing.assert_array_equal(old.memories[0].embedding, old_vector)


def test_generation_snapshot_rejects_missing_vectors(tmp_path):
    a = engine_at(tmp_path)
    a.config.persistence.enabled = False
    a.add("retained memory")
    path = a.save()
    data = json.loads(snapshots._read_json(path))
    (Path(path).parent / data["embedding_file"]).unlink()
    with pytest.raises(ValueError, match="embedding|vector"):
        snapshots.load_snapshot(path)


def test_checkpoint_bundles_modules_and_removes_obsolete_generations(tmp_path):
    a = engine_at(tmp_path, qapair=True)
    a.add("What is your name?")
    a.add("Robin", source="assistant")
    path = a.save()
    a.add("another useful memory")
    a.save()
    # New checkpoints are self-contained for module state; legacy sidecars
    # may be deleted without changing what the committed snapshot restores.
    for p in tmp_path.glob("*.qapairs.json"):
        p.unlink()
    b = engine_at(tmp_path)
    assert b.load()
    assert b.qapair.count() == 1
    assert len(list(tmp_path.glob("*.embeddings*.npz"))) == 1
    assert not list(tmp_path.glob("*.tmp"))
    b.wal.close()


@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("mode", ["fsync", "grouped"])
@pytest.mark.parametrize("stage", ["before_checkpoint", "before_truncate"])
def test_process_death_during_recovery_is_restart_safe(tmp_path, backend, mode, stage):
    a = engine_at(tmp_path, backend, mode)
    a.add("base")
    a.save()
    tail = a.add("confirmed tail", ns="alice")
    a.wal.close()
    program = r'''
import os, sys
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
cfg = SMEConfig()
cfg.storage.path, cfg.storage.backend = sys.argv[1:3]
cfg.persistence.enabled = True
cfg.storage.autosave = False
e = SpatialMemoryEngine(cfg)
def die(*args, **kwargs):
    os._exit(73)
if sys.argv[3] == "before_checkpoint":
    type(e.storage_backend).save = die
else:
    e.wal.reset = die
e.load()
raise AssertionError("crash point was not reached")
'''
    child = subprocess.run([sys.executable, "-B", "-c", program,
                            a.config.storage.path, backend, stage],
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                           capture_output=True, timeout=30)
    assert child.returncode == 73, child.stderr.decode(errors="replace")
    b = engine_at(tmp_path, backend, mode)
    assert b.load()
    assert tail.id in b.memories
    assert b.memories[tail.id].metadata["ns"] == "alice"
    b.wal.close()
    c = engine_at(tmp_path, backend, mode)
    assert c.load()
    assert {m.text for m in c.memories.values()} == {"base", "confirmed tail"}
    c.wal.close()


def test_old_transaction_cannot_overwrite_a_newer_committed_snapshot(tmp_path, monkeypatch):
    a = engine_at(tmp_path, extraction=True)
    a.add("user likes coffee beans")
    # Simulate a durable snapshot followed by failed WAL cleanup.
    with monkeypatch.context() as m:
        m.setattr(a.wal, "reset", lambda: (_ for _ in ()).throw(OSError("cleanup failed")))
        with pytest.raises(OSError):
            a.save()
    memory = next(iter(a.memories.values()))
    a.update(memory.id, importance=0.95)
    a.save()
    # Reintroducing an obsolete WAL after a committed checkpoint must be safe.
    old_op = {"seq": 1, "op": "transaction", "memories": [
        {**memory.to_dict(), "importance": 0.1}], "modules": {}}
    Path(a.wal.path).write_text(json.dumps(old_op) + "\n", encoding="utf-8")
    b = engine_at(tmp_path)
    assert b.load()
    assert b.memories[memory.id].importance == 0.95
    b.wal.close()


def test_snapshot_fsync_failure_preserves_previous_commit_and_wal(tmp_path, monkeypatch):
    a = engine_at(tmp_path)
    base = a.add("base")
    path = a.save()
    a.add("tail")
    with monkeypatch.context() as m:
        m.setattr(snapshots.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk flush failed")))
        with pytest.raises(OSError, match="disk flush"):
            a.save()
    assert a.wal.has_pending()
    assert [m.id for m in snapshots.load_snapshot(path).memories] == [base.id]
    a.wal.close()
    b = engine_at(tmp_path)
    assert b.load()
    assert {m.text for m in b.memories.values()} == {"base", "tail"}
    b.wal.close()


def test_generation_mismatch_is_detected(tmp_path):
    a = engine_at(tmp_path)
    a.config.persistence.enabled = False
    m = a.add("original")
    path = a.save()
    old = json.loads(snapshots._read_json(path))
    old_bytes = (Path(path).parent/old["embedding_file"]).read_bytes()
    a.update(m.id, text="changed")
    a.save()
    new = json.loads(snapshots._read_json(path))
    (Path(path).parent/new["embedding_file"]).write_bytes(old_bytes)
    with pytest.raises(ValueError, match="generation"):
        snapshots.load_snapshot(path)


def test_failed_wal_probe_never_reports_empty_store(tmp_path, monkeypatch):
    a = engine_at(tmp_path, "sqlite")
    a.wal.append({"op": "add", "mid": "m1", "text": "pending"})
    a.wal.close()
    monkeypatch.setattr(a.wal, "_open", lambda: (_ for _ in ()).throw(OSError("cannot open log")))
    with pytest.raises(Exception, match="Cannot inspect pending WAL"):
        a.wal.has_pending()


def test_disabled_module_state_and_pending_qa_survive_hot_rebuild(tmp_path):
    from sme.api.server import rebuild_engine_with
    a = engine_at(tmp_path, mode="grouped", qapair=True)
    a.add("What is your name?", ns="alice")
    cfg = SMEConfig.from_dict(a.config.to_dict())
    cfg.qapair.enabled = False
    b = rebuild_engine_with(a, cfg)
    b.save()
    b.wal.close()
    c = engine_at(tmp_path)
    assert c.load()
    c.config.qapair.enabled = True
    c.add("Robin", source="assistant", ns="alice")
    assert c.qapair.count() == 1
    c.wal.close()


def test_rebuild_vectors_preserves_module_state(tmp_path):
    from sme.rebuild_embeddings import main
    a = engine_at(tmp_path, qapair=True)
    a.add("What is your name?", ns="alice")
    a.add("Robin", source="assistant", ns="alice")
    expected = a.qapair.to_dict()
    a.save()
    a.wal.close()
    cfg = a.config.to_dict()
    cfg["embedding"]["dim"] = 32
    path = tmp_path/"rebuild-config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert main(["--config", str(path)]) == 0
    b = engine_at(tmp_path)
    assert b.load()
    assert b.qapair.to_dict() == expected
    assert all(len(m.embedding) == 32 for m in b.memories.values())
    b.wal.close()


def test_parallel_search_write_and_checkpoint_recover_every_acknowledged_id(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    e = engine_at(tmp_path, mode="grouped")
    e.config.storage.autosave = True
    e.add("alice starting memory", ns="alice")
    def write(namespace):
        ids = []
        for i in range(60):
            ids.append(e.add(f"{namespace} memory number {i}", ns=namespace).id)
        return ids
    def read():
        for _ in range(80):
            assert all(h.memory.metadata.get("ns") == "alice"
                       for h in e.search("memory number", ns="alice", top_k=5))
    def checkpoint():
        for _ in range(12):
            e.save()
    with ThreadPoolExecutor(max_workers=5) as pool:
        tasks = [pool.submit(write, "alice"), pool.submit(write, "bob"),
                 pool.submit(read), pool.submit(read), pool.submit(checkpoint)]
        acknowledged = set(tasks[0].result()) | set(tasks[1].result())
        for task in tasks[2:]:
            task.result()
    e.wal.close()
    recovered = engine_at(tmp_path)
    assert recovered.load()
    assert acknowledged <= set(recovered.memories)
    assert len(recovered.memories) == 121
    recovered.wal.close()
