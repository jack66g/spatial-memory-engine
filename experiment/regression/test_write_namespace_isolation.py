"""Namespace isolation applies before write-side similarity/deduplication."""
from __future__ import annotations

import pytest

from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
from sme.models import Fact


def make_engine(tmp_path):
    cfg = SMEConfig()
    cfg.storage.path = str(tmp_path / "state.json")
    cfg.storage.autosave = False
    cfg.extraction.enabled = True
    cfg.extraction.mode = "rules"
    cfg.factversion.enabled = True
    return SpatialMemoryEngine(cfg)


@pytest.mark.parametrize("first_ns,second_ns", [("alice", "bob"), (None, "bob"), ("alice", None)])
def test_identical_facts_in_different_namespaces_are_both_stored(tmp_path, first_ns, second_ns):
    e = make_engine(tmp_path)
    a = e.add("user likes coffee beans", ns=first_ns)
    b = e.add("user likes coffee beans", ns=second_ns)
    assert a.id != b.id
    assert len(e.memories) == 2
    assert {m.metadata.get("ns") for m in e.memories.values()} == {first_ns, second_ns}


def test_correction_never_invalidates_foreign_facts(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    alice = e.add("user likes coffee beans", ns="alice")
    bob = e.add("user likes coffee beans", ns="bob")
    monkeypatch.setattr(e.extraction, "extract", lambda *args, **kwargs: [
        Fact(text="user likes coffee tea", kind="correction", subject="user")])
    corrected = e.add("correction", ns="bob", metadata={"origin": "bob-message"})
    assert corrected.metadata.get("ns") == "bob"
    assert corrected.metadata.get("origin") == "bob-message"
    assert corrected.metadata.get("supersedes") == bob.id
    assert "superseded_by" not in alice.metadata
    assert bob.metadata.get("superseded_by") == corrected.id


def test_same_namespace_dedup_still_works(tmp_path):
    e = make_engine(tmp_path)
    e.add("user likes coffee beans", ns="alice")
    e.add("user likes coffee beans", ns="alice")
    assert len(e.memories) == 1


def test_keep_raw_dedup_is_namespace_local(tmp_path, monkeypatch):
    e = make_engine(tmp_path)
    e.config.extraction.keep_raw = True
    monkeypatch.setattr(e.extraction, "extract", lambda *args, **kwargs: [
        Fact(text="user likes coffee beans", subject="user")])
    raw = "Original long message about coffee plus unrelated context to retain."
    e.add(raw, ns="alice")
    e.add(raw, ns="bob")
    assert {m.metadata.get("ns") for m in e.memories.values() if m.text == raw} == {"alice", "bob"}
