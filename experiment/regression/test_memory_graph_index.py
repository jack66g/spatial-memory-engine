# -*- coding: utf-8 -*-
"""MemoryGraph adjacency-index regression tests.

Covers: index/edges reconciliation invariants after every mutation path
(add_edge / remove_edges_for / load_dict), legacy snapshot compatibility,
dedup semantics, edge-list ordering, and a bulk smoke test.
"""

from __future__ import annotations

import random

import pytest

from sme.modules.memory_graph import (
    KIND_CAUSE,
    KIND_CONVERSATION,
    KIND_NEIGHBOR,
    KIND_REFERENCE,
    KIND_SUMMARY,
    MemoryGraph,
)


# --------------------------------------------------------------------- #
# index invariants
# --------------------------------------------------------------------- #
def test_invariant_after_mixed_mutations():
    g = MemoryGraph()
    assert g._check_invariants() is None  # empty graph

    e1 = g.add_edge("a", "b", KIND_REFERENCE, weight=2.0, note="n1")
    g.add_edge("b", "c", KIND_NEIGHBOR, weight=0.9)
    g.add_edge("c", "d", KIND_CONVERSATION)
    g._check_invariants()

    # dedup re-add mutates the existing edge in place, list must not grow
    again = g.add_edge("a", "b", KIND_REFERENCE, weight=3.5)
    assert again is e1
    assert len(g) == 3
    assert e1.weight == 3.5
    assert e1.note == "n1"  # note is kept when re-add passes empty note
    g._check_invariants()

    removed = g.remove_edges_for("b")
    assert removed == 2
    assert len(g) == 1
    g._check_invariants()

    # re-add after removal lands in the index again
    g.add_edge("b", "e", KIND_SUMMARY)
    g._check_invariants()
    assert g.neighbors_of("b") == {"e"}

    # removing an unknown / edge-less node is a no-op
    assert g.remove_edges_for("zzz") == 0
    g._check_invariants()


def test_invariant_after_load_dict_and_overwrite():
    g = MemoryGraph()
    g.add_edge("a", "b", KIND_REFERENCE)
    g.add_edge("x", "y", KIND_NEIGHBOR)
    # load_dict REPLACES state entirely (engine restore path)
    g.load_dict({"edges": [{"source": "p", "target": "q", "kind": "summary"}]})
    g._check_invariants()
    assert len(g) == 1
    assert g.neighbors_of("p") == {"q"}
    # old adjacency must be gone
    assert g.neighbors_of("a") == set()
    assert g.find("a", "b", KIND_REFERENCE) is None


# --------------------------------------------------------------------- #
# legacy snapshot compatibility (format produced by the pre-index version)
# --------------------------------------------------------------------- #
def test_load_legacy_snapshot():
    legacy = {
        "edges": [
            {"source": "m1", "target": "m2", "kind": "reference",
             "weight": 1.0, "note": ""},
            {"source": "m2", "target": "m3", "kind": "neighbor",
             "weight": 0.87, "note": "auto"},
            # older writers may omit optional fields entirely
            {"source": "m3", "target": "m4"},
            {"source": "m4", "target": "m5", "kind": "conversation"},
        ]
    }
    g = MemoryGraph()
    g.load_dict(legacy)
    g._check_invariants()

    assert len(g) == 4
    # default kind is "reference" (MemoryEdge.from_dict semantics)
    assert g.find("m3", "m4", KIND_REFERENCE) is not None
    assert g.find("m2", "m3", KIND_NEIGHBOR).weight == 0.87
    # snapshot format is byte-identical to the legacy shape
    assert g.to_dict() == {
        "edges": [
            {"source": "m1", "target": "m2", "kind": "reference",
             "weight": 1.0, "note": ""},
            {"source": "m2", "target": "m3", "kind": "neighbor",
             "weight": 0.87, "note": "auto"},
            {"source": "m3", "target": "m4", "kind": "reference",
             "weight": 1.0, "note": ""},
            {"source": "m4", "target": "m5", "kind": "conversation",
             "weight": 1.0, "note": ""},
        ]
    }


def test_roundtrip_preserves_edge_order():
    g = MemoryGraph()
    order = [("a", "b"), ("c", "d"), ("a", "d"), ("b", "c"), ("e", "f")]
    for i, (a, b) in enumerate(order):
        g.add_edge(a, b, KIND_REFERENCE, weight=float(i))
    g.remove_edges_for("c")  # drops (c,d) and (b,c), keeps relative order
    g.add_edge("g", "h", KIND_REFERENCE)

    h = MemoryGraph()
    h.load_dict(g.to_dict())
    assert h.to_dict() == g.to_dict()
    assert [(e.source, e.target) for e in h.edges] == [
        ("a", "b"), ("a", "d"), ("e", "f"), ("g", "h")
    ]
    h._check_invariants()


# --------------------------------------------------------------------- #
# dedup semantics
# --------------------------------------------------------------------- #
def test_dedup_directed_vs_symmetric_kinds():
    g = MemoryGraph()

    # directed kinds: (a,b) and (b,a) are distinct edges
    ab = g.add_edge("a", "b", KIND_REFERENCE)
    ba = g.add_edge("b", "a", KIND_REFERENCE)
    assert ab is not ba
    assert len(g) == 2

    # symmetric kinds: (b,a,neighbor) resolves to the (a,b,neighbor) edge
    nb = g.add_edge("a", "b", KIND_NEIGHBOR, weight=0.5)
    nb_rev = g.add_edge("b", "a", KIND_NEIGHBOR, weight=0.6)
    assert nb_rev is nb
    assert nb.weight == 0.6
    # find matches from either direction for symmetric kinds
    assert g.find("a", "b", KIND_NEIGHBOR) is nb
    assert g.find("b", "a", KIND_NEIGHBOR) is nb
    # ...but not for directed kinds
    assert g.find("b", "a", KIND_REFERENCE) is ba
    assert g.find("a", "b", KIND_REFERENCE) is ab
    assert len(g) == 3
    g._check_invariants()


def test_self_loop_still_rejected():
    g = MemoryGraph()
    with pytest.raises(ValueError):
        g.add_edge("a", "a", KIND_REFERENCE)
    g._check_invariants()
    assert len(g) == 0


# --------------------------------------------------------------------- #
# neighbors / traversal behavior via the index
# --------------------------------------------------------------------- #
def test_neighbors_of_kind_filter_and_both_directions():
    g = MemoryGraph()
    g.add_edge("a", "b", KIND_REFERENCE)      # a -> b
    g.add_edge("c", "a", KIND_CAUSE)  # c -> a
    g.add_edge("a", "d", KIND_NEIGHBOR)
    g.add_edge("a", "e", KIND_CONVERSATION)

    assert g.neighbors_of("a") == {"b", "c", "d", "e"}
    assert g.neighbors_of("b") == {"a"}
    assert g.neighbors_of("c") == {"a"}
    assert g.neighbors_of("zzz") == set()
    assert g.neighbors_of("a", kinds=["neighbor", "conversation"]) == {"d", "e"}
    assert g.neighbors_of("a", kinds=["reference"]) == {"b"}
    g._check_invariants()


def test_traverse_unchanged():
    g = MemoryGraph()
    g.add_edge("a", "b", KIND_REFERENCE)
    g.add_edge("b", "c", KIND_REFERENCE)
    g.add_edge("c", "d", KIND_REFERENCE)
    assert g.traverse("a", max_depth=2) == ["a", "b", "c"]
    assert g.traverse("a", mode="dfs", max_depth=3) == ["a", "b", "c", "d"]


# --------------------------------------------------------------------- #
# bulk smoke: index agrees with brute-force scans over the edge list
# --------------------------------------------------------------------- #
def test_bulk_smoke_20k_edges():
    rng = random.Random(2026)
    g = MemoryGraph()
    kinds = ["reference", "neighbor", "conversation", "summary", "cause"]
    added: list = []
    for _ in range(20_000):
        a, b = (f"n{rng.randrange(500):03d}" for _ in range(2))
        if a == b:
            continue
        added.append((a, b, kinds[rng.randrange(len(kinds))]))

    for a, b, k in added:
        g.add_edge(a, b, k)
    g._check_invariants()

    # spot-check neighbors/find against brute-force scans of `edges`
    for _ in range(50):
        node = f"n{rng.randrange(500):03d}"
        brute = {
            e.target if e.source == node else e.source
            for e in g.edges
            if e.source == node or e.target == node
        }
        assert g.neighbors_of(node) == brute

    for _ in range(50):
        e = g.edges[rng.randrange(len(g.edges))]
        found = g.find(e.source, e.target, e.kind)
        assert found is not None
        if e.kind in (KIND_NEIGHBOR, KIND_CONVERSATION):
            assert g.find(e.target, e.source, e.kind) is e

    # bulk removal agrees with brute force and keeps invariants
    victims = [f"n{i:03d}" for i in range(0, 500, 25)]
    expected_removed = sum(
        1 for e in g.edges if e.source in victims or e.target in victims
    )
    before = len(g)
    got = sum(g.remove_edges_for(v) for v in victims)
    assert got == expected_removed
    assert len(g) == before - got
    g._check_invariants()
    for v in victims:
        assert g.neighbors_of(v) == set()
    # survivors untouched
    for e in g.edges:
        assert e.source not in victims and e.target not in victims
