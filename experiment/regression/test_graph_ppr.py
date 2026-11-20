# -*- coding: utf-8 -*-
"""PPR graph expansion (HippoRAG 2 route) vs the legacy BFS control path.

Covered:
- fan-in hub ("many relevant nodes point at one key node"): BFS's 0.5**depth
  penalty buries the hub, PPR mass concentration surfaces it
- star graph: BFS cannot separate equal-text depth-1 nodes, PPR separates
  them by in-degree
- two-cluster bridge: the bridging hub floats up, the far cluster does not
  spill into the results (teleport anchors the walk on the seeds)
- empty / disconnected graphs: byte-identical to the BFS path
- tiny graphs (< _PPR_MIN_EDGES edges): deterministic BFS fallback
- archived nodes neither surface nor relay the walk
- 2000-node graph: single expansion stays under the 50 ms budget (node cap hit)
"""
from __future__ import annotations

import time

from sme.retrieval.retriever import SearchQuery, TwoStageRetriever, _PPR_MIN_EDGES
from sme.utils import now

# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _ab_expand(eng, text: str, top_k: int, depth: int):
    """Run BOTH expansion paths on identical inputs.

    The direct top-k (graph_expand=0) seeds both paths, so any output
    difference is attributable to the expansion algorithm alone.
    """
    retriever = eng.retriever
    base = retriever.search(eng, SearchQuery(text=text, top_k=top_k, graph_expand=0))
    query = SearchQuery(text=text, top_k=top_k, graph_expand=depth)
    query_vec = eng.embeddings.embed_one(text)
    ref = now()
    args = (eng, query, base, base, query_vec, {}, ref)
    return (
        retriever._graph_expand_ppr(*args),
        retriever._graph_expand_bfs(*args),
        args,
    )


def _ids(hits) -> list[str]:
    return [h.memory.id for h in hits]


def _scores(hits) -> dict[str, float]:
    return {h.memory.id: h.score for h in hits}


def _spy(monkeypatch, attr: str) -> list:
    """Record calls to a TwoStageRetriever method (keeps original behavior)."""
    calls: list = []
    original = getattr(TwoStageRetriever, attr)

    def wrapper(self, *a, **k):
        calls.append(a)
        return original(self, *a, **k)

    monkeypatch.setattr(TwoStageRetriever, attr, wrapper)
    return calls


# --------------------------------------------------------------------- #
# PPR math (pure static method, no engine)
# --------------------------------------------------------------------- #
def test_ppr_rank_chain_respects_distance():
    """单链：质量沿跳数单调衰减，质量守恒。"""
    nodes = ["s", "c1", "c2", "c3"]
    adj = {
        "s": {"c1": 1.0},
        "c1": {"s": 1.0, "c2": 1.0},
        "c2": {"c1": 1.0, "c3": 1.0},
        "c3": {"c2": 1.0},
    }
    rank = TwoStageRetriever._ppr_rank(nodes, adj, {"s": 1.0}, 1.0)
    # the first hop may exceed the seed itself (walk mass + return flow),
    # but mass strictly decays with hop count past it
    assert rank["c1"] > rank["c2"] > rank["c3"] > 0.0
    assert rank["s"] > rank["c2"]
    assert abs(sum(rank.values()) - 1.0) < 1e-6  # mass conservation


def test_ppr_rank_weighted_edges_and_dangling():
    """重边吸走更多质量；悬空节点拿 0 且不炸。"""
    nodes = ["s", "hi", "lo", "dangle"]
    adj = {
        "s": {"hi": 3.0, "lo": 1.0},
        "hi": {"s": 3.0},
        "lo": {"s": 1.0},
        "dangle": {},  # no out-edges: mass must return to the seeds
    }
    rank = TwoStageRetriever._ppr_rank(nodes, adj, {"s": 1.0}, 1.0)
    assert rank["hi"] > rank["lo"]
    assert rank["dangle"] == 0.0
    assert abs(sum(rank.values()) - 1.0) < 1e-6


def test_ppr_rank_personalization_follows_seed_weights():
    """两个 seed 的 personalize 权重决定各自邻域分到多少质量。"""
    nodes = ["a", "b", "na", "nb"]
    adj = {
        "a": {"na": 1.0}, "na": {"a": 1.0},
        "b": {"nb": 1.0}, "nb": {"b": 1.0},
    }
    heavy = TwoStageRetriever._ppr_rank(nodes, adj, {"a": 3.0, "b": 1.0}, 4.0)
    assert heavy["na"] > heavy["nb"]
    assert heavy["a"] > heavy["b"]


# --------------------------------------------------------------------- #
# 链形 fan-in：多相关节点指向的关键节点（核心场景）
# --------------------------------------------------------------------- #
def test_ppr_chain_fanin_hub_beats_depth_penalty(fresh_engine):
    """6 条链在 2 跳外汇聚于一个关键节点：BFS 深度惩罚看不到，PPR 浮上来。

    seeds(深度0) -> noise(深度1) -> hub(深度2)。hub 的 hybrid 略强于浅层
    noise，但 BFS 给深度1 x0.5、深度2 x0.25 -> 槽位被浅层噪声吃掉；
    PPR 把 6 条链的质量聚集到 hub 上 -> hub 占据扩展槽。
    """
    eng = fresh_engine
    seeds = [eng.add(f"alpha beta gamma delta epsilon memo {i}") for i in range(6)]
    noise = [eng.add(f"alpha beta gamma misc note {i}") for i in range(6)]
    hub = eng.add("alpha beta gamma delta key conclusion")
    for i, s in enumerate(seeds):
        eng.link(s.id, noise[i].id, "reference", 1.0)
    for n in noise:
        eng.link(n.id, hub.id, "reference", 1.0)

    ppr, bfs, _ = _ab_expand(eng, "alpha beta gamma delta epsilon", top_k=6, depth=2)
    assert hub.id in _ids(ppr), "PPR 扩展槽必须浮出 2 跳外的关键节点"
    assert hub.id not in _ids(bfs), "BFS 的深度惩罚必然把 2 跳节点压在浅层噪声下"
    assert set(_ids(bfs)) & {n.id for n in noise}, "BFS 槽位被浅层噪声占据"
    # 量纲兼容：PPR 扩展命中分数与 BFS 深度1 的衰减同一量级（锚定 x0.5）
    hub_score = next(h.score for h in ppr if h.memory.id == hub.id)
    bfs_expansion = next(
        h.score for h in bfs if h.memory.id in {n.id for n in noise}
    )
    assert 0.0 < hub_score < bfs_expansion * 3.0  # same band, not blown up


def test_ppr_expansion_repeatable(fresh_engine):
    """同一输入连续两次扩展结果完全一致（无隐藏状态）。"""
    eng = fresh_engine
    a = eng.add("alpha beta gamma delta epsilon memo 0")
    b = eng.add("alpha beta gamma misc note 0")
    c = eng.add("alpha beta gamma delta key conclusion")
    eng.link(a.id, b.id, "reference", 1.0)
    eng.link(b.id, c.id, "reference", 1.0)
    eng.link(a.id, c.id, "reference", 1.0)
    eng.link(a.id, b.id, "neighbor", 0.9)  # parallel edge of another kind
    ppr1, _, args = _ab_expand(eng, "alpha beta gamma delta epsilon", 5, 2)
    ppr2 = eng.retriever._graph_expand_ppr(*args)
    assert [(i, round(h.score, 9)) for i, h in zip(_ids(ppr1), ppr1)] == [
        (i, round(h.score, 9)) for i, h in zip(_ids(ppr2), ppr2)
    ]


# --------------------------------------------------------------------- #
# 星形：同深度同文本节点，BFS 无法区分，PPR 按入度区分
# --------------------------------------------------------------------- #
def test_ppr_star_concentrates_on_hub(fresh_engine):
    """星形：center 被 5 个叶子指向，outlier 只被 1 个叶子指向。

    两者深度同为 1、文本完全相同：BFS 的 0.5**1 衰减 + 相同 hybrid ->
    分数打平（没有任何区分信号）；PPR 的入流集中度把它们拉开。
    """
    eng = fresh_engine
    leaves = [eng.add(f"alpha beta gamma delta epsilon leaf {i}") for i in range(10)]
    center = eng.add("alpha beta gamma hub central fact")
    outlier = eng.add("alpha beta gamma hub central fact")  # identical text
    for leaf in leaves[:5]:
        eng.link(leaf.id, center.id, "reference", 1.0)
    eng.link(leaves[0].id, outlier.id, "reference", 1.0)

    ppr, bfs, _ = _ab_expand(eng, "alpha beta gamma delta epsilon", top_k=10, depth=1)
    # both paths surface both depth-1 nodes (2 reserved slots at top_k=10)
    for hits in (ppr, bfs):
        assert center.id in _ids(hits) and outlier.id in _ids(hits)
    # BFS: identical depth + identical text -> identical scores (no signal)
    bfs_scores = _scores(bfs)
    assert abs(bfs_scores[center.id] - bfs_scores[outlier.id]) < 1e-6
    # PPR: in-degree concentration strictly separates them
    ppr_scores = _scores(ppr)
    assert ppr_scores[center.id] - ppr_scores[outlier.id] > 0.1
    assert _ids(ppr).index(center.id) < _ids(ppr).index(outlier.id)


# --------------------------------------------------------------------- #
# 双簇 + 桥接：桥接枢纽浮出，远簇不外溢
# --------------------------------------------------------------------- #
def test_ppr_two_cluster_bridge(fresh_engine):
    """簇A(seeds) -> feeders(深度1) -> bridge(深度2) -> 远簇(深度3)。

    PPR：bridge 聚集 4 条链的质量占据扩展槽；远簇仅单路径（弱权重边）
    可达，质量被 teleport 锚定在 seeds 附近，不外溢进 top-k。
    BFS：深度惩罚让 x0.25 的 bridge 输给 x0.5 的浅层 feeder。
    """
    eng = fresh_engine
    seeds = [eng.add(f"alpha beta gamma delta epsilon record {i}") for i in range(4)]
    feeders = [eng.add(f"alpha beta gamma feed note {i}") for i in range(4)]
    bridge = eng.add("alpha beta gamma delta bridge summary")
    far = [eng.add(f"omega sigma unrelated topic {i}") for i in range(4)]
    for i, s in enumerate(seeds):
        eng.link(s.id, feeders[i].id, "reference", 1.0)
    for f in feeders:
        eng.link(f.id, bridge.id, "reference", 1.0)
    for f in far:
        eng.link(bridge.id, f.id, "reference", 0.2)  # weak far bridge

    ppr, bfs, _ = _ab_expand(eng, "alpha beta gamma delta epsilon", top_k=4, depth=3)
    assert bridge.id in _ids(ppr), "PPR 必须浮出跨簇桥接枢纽"
    assert bridge.id not in _ids(bfs), "BFS 深度惩罚把 2 跳桥接压在浅层下"
    assert set(_ids(bfs)) & {f.id for f in feeders}, "BFS 槽位被浅层 feeder 占据"
    far_ids = {f.id for f in far}
    assert not (far_ids & set(_ids(ppr))), "远簇不得因桥接边外溢进结果"


# --------------------------------------------------------------------- #
# 退化护栏：空图 / 断连图 / 小图 / 开关
# --------------------------------------------------------------------- #
def test_empty_graph_matches_bfs_exactly(fresh_engine):
    """无图边：与旧路径完全一致（直接返回 top）。"""
    eng = fresh_engine
    for i in range(5):
        eng.add(f"alpha beta gamma delta epsilon memo {i}")
    ppr, bfs, args = _ab_expand(eng, "alpha beta gamma delta epsilon", 5, 3)
    assert _ids(ppr) == _ids(bfs) == _ids(eng.retriever._graph_expand(*args))
    assert [h.score for h in ppr] == [h.score for h in bfs]


def test_disconnected_graph_matches_bfs_exactly(fresh_engine):
    """有边但与 seeds 不连通：与旧路径完全一致。"""
    eng = fresh_engine
    for i in range(3):
        eng.add(f"alpha beta gamma delta epsilon memo {i}")
    a = eng.add("island node about cooking pasta recipes")
    b = eng.add("island node about gardening herbs")
    c = eng.add("island node about weather forecasts")
    d = eng.add("island node about travel plans")
    # a connected 4-edge island the seeds never touch
    eng.link(a.id, b.id, "reference", 1.0)
    eng.link(b.id, c.id, "reference", 1.0)
    eng.link(c.id, d.id, "reference", 1.0)
    eng.link(d.id, a.id, "reference", 1.0)
    ppr, bfs, args = _ab_expand(eng, "alpha beta gamma delta epsilon", 5, 2)
    assert _ids(ppr) == _ids(bfs)
    assert _ids(eng.retriever._graph_expand(*args)) == _ids(bfs)


def test_small_graph_falls_back_to_bfs(fresh_engine, monkeypatch):
    """< _PPR_MIN_EDGES 条边：PPR 退化为 BFS 对照路径，结果逐字节一致。"""
    eng = fresh_engine
    s0 = eng.add("alpha beta gamma delta epsilon memo 0")
    s1 = eng.add("alpha beta gamma delta epsilon memo 1")
    a = eng.add("alpha beta gamma note a")
    b = eng.add("alpha beta gamma note b")
    c = eng.add("alpha beta gamma delta key c")
    eng.link(s0.id, a.id, "reference", 1.0)
    eng.link(s1.id, b.id, "reference", 1.0)
    eng.link(a.id, c.id, "reference", 1.0)
    eng.link(b.id, c.id, "reference", 1.0)
    assert len(eng.graph.edges) == _PPR_MIN_EDGES - 1

    calls = _spy(monkeypatch, "_graph_expand_bfs")
    retriever = eng.retriever
    base = retriever.search(eng, SearchQuery(text="alpha beta gamma delta epsilon",
                                             top_k=4, graph_expand=0))
    query = SearchQuery(text="alpha beta gamma delta epsilon", top_k=4, graph_expand=2)
    query_vec = eng.embeddings.embed_one(query.text)
    args = (eng, query, base, base, query_vec, {}, now())
    via_dispatcher = retriever._graph_expand(*args)
    assert len(calls) == 1, "小图必须恰好走一次 BFS 退化路径"
    direct_bfs = retriever._graph_expand_bfs(*args)
    assert [(i, h.score) for i, h in zip(_ids(via_dispatcher), via_dispatcher)] == [
        (i, h.score) for i, h in zip(_ids(direct_bfs), direct_bfs)
    ]


def test_large_enough_graph_uses_ppr(fresh_engine, monkeypatch):
    """边数达到阈值后走 PPR 主路径（不再经过 BFS）。"""
    eng = fresh_engine
    s0 = eng.add("alpha beta gamma delta epsilon memo 0")
    a = eng.add("alpha beta gamma note a")
    b = eng.add("alpha beta gamma note b")
    c = eng.add("alpha beta gamma delta key c")
    d = eng.add("alpha beta gamma note d")
    eng.link(s0.id, a.id, "reference", 1.0)
    eng.link(s0.id, b.id, "reference", 1.0)
    eng.link(a.id, c.id, "reference", 1.0)
    eng.link(b.id, c.id, "reference", 1.0)
    eng.link(a.id, d.id, "reference", 1.0)
    assert len(eng.graph.edges) >= _PPR_MIN_EDGES

    bfs_calls = _spy(monkeypatch, "_graph_expand_bfs")
    ppr_calls = _spy(monkeypatch, "_graph_expand_ppr")
    eng.search("alpha beta gamma delta epsilon", top_k=4, graph_expand=2)
    assert len(ppr_calls) == 1
    assert len(bfs_calls) == 0


def test_graph_expand_disabled_returns_top(fresh_engine):
    """graph_expand=0：两条路径都直接返回 top（门控行为不变）。"""
    eng = fresh_engine
    m = eng.add("alpha beta gamma delta epsilon memo 0")
    n = eng.add("alpha beta gamma note x")
    eng.link(m.id, n.id, "reference", 1.0)
    retriever = eng.retriever
    base = retriever.search(eng, SearchQuery(text="alpha beta gamma delta epsilon",
                                             top_k=3, graph_expand=0))
    query_vec = eng.embeddings.embed_one("alpha beta gamma delta epsilon")
    args = (eng, SearchQuery(text="alpha beta gamma delta epsilon", top_k=3,
                             graph_expand=0), base, base, query_vec, {}, now())
    assert retriever._graph_expand(*args) is base
    assert retriever._graph_expand_bfs(*args) is base
    assert retriever._graph_expand_ppr(*args) is base


# --------------------------------------------------------------------- #
# 归档护栏：归档节点不浮出也不中继
# --------------------------------------------------------------------- #
def test_ppr_archived_nodes_do_not_relay(fresh_engine):
    """链中间节点归档后：hub 不可达（与 BFS 行为一致）。"""
    eng = fresh_engine
    seeds = [eng.add(f"alpha beta gamma delta epsilon memo {i}") for i in range(4)]
    mids = [eng.add(f"alpha beta gamma mid {i}") for i in range(4)]
    hub = eng.add("alpha beta gamma delta key conclusion")
    for i, s in enumerate(seeds):
        eng.link(s.id, mids[i].id, "reference", 1.0)
    for m in mids:
        eng.link(m.id, hub.id, "reference", 1.0)

    # sanity: before archiving the PPR slot does surface the multi-hop hub
    ppr, _, _ = _ab_expand(eng, "alpha beta gamma delta epsilon", 4, 3)
    assert hub.id in _ids(ppr)

    for m in mids:
        eng.archive(m.id)
    ppr, bfs, _ = _ab_expand(eng, "alpha beta gamma delta epsilon", 4, 3)
    for hits, name in ((ppr, "ppr"), (bfs, "bfs")):
        ids = _ids(hits)
        assert hub.id not in ids, f"归档中继必须切断 {name} 的多跳路径"
        assert not ({m.id for m in mids} & set(ids)), f"{name} 不得浮出归档节点"


# --------------------------------------------------------------------- #
# 性能冒烟：2000 节点大图
# --------------------------------------------------------------------- #
def _timed_ms(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def test_ppr_large_graph_perf(fresh_engine):
    """2000 节点、~50000 条边：单次 PPR 扩展 < 50ms（500 节点截断被触发）。"""
    eng = fresh_engine
    n = 2000
    ids = []
    for i in range(n):
        text = (f"alpha beta target special doc {i}" if i < 3
                else f"generic filler node topic {i} notes")
        ids.append(eng.add(text).id)
    for i in range(n):
        for j in range(1, 26):  # ring + chords -> degree 50
            eng.link(ids[i], ids[(i + j) % n], "neighbor", 0.6)
    assert len(eng.graph.edges) == 25 * n

    retriever = eng.retriever
    query = SearchQuery(text="alpha beta target special", top_k=10, graph_expand=3)
    query_vec = eng.embeddings.embed_one(query.text)
    base = retriever.search(
        eng, SearchQuery(text=query.text, top_k=10, graph_expand=0)
    )
    args = (eng, query, base, base, query_vec, {}, now())

    # subgraph collection must hit (and respect) the node cap
    seeds = sorted({h.memory.id for h in base})
    sub = TwoStageRetriever._ppr_subgraph(eng, query, seeds)
    assert len(sub) == 500

    retriever._graph_expand_ppr(*args)  # warm-up (numpy / code paths)
    best = min(
        _timed_ms(lambda: retriever._graph_expand_ppr(*args)) for _ in range(3)
    )
    assert best < 50.0, f"single PPR expansion took {best:.1f}ms on a 2000-node graph"
    out = retriever._graph_expand_ppr(*args)
    assert 0 < len(out) <= 10
    assert all(h.score >= 0.0 for h in out)
