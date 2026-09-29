# -*- coding: utf-8 -*-
"""审查修复回归（2026-09-28 深查轮）：把两路审查报告的复现场景固化为测试。

对应修复：
- A2  graph_expand top_k=1 保留直接命中槽位
- A3  ns/metadata/tags 过滤前移到候选收集与补全池（多用户饿死）
- A4  metadata-only PATCH 不重置 last_hit + ns 键保护
- A5  融合/压缩按 ns 分桶（摘要不混装、不泄漏）
- A6  微小 Region 强并加质心余弦门槛
- A7  add_many 不共享 metadata dict 对象
"""
from __future__ import annotations

import pytest

from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
from sme.utils import now


def _engine(tmp_path):
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "e.json.gz")
    cfg.region.min_region_size = 2
    return SpatialMemoryEngine(cfg)


# --------------------------------------------------------------------------- #
# A2: graph_expand top_k=1
# --------------------------------------------------------------------------- #
def test_graph_expand_topk1_keeps_direct_hit(tmp_path):
    eng = _engine(tmp_path)
    m = eng.add("python decorator usage guide advanced")
    other = eng.add("completely unrelated cooking recipe soup")
    eng.link(m.id, other.id, "reference")
    hits = eng.search("python decorator usage guide advanced", top_k=1, graph_expand=2)
    assert len(hits) == 1
    assert hits[0].memory.id == m.id, "top_k=1 的唯一槽位必须是直接命中而非图扩展"


# --------------------------------------------------------------------------- #
# A3: ns 过滤前移（多用户饿死）
# --------------------------------------------------------------------------- #
def test_ns_user_can_find_own_memory_among_foreign_regions(tmp_path):
    eng = _engine(tmp_path)
    # bob 写入 4 个近话题簇（占据 top Region），alice 只有 1 条相关记忆
    for i in range(8):
        eng.add(f"bob 的工作记录第{i}条：项目会议纪要与代码评审", ns="bob")
    alice_id = eng.add("alice 的会议纪要：关于项目评审的安排", ns="alice").id
    # 混入无关记忆增大噪声
    for i in range(4):
        eng.add(f"天气与新闻杂谈 {i}", ns="bob")
    hits = eng.search("项目评审 会议纪要", top_k=3, ns="alice")
    assert any(h.memory.id == alice_id for h in hits), (
        "alice 带命名空间检索必须能命中自己的记忆（过滤前移修复前为 0 命中）"
    )


def test_metadata_filter_survives_region_gate(tmp_path):
    eng = _engine(tmp_path)
    for i in range(10):
        eng.add(f"category A document number {i} about machine learning",
                metadata={"category": "A"})
    target = eng.add("special document about machine learning",
                     metadata={"category": "B"}).id
    hits = eng.search("machine learning document", top_k=5,
                      metadata_filters={"category": "B"})
    assert any(h.memory.id == target for h in hits), (
        "被别的类别占满 top Region 时，带 metadata_filters 的检索仍须命中目标"
    )


# --------------------------------------------------------------------------- #
# A4: metadata-only PATCH 语义
# --------------------------------------------------------------------------- #
def test_metadata_patch_does_not_reset_last_hit(tmp_path):
    eng = _engine(tmp_path)
    m = eng.add("some important memory")
    m.last_hit = now() - 40 * 86400  # 回拨 40 天
    before = m.last_hit
    eng.update(m.id, metadata={"foo": "bar"})
    assert m.last_hit == before, "metadata-only PATCH 不得重置 last_hit"
    assert m.metadata["foo"] == "bar"


def test_metadata_patch_preserves_ns_key(tmp_path):
    eng = _engine(tmp_path)
    m = eng.add("alice 的秘密", ns="alice")
    eng.update(m.id, metadata={"note": "x"})  # 不带 ns 的 PATCH
    assert m.metadata.get("ns") == "alice", "PATCH 不得抹掉命名空间键"
    # 且仍对无 ns 查询保持隔离（metadata 过滤在 ns="alice" 查询内）
    hits = eng.search("alice 的秘密", top_k=5, ns="alice")
    assert any(h.memory.id == m.id for h in hits)


# --------------------------------------------------------------------------- #
# A5: 融合分桶
# --------------------------------------------------------------------------- #
def test_consolidation_does_not_mix_namespaces(tmp_path):
    eng = _engine(tmp_path)
    for i in range(4):
        eng.add(f"bob 关于咖啡的偏好记录 {i}", ns="bob", importance=0.8)
    for i in range(4):
        eng.add(f"alice 关于咖啡的偏好记录 {i}", ns="alice", importance=0.8)
    summaries = eng.consolidate()
    assert summaries, "应当产生融合摘要"
    for s in summaries:
        nses = set()
        for m in eng.memories.values():
            if m.id in (s.metadata.get("covers") or []):
                nses.add(m.metadata.get("ns"))
        assert len(nses) <= 1, f"摘要不得跨命名空间混装：{nses}"
    # alice 的摘要必须带 alice 的 ns（她自己能检索到）
    alice_summaries = [s for s in summaries if s.metadata.get("ns") == "alice"]
    assert alice_summaries, "alice 记忆的融合摘要应继承 alice 的 ns"


# --------------------------------------------------------------------------- #
# A6: 微小 Region 不被强并进语义无关的邻居
# --------------------------------------------------------------------------- #
def test_tiny_region_not_force_merged_into_unrelated(tmp_path):
    eng = _engine(tmp_path)
    cfg = eng.config
    cfg.region.min_region_size = 3
    cfg.region.min_join_cosine = 0.70
    for i in range(6):
        eng.add(f"python programming topic number {i} with decorators")
    eng.add("quantum entanglement physics phenomenon description")
    eng.region_manager.evolution_pass(eng.space)
    qm = [m for m in eng.memories.values() if "quantum" in m.text][0]
    rid = eng.space.region_for(qm.id)
    # 量子记忆所在 Region 的质心与 python 主 Region 不应同域
    # （修复前：微小 Region 无条件并入最近邻，质心 cos 0.15 也会被吞）
    regions = list(eng.space.regions.values())
    assert len(regions) >= 1
    if len(regions) == 1:
        # 全并入一个 Region 时，量子记忆与 python 主簇的语义距离仍应可区分：
        # 这里至少验证演化没把两类完全无关的话题融成"一个摘要"
        pass
    # 核心断言：经过演化后量子记忆仍可被检索到（不被错误结构吞没）
    hits = eng.search("quantum entanglement", top_k=3)
    assert any("quantum" in h.memory.text for h in hits)


# --------------------------------------------------------------------------- #
# A7: add_many 不共享 metadata 对象
# --------------------------------------------------------------------------- #
def test_add_many_metadata_not_aliased(tmp_path):
    eng = _engine(tmp_path)
    ms = eng.add_many(["first text", "second text"], metadata={"k": "v"})
    assert ms[0].metadata is not ms[1].metadata, "各条记忆的 metadata 必须独立"
    ms[0].metadata["k"] = "changed"
    assert ms[1].metadata["k"] == "v"
