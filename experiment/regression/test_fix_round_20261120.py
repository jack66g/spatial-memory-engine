# -*- coding: utf-8 -*-
"""2026-11-20 大型测试战役修复轮回归(战报:TestLab reports/测试战报_2026-11-20.md)。

覆盖本轮实测确认的缺陷:
P2-2  importance NaN/inf 穿透 clamp(SDK add/add_many/update)
P2-4  import_memories 负值/非有限数值毒数据 → 检索除零
P2-3  热重建闸门漏 mrl_dim → 新旧维度混存崩库
P2-并发 cleanup 竞态删除并发写方刚生成的 sidecar(孤儿快照)
P3-8  SDK update() 接受纯空白文本
P3-9  add_many importance 不 clamp
P3-10 注册表默认值与 dataclass 不一致(llm.temperature / llm.model)
P3-15 SMEConfig.from_dict 零校验(evolve_interval=0 → 写入 ZeroDivision)
新增   预设 06 静态检索·UNITED(权重寻优成果落地)
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sme.config import SMEConfig
from sme.config_items import ITEM_BY_PATH, PRESET_BY_KEY
from sme.engine import SpatialMemoryEngine


# ------------------------------------------------------------------ #
# P2-2 / P3-9: importance NaN/inf 与 add_many clamp
# ------------------------------------------------------------------ #
def test_add_importance_nan_rejected(fresh_engine):
    with pytest.raises(ValueError, match="importance"):
        fresh_engine.add("用户喜欢苹果", importance=float("nan"))


def test_add_importance_inf_rejected(fresh_engine):
    with pytest.raises(ValueError, match="importance"):
        fresh_engine.add("用户喜欢苹果", importance=float("inf"))


def test_add_many_importance_clamped_and_finite_checked(fresh_engine):
    ms = fresh_engine.add_many(["批量一", "批量二"], importance=7.5)
    imps = [fresh_engine.memories[m.id].importance for m in ms]
    assert all(0.0 <= i <= 1.0 for i in imps), f"add_many importance 未 clamp: {imps}"
    with pytest.raises(ValueError, match="importance"):
        fresh_engine.add_many(["批量三"], importance=float("nan"))


def test_update_importance_nan_rejected(fresh_engine):
    m = fresh_engine.add("用户喜欢橙子")
    with pytest.raises(ValueError, match="importance"):
        fresh_engine.update(m.id, importance=float("nan"))
    with pytest.raises(ValueError, match="weight"):
        fresh_engine.update(m.id, weight=float("-inf"))


# ------------------------------------------------------------------ #
# P2-4: import 毒数据校验
# ------------------------------------------------------------------ #
def test_import_negative_hit_count_rejected(fresh_engine):
    with pytest.raises(ValueError):
        fresh_engine.import_memories([
            {"id": "p1", "text": "毒记忆", "hit_count": -1, "weight": 1.0}])


def test_import_negative_weight_rejected(fresh_engine):
    with pytest.raises(ValueError):
        fresh_engine.import_memories([
            {"id": "p2", "text": "毒记忆", "hit_count": 2, "weight": -1.0}])


def test_import_out_of_range_importance_rejected(fresh_engine):
    with pytest.raises(ValueError):
        fresh_engine.import_memories([
            {"id": "p3", "text": "毒记忆", "importance": 7.5}])


def test_import_nan_importance_rejected(fresh_engine):
    with pytest.raises(ValueError):
        fresh_engine.import_memories([
            {"id": "p4", "text": "毒记忆", "importance": float("nan")}])


def test_import_valid_items_still_work(fresh_engine):
    n = fresh_engine.import_memories([
        {"id": "ok1", "text": "正常记忆", "hit_count": 3, "weight": 1.4, "importance": 0.8}])
    assert n == 1
    assert fresh_engine.memories["ok1"].hit_count == 3
    hits = fresh_engine.search("正常", top_k=5)   # 不再除零
    assert len(hits) >= 1


# ------------------------------------------------------------------ #
# P3-8: update 空白文本
# ------------------------------------------------------------------ #
def test_update_blank_text_rejected(fresh_engine):
    m = fresh_engine.add("原始记忆内容")
    with pytest.raises(ValueError, match="empty"):
        fresh_engine.update(m.id, text="   ")
    assert fresh_engine.memories[m.id].text == "原始记忆内容"


# ------------------------------------------------------------------ #
# P3-15 / from_dict 校验
# ------------------------------------------------------------------ #
def test_from_dict_rejects_zero_interval():
    with pytest.raises(ValueError):
        SMEConfig.from_dict({"region": {"evolve_interval": 0}})


def test_from_dict_rejects_out_of_range_temperature():
    with pytest.raises(ValueError):
        SMEConfig.from_dict({"llm": {"temperature": 99}})


def test_from_dict_rejects_wrong_type():
    with pytest.raises(ValueError):
        SMEConfig.from_dict({"region": {"min_region_size": "not_an_int"}})


def test_from_dict_accepts_valid_and_none():
    cfg = SMEConfig.from_dict({"region": {"evolve_interval": 10},
                               "llm": {"reasoning_effort": None}})
    assert cfg.region.evolve_interval == 10
    assert cfg.llm.reasoning_effort is None


# ------------------------------------------------------------------ #
# P3-10: 注册表默认与 dataclass 对齐
# ------------------------------------------------------------------ #
def test_registry_defaults_match_dataclass():
    cfg = SMEConfig()
    assert ITEM_BY_PATH["llm.temperature"].default == cfg.llm.temperature
    assert ITEM_BY_PATH["llm.model"].default == cfg.llm.model


# ------------------------------------------------------------------ #
# P2-3: 热重建闸门覆盖 mrl_dim
# ------------------------------------------------------------------ #
def test_hot_rebuild_gate_blocks_mrl_dim(tmp_path):
    from fastapi.testclient import TestClient
    from sme.api.server import create_app

    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "state.json")
    engine = SpatialMemoryEngine(cfg)
    client = TestClient(create_app(engine, config_file=str(tmp_path / "c.json")))
    for i in range(5):
        assert client.post("/memories", json={"text": f"记忆{i}关于主题{i}"}).status_code == 200
    r = client.put("/config", json={"values": {"embedding.mrl_dim": 32}})
    # PUT /config 的契约:拒绝走 HTTP 200 + ok:false + errors(库非空时闸门拦截)
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is False, f"闸门未拦截 mrl_dim: {body}"
    assert "mrl" in body.get("errors", {}).get("_global", "")
    # 库未毒化:检索仍正常
    rs = client.post("/memories/search", json={"text": "主题", "top_k": 3})
    assert rs.status_code == 200


# ------------------------------------------------------------------ #
# REST 面数值校验(NaN 在入口被拒)
# ------------------------------------------------------------------ #
def test_rest_add_importance_nan_rejected(tmp_path):
    from fastapi.testclient import TestClient
    from sme.api.server import create_app

    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "state.json")
    engine = SpatialMemoryEngine(cfg)
    client = TestClient(create_app(engine, config_file=str(tmp_path / "c.json")))
    body = '{"text": "NaN importance", "importance": NaN}'.encode("ascii")
    r = client.post("/memories", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422, f"NaN importance 未被 REST 拒绝: {r.status_code}"


# ------------------------------------------------------------------ #
# P2-并发: cleanup 不得删除"新鲜"的未引用 sidecar(双进程竞态)
# ------------------------------------------------------------------ #
def test_cleanup_grace_window_protects_fresh_sidecars(fresh_engine, tmp_path):
    from sme.storage.snapshot import embedding_path

    for i in range(3):
        fresh_engine.add(f"记忆{i}内容{i}")
    state = str(tmp_path / "race.json")
    fresh_engine.save(state)

    emb_base = embedding_path(state if not state.endswith(".gz") else state[:-3])
    # 模拟并发进程 B 的在途 sidecar:已写盘、manifest 尚未写 → mtime 晚于被引用代
    import uuid as _uuid
    peer = emb_base[:-4] + "." + _uuid.uuid4().hex + ".npz"
    Path(peer).write_bytes(b"fresh")
    future = time.time() + 5
    os.utime(peer, (future, future))
    # 模拟陈旧孤儿(mtime 早于被引用 sidecar)
    stale = emb_base[:-4] + "." + _uuid.uuid4().hex + ".npz"
    Path(stale).write_bytes(b"stale")
    old = time.time() - 7200
    os.utime(stale, (old, old))

    fresh_engine.save(state)   # 触发 cleanup
    assert Path(peer).exists(), "cleanup 删除了并发写方的在途 sidecar(孤儿快照根因)"
    assert not Path(stale).exists(), "陈旧孤儿未被清理"


import time  # noqa: E402  (供上面 utime 用,放尾部避免顶部未用告警)


# ------------------------------------------------------------------ #
# 新增预设:UNITED 静态检索
# ------------------------------------------------------------------ #
def test_united_preset_exists_and_applies(tmp_path):
    preset = PRESET_BY_KEY["united_static"]
    cfg = SMEConfig.from_dict(preset_flatten(preset))
    assert cfg.policy.decay_enabled is False
    assert cfg.decay.max_weight_decay == 0.0
    assert cfg.retrieval.keyword_candidates == 20
    assert cfg.retrieval.top_regions == 1
    assert cfg.retrieval.vector_weight == 0.0
    assert cfg.ranking.semantic == 1.0
    assert cfg.ranking.freshness == 0.0
    # 引擎能跑通基本链路
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "u.json")
    e = SpatialMemoryEngine(cfg)
    e.add("用户喜欢苹果")
    e.add("用户周末打篮球")
    hits = e.search("用户喜欢什么水果", top_k=5)
    assert hits and hits[0].memory.text == "用户喜欢苹果"


def preset_flatten(preset) -> dict:
    """PRESETS values 是注册表扁平路径 → 嵌套 dict(剥掉非引擎的 memory.* 会话层键)"""
    tree: dict = {}
    for path, value in preset["values"].items():
        section, field = path.split(".", 1)
        if section == "memory":
            continue
        tree.setdefault(section, {})[field] = value
    return tree
