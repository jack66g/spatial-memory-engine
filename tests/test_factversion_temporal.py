"""Module 05 bi-temporal 时间线专项测试：双时间轴 / 版本链 / 旧数据兼容 / REST。

改口链用「喜欢咖啡 -> 不喜欢了 -> 又喜欢了」：hashing 嵌入下改写句与旧句的
余弦必须 >= correction_threshold（0.60），这对措辞经实测稳定通过（0.66/0.79），
且语义上正是"同一事实三次改版本"。
"""

from __future__ import annotations

import pytest

from sme.modules.factversion import (
    INVALID_AT_KEY,
    STALE_TAG,
    SUPERSEDE_TAG,
    VALID_AT_KEY,
    invalid_at_of,
    valid_at_of,
)
from sme.utils import now


def u(*cps):
    """Chinese string from code points (console-encoding safe)."""
    return "".join(chr(c) for c in cps)


COFFEE = u(0x7528, 0x6237, 0x559C, 0x6B22, 0x559D, 0x5496, 0x5561)          # 用户喜欢喝咖啡
NO_COFFEE = u(0x5176, 0x5B9E, 0x7528, 0x6237, 0x4E0D, 0x559C, 0x6B22,        # 其实用户不喜欢喝咖啡了
              0x559D, 0x5496, 0x5561, 0x4E86)
AGAIN_COFFEE = u(0x5176, 0x5B9E, 0x7528, 0x6237, 0x53C8, 0x559C, 0x6B22,     # 其实用户又喜欢喝咖啡了
                 0x559D, 0x5496, 0x5561, 0x4E86)
WUHOU = u(0x7528, 0x6237, 0x4F4F, 0x5728, 0x6B66, 0x4FAF, 0x533A)            # 用户住在武侯区
CHENGHUA = u(0x5176, 0x5B9E, 0x7528, 0x6237, 0x4F4F, 0x5728,                  # 其实用户住在成华区了
             0x6210, 0x534E, 0x533A, 0x4E86)
UNRELATED = u(0x91CF, 0x5B50, 0x7EA0, 0x7E20, 0x4E0E, 0x9ED1, 0x6D1E, 0x84B8, 0x53D1)  # 量子纠缠与黑洞蒸发

KAFEI = u(0x5496, 0x5561)                # 咖啡
BU_XIHUAN = u(0x4E0D, 0x559C, 0x6B22)    # 不喜欢
YOU_XIHUAN = u(0x53C8, 0x559C, 0x6B22)   # 又喜欢


def _temporal_engine(fresh_engine):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.factversion.enabled = True
    return e


# ------------------------- version chain building --------------------------- #
def test_timeline_three_corrections_builds_chain(fresh_engine, new_engine):
    e = _temporal_engine(fresh_engine)
    e.add(COFFEE)
    e.add(NO_COFFEE)     # 改口 1：喜欢 -> 不喜欢
    e.add(AGAIN_COFFEE)  # 改口 2：不喜欢 -> 又喜欢

    versions = e.factversion.timeline(AGAIN_COFFEE, e)
    assert len(versions) == 3
    v1, v2, v3 = versions
    # 旧 -> 新的文本顺序（rules 提取会剥离句尾语气词，只断言关键词）
    assert KAFEI in v1["text"] and BU_XIHUAN not in v1["text"]
    assert BU_XIHUAN in v2["text"]
    assert YOU_XIHUAN in v3["text"]
    # 链指针
    assert v1["superseded_by"] == v2["id"] and v1["supersedes"] is None
    assert v2["superseded_by"] == v3["id"] and v2["supersedes"] == v1["id"]
    assert v3["superseded_by"] is None and v3["supersedes"] == v2["id"]
    # 双时间戳：首版 valid_at 未盖章（入链前没有 supersede 判定时刻）
    assert v1["valid_at"] is None
    stamps = [v1["invalid_at"], v2["valid_at"], v2["invalid_at"], v3["valid_at"]]
    assert all(s is not None for s in stamps)
    assert stamps == sorted(stamps)                 # 单调不减
    # invalid_at 链闭合：旧条失效时刻 == 新条生效时刻（同一 supersede 瞬间）
    assert v1["invalid_at"] == v2["valid_at"]
    assert v2["invalid_at"] == v3["valid_at"]
    # 最新版本现行有效
    assert v3["invalid_at"] is None and v3["is_current"]
    assert not v1["is_current"] and not v2["is_current"]


def test_timeline_lookup_by_object_and_memory_id(fresh_engine, new_engine):
    e = _temporal_engine(fresh_engine)
    e.add(COFFEE)
    e.add(NO_COFFEE)

    by_text = e.factversion.timeline(NO_COFFEE, e)
    newest_id = by_text[-1]["id"]
    by_id = e.factversion.timeline(newest_id, e)
    by_obj = e.factversion.timeline(e.memories[newest_id], e)
    by_old_text = e.factversion.timeline(COFFEE, e)   # 用旧说法也能查到同一条链
    assert by_text == by_id == by_obj == by_old_text
    assert len(by_text) == 2
    assert e.factversion.timeline(UNRELATED, e) == []
    assert e.factversion.timeline(None, e) == []


def test_timeline_returns_single_version_for_unsuperseded_fact(
    fresh_engine, new_engine
):
    e = _temporal_engine(fresh_engine)
    e.add(COFFEE)
    versions = e.factversion.timeline(COFFEE, e)
    assert len(versions) == 1
    assert versions[0]["is_current"] and versions[0]["invalid_at"] is None


# ------------------------- legacy snapshot compat --------------------------- #
def test_legacy_records_compat_read(fresh_engine, new_engine):
    """旧快照（无 valid_at/invalid_at 字段）兼容读取。"""
    e = fresh_engine
    e.config.factversion.enabled = True
    # 手工构造旧格式：旧条只有 superseded_by，新条只有 supersedes（无时间戳）
    old = e.memory_manager.add_memory(
        text=WUHOU, metadata={"fact_kind": "fact"}
    )
    t1 = now()
    new = e.memory_manager.add_memory(
        text=CHENGHUA,
        metadata={
            "fact_kind": "fact",
            SUPERSEDE_TAG: old.id,
            "corrects": old.id,
            VALID_AT_KEY: t1,
        },
    )
    old.metadata[STALE_TAG] = new.id  # legacy：没有 invalid_at 戳

    # 旧条：valid_at 视为 None；invalid_at 从 superseded 状态推断
    assert valid_at_of(old) is None
    assert invalid_at_of(old, e.memories) == t1
    # 新条：现行有效
    assert valid_at_of(new) == t1
    assert invalid_at_of(new, e.memories) is None
    # 完全没有链指针的旧条：双时间戳都视为 None
    plain = e.memory_manager.add_memory(
        text=UNRELATED, metadata={"fact_kind": "fact"}
    )
    assert valid_at_of(plain) is None and invalid_at_of(plain, e.memories) is None
    # 推断不落盘：兼容读取是纯读路径，旧 metadata 不被改写
    assert INVALID_AT_KEY not in old.metadata
    # timeline 走同一套兼容读取
    versions = e.factversion.timeline(CHENGHUA, e)
    assert len(versions) == 2
    assert versions[0]["valid_at"] is None and versions[0]["invalid_at"] == t1
    assert versions[1]["invalid_at"] is None


def test_legacy_successor_without_valid_at_falls_back_to_created_at(
    fresh_engine, new_engine
):
    """更老的快照：连新条也没有 valid_at，invalid_at 推断退回新条 created_at。"""
    e = fresh_engine
    e.config.factversion.enabled = True
    old = e.memory_manager.add_memory(
        text=WUHOU, metadata={"fact_kind": "fact"}
    )
    new = e.memory_manager.add_memory(
        text=CHENGHUA,
        metadata={"fact_kind": "fact", SUPERSEDE_TAG: old.id},
    )
    old.metadata[STALE_TAG] = new.id
    assert invalid_at_of(old, e.memories) == pytest.approx(new.created_at)
    versions = e.factversion.timeline(new, e)
    assert versions[0]["invalid_at"] == pytest.approx(new.created_at)


# ------------------------- zero drift with module off ----------------------- #
def test_module_off_zero_temporal_drift(fresh_engine, new_engine):
    e = fresh_engine
    e.config.extraction.enabled = True
    e.config.extraction.mode = "rules"
    e.config.factversion.enabled = False
    e.add(COFFEE)
    e.add(NO_COFFEE)   # 模块关：纠正句当普通新事实存，无版本链无时间戳
    assert len(e.memories) == 2
    for m in e.memories.values():
        assert not any(
            k in m.metadata for k in (STALE_TAG, SUPERSEDE_TAG, VALID_AT_KEY, INVALID_AT_KEY)
        )
    # 模块关 => timeline 恒为空（REST 同口径 enabled=False）
    assert e.factversion.timeline(NO_COFFEE, e) == []


# ------------------------- REST smoke ---------------------------------------- #
def test_rest_facts_and_history_smoke(fresh_engine, new_engine):
    from fastapi.testclient import TestClient

    from sme.api.server import create_app

    e = _temporal_engine(fresh_engine)
    e.add(COFFEE)
    e.add(NO_COFFEE)
    e.add(AGAIN_COFFEE)
    client = TestClient(create_app(e))

    # GET /facts：每条 fact 附 valid_at / invalid_at
    r = client.get("/facts")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["facts"], "/facts should list fact versions"
    for f in body["facts"]:
        assert "valid_at" in f and "invalid_at" in f
        assert f["is_current"] == (f["invalid_at"] is None)
    assert any(f["invalid_at"] is not None for f in body["facts"])   # 旧版本
    assert any(f["is_current"] for f in body["facts"])               # 现行版本
    # 旧 -> 新排序
    facts = body["facts"]
    assert facts == sorted(facts, key=lambda f: f["created_at"])

    # GET /facts/history?text=...：返回版本链（旧 -> 新）
    r = client.get("/facts/history", params={"text": AGAIN_COFFEE})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True and body["count"] == 3
    v1, v2, v3 = body["versions"]
    assert v1["invalid_at"] == v2["valid_at"] and v2["invalid_at"] == v3["valid_at"]
    assert v3["invalid_at"] is None and v3["is_current"]

    # memory_id 参数等价
    r = client.get("/facts/history", params={"memory_id": v3["id"]})
    assert r.status_code == 200 and r.json()["count"] == 3

    # 参数缺失 / 查无此事实
    assert client.get("/facts/history").status_code == 400
    assert client.get("/facts/history", params={"text": UNRELATED}).status_code == 404


def test_rest_history_disabled_module(fresh_engine, new_engine):
    from fastapi.testclient import TestClient

    from sme.api.server import create_app

    e = fresh_engine
    e.config.factversion.enabled = False
    client = TestClient(create_app(e))
    body = client.get("/facts").json()
    assert body == {"enabled": False, "entities": [], "relations": [], "facts": []}
    body = client.get("/facts/history", params={"text": COFFEE}).json()
    assert body == {"enabled": False, "versions": []}
