"""W26：REST 层 ns（命名空间）隔离——把引擎既有能力暴露为一等 API 字段。

前置事实：引擎 add/search 的 ns 参数与 Module 12（namespaces 视图层）
早已存在，REST 侧的 ``AddMemoryRequest.ns`` / ``SearchRequest.ns`` 只是
透传。本文件锁住对外契约：

  - 带 ns 写入 → ``metadata["ns"]`` 打标签
  - 带 ns 检索 → 只见本 ns 的记忆（跨 ns / 无标签记忆均不可见）
  - 不带 ns   → 与 v1 行为完全一致（写入不打标签、检索不过滤）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client(fresh_engine):
    from fastapi.testclient import TestClient

    from sme.api.server import create_app

    return TestClient(create_app(fresh_engine))


def test_rest_ns_write_and_isolation(fresh_engine, zh):
    client = _client(fresh_engine)
    r_a = client.post("/memories", json={"text": zh["likes_coffee"], "ns": "user_a"})
    r_b = client.post("/memories", json={"text": zh["lives_beijing"], "ns": "user_b"})
    assert r_a.status_code == 200 and r_b.status_code == 200
    # 写入侧：metadata 打 ns 标签
    assert r_a.json()["metadata"]["ns"] == "user_a"
    assert r_b.json()["metadata"]["ns"] == "user_b"

    # 检索侧：user_a 只见自己的记忆（互不可见是双向的）
    hits_a = client.post(
        "/memories/search", json={"text": zh["likes_coffee"], "ns": "user_a"}
    ).json()
    texts_a = [h["text"] for h in hits_a["results"]]
    assert zh["likes_coffee"] in texts_a
    assert all(h["metadata"].get("ns") == "user_a" for h in hits_a["results"])

    hits_b = client.post(
        "/memories/search", json={"text": zh["lives_beijing"], "ns": "user_b"}
    ).json()
    texts_b = [h["text"] for h in hits_b["results"]]
    assert zh["lives_beijing"] in texts_b
    assert zh["likes_coffee"] not in texts_b


def test_rest_ns_absent_is_v1(fresh_engine, zh):
    client = _client(fresh_engine)
    client.post("/memories", json={"text": zh["likes_coffee"], "ns": "user_a"})
    r_plain = client.post("/memories", json={"text": zh["lives_beijing"]})
    # 不带 ns 写入：无标签（v1 行为）
    assert "ns" not in (r_plain.json()["metadata"] or {})

    # 不带 ns 检索 = 不过滤（v1 行为）：ns 记忆照常可见
    hits = client.post(
        "/memories/search", json={"text": zh["likes_coffee"]}
    ).json()
    assert zh["likes_coffee"] in [h["text"] for h in hits["results"]]

    # ns 过滤检索：无标签记忆对本 ns 不可见
    hits_a = client.post(
        "/memories/search", json={"text": zh["lives_beijing"], "ns": "user_a"}
    ).json()
    assert zh["lives_beijing"] not in [h["text"] for h in hits_a["results"]]
