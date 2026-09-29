"""模型无关化基建的回归测试：embedding revision 标记 + MRL 截维 + 重建工具。

覆盖三块：
1. revision 不一致时 load 的 WARNING 软防护（capsys 捕获 stderr handler）；
2. MRL 截维机制（先截再归一、维度上报、排序基本保持 Spearman > 0.9）；
3. ``python -m sme.rebuild_embeddings`` 端到端（含 WAL 未落盘 checkpoint、
   非向量状态无损、dry-run 不动数据）。
"""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import contextmanager

import numpy as np
import pytest

import sme.rebuild_embeddings as rb
from sme.config import SMEConfig
from sme.config_items import ITEMS, ITEM_BY_PATH, load_config
from sme.embedding import build_embedding_provider
from sme.embedding.hashing import HashingEmbeddingProvider
from sme.engine import SpatialMemoryEngine, embedding_revision
from sme.storage.snapshot import load_snapshot


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _engine(tmp_path, name="e.json", embedding=None, **overrides):
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / name)
    for key, value in (embedding or {}).items():
        setattr(cfg.embedding, key, value)
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(cfg, section), field, value)
    return SpatialMemoryEngine(cfg)


@contextmanager
def _sme_stderr_handler():
    """把 sme logger 的输出接到当前 sys.stderr（capsys 可捕获）。

    pytest 的 logging 插件会在根上挂捕获 handler，使 logging 的 lastResort
    stderr 通道不触发；这里显式挂一个流 handler 保证 capsys 能看到 WARNING。
    """
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("sme")
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def _ranks(values):
    """Average ranks（并列取平均），Spearman 用。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    if denom < 1e-12:
        return 0.0
    return float(a @ b) / denom


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _pairwise_cos(vectors):
    mat = np.stack(vectors)
    return [float(x) for x in (mat @ mat.T).reshape(-1)]


# --------------------------------------------------------------------------- #
# 1. revision 标记与 load 防护
# --------------------------------------------------------------------------- #
def test_load_warns_on_revision_mismatch(tmp_path, capsys):
    path = str(tmp_path / "e.json")
    a = _engine(tmp_path)
    a.add("user likes apples")
    a.save()

    b = _engine(tmp_path, embedding={"revision": "space-v2"})
    with _sme_stderr_handler():
        b.load(path)
    err = capsys.readouterr().err
    assert "向量空间不匹配" in err
    assert "python -m sme.rebuild_embeddings" in err
    # 两个 revision 都要在告警里说清楚
    assert "space-v2" in err


def test_load_warns_on_mrl_change(tmp_path, capsys):
    """mrl_dim 改变 = 向量空间改变（provider/model/dim 全相同也要告警）。"""
    path = str(tmp_path / "e.json")
    a = _engine(tmp_path, embedding={"dim": 1024})
    a.add("user lives in beijing")
    a.save()

    b = _engine(tmp_path, embedding={"dim": 1024, "mrl_dim": 512})
    with _sme_stderr_handler():
        b.load(path)
    err = capsys.readouterr().err
    assert "向量空间不匹配" in err


def test_load_no_warning_when_revision_matches(tmp_path, capsys):
    path = str(tmp_path / "e.json")
    a = _engine(tmp_path)
    a.add("user likes apples")
    a.save()

    b = _engine(tmp_path)
    with _sme_stderr_handler():
        assert b.load(path)
    err = capsys.readouterr().err
    assert "向量空间不匹配" not in err


def test_load_no_warning_on_empty_library(tmp_path, capsys):
    """空库没有向量可信度问题，不告警。"""
    path = str(tmp_path / "e.json")
    a = _engine(tmp_path)
    a.save()

    b = _engine(tmp_path, embedding={"revision": "space-v2"})
    with _sme_stderr_handler():
        assert b.load(path)
    err = capsys.readouterr().err
    assert "向量空间不匹配" not in err


def test_revision_stamp_written_into_snapshot(tmp_path):
    """save() 把组合 revision 写进快照 config（含 mrl/hash_seed/用户标签）。"""
    path = str(tmp_path / "e.json")
    a = _engine(tmp_path, embedding={"mrl_dim": 32, "revision": "qwen3-v2"})
    a.add("stamp me")
    a.save()

    snap = load_snapshot(path)
    assert snap.config.embedding.revision_stamp == embedding_revision(a.config)
    assert "mrl32" in snap.config.embedding.revision_stamp
    assert "qwen3-v2" in snap.config.embedding.revision_stamp


def test_embedding_revision_composition():
    cfg = SMEConfig()
    assert embedding_revision(cfg) == "hashing|text-embedding-3-small|64"
    cfg.embedding.revision = " v9 "
    assert embedding_revision(cfg).endswith("|v9")
    cfg.embedding.mrl_dim = 512
    cfg.embedding.hash_seed = 7
    assert embedding_revision(cfg) == (
        "hashing|text-embedding-3-small|64|mrl512|seed7|v9"
    )


def test_new_config_items_registered():
    assert "embedding.revision" in ITEM_BY_PATH
    assert "embedding.mrl_dim" in ITEM_BY_PATH
    mrl = ITEM_BY_PATH["embedding.mrl_dim"]
    assert mrl.kind == "int" and mrl.default == 0
    assert ITEM_BY_PATH["embedding.revision"].group == "Embedding 向量"
    # SMEConfig 往返序列化不丢新字段
    cfg = SMEConfig()
    cfg.embedding.revision = "v2"
    cfg.embedding.mrl_dim = 512
    restored = SMEConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert restored.embedding.revision == "v2"
    assert restored.embedding.mrl_dim == 512


# --------------------------------------------------------------------------- #
# 2. MRL 截维
# --------------------------------------------------------------------------- #
_MRL_TOPICS = [
    "apple banana orange fruit garden",
    "engine wheel brake car garage",
    "python java rust code compiler",
    "river mountain forest hiking trail",
    "coffee tea sugar kitchen morning",
]


def test_mrl_truncation_mechanics():
    texts = [f"{t} note {i}" for t in _MRL_TOPICS for i in range(6)]
    full = HashingEmbeddingProvider(dim=1024, factors=3, window=3)
    trunc = HashingEmbeddingProvider(dim=1024, factors=3, window=3, mrl_dim=512)

    assert full.effective_dim == 1024
    assert trunc.effective_dim == 512  # dim 上报 = mrl_dim（截维生效时）

    vf = full.embed(texts)
    vt = trunc.embed(texts)
    assert all(v.shape[0] == 1024 for v in vf)
    assert all(v.shape[0] == 512 for v in vt)
    # 截断后必须重归一（前缀范数 < 1，不归一点积就不是余弦）
    assert all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-9 for v in vt)

    # 机制层面精确性：截断向量 == normalize(全维向量前 512 维)（先截再归一）
    for a, b in zip(vf, vt):
        prefix = a[:512]
        assert np.allclose(b, prefix / np.linalg.norm(prefix), atol=1e-12)

    # 排序基本保持：截维后 cos 与全维 cos 的 Spearman > 0.9
    rho = _spearman(_pairwise_cos(vf), _pairwise_cos(vt))
    assert rho > 0.9, f"MRL 截维后相似度排序失真：Spearman={rho:.3f}"


def test_mrl_inactive_when_ge_native_dim():
    """mrl_dim >= 原生维度（或 0）时不截断，行为与旧版一致。"""
    off = HashingEmbeddingProvider(dim=64, mrl_dim=0)
    equal = HashingEmbeddingProvider(dim=64, mrl_dim=64)
    bigger = HashingEmbeddingProvider(dim=64, mrl_dim=128)
    for p in (off, equal, bigger):
        assert p.effective_dim == 64
        assert p.embed_one("hello world").shape[0] == 64


def test_mrl_factory_and_engine_space_dim(tmp_path):
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = str(tmp_path / "mrl.json")
    cfg.embedding.dim = 1024
    cfg.embedding.mrl_dim = 512
    engine = SpatialMemoryEngine(cfg)

    provider = build_embedding_provider(cfg.embedding)
    assert provider.mrl_dim == 512 and provider.effective_dim == 512
    # 空间维度跟随 effective_dim（MRL 生效时是 512 而非原生 1024）
    assert engine.space.dim == 512
    mem = engine.add("mrl aware memory")
    assert mem.embedding.shape[0] == 512

    engine.save()
    reloaded = SpatialMemoryEngine(cfg)
    assert reloaded.load(str(tmp_path / "mrl.json"))
    assert reloaded.space.dim == 512
    hits = reloaded.search("mrl aware", top_k=3)
    assert hits and hits[0].memory.id == mem.id


# --------------------------------------------------------------------------- #
# 3. rebuild 端到端
# --------------------------------------------------------------------------- #
def _write_config(path, storage, embedding=None, extra_sme=None):
    sme = {
        "embedding": {"provider": "hashing", "dim": 64},
        "storage": {"path": storage, "autosave": False},
    }
    if embedding:
        sme["embedding"].update(embedding)
    if extra_sme:
        sme.update(extra_sme)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"sme": sme}, fh, ensure_ascii=False, indent=2)
    return str(path)


def _snapshot_files(storage):
    base = storage[:-3] if storage.endswith(".gz") else storage
    if base.endswith(".json"):
        base = base[:-5]
    files = {}
    for suffix in (".json", ".json.gz", ".embeddings.npz"):
        p = base + suffix
        if os.path.exists(p):
            with open(p, "rb") as fh:
                files[p] = fh.read()
    return files


def test_rebuild_end_to_end(tmp_path, capsys):
    storage = str(tmp_path / "engine.json")
    a = _engine(tmp_path, name="engine.json")
    texts = [f"记忆条目 {i}：关于主题 {i % 4} 的第 {i} 条记录" for i in range(20)]
    mems = a.add_many(texts)
    # 图边（多种 kind）+ 版本链（parent/children）+ 版本号扰动
    a.link(mems[0].id, mems[1].id, kind="reference", note="related")
    a.link(mems[2].id, mems[3].id, kind="cause")
    a.memory_manager.set_parent(mems[5].id, mems[4].id)
    a.update(mems[6].id, importance=0.9)  # version -> 2
    a.save()

    before = {mid: m.to_dict() for mid, m in a.memories.items()}
    old_edges = sorted(
        (e.source, e.target, e.kind, e.weight, e.note) for e in a.graph_edges()
    )
    old_vectors = {
        mid: np.array(m["embedding"]) for mid, m in before.items()
    }
    files_before = _snapshot_files(storage)

    # 新配置：同 provider/同维度、换 hash_seed（模拟换向量空间）+ 新 revision
    cfg_file = _write_config(
        tmp_path / "sme.config.json", storage,
        embedding={"hash_seed": 7, "revision": "space-v2"},
    )

    # --- dry-run：报告影响面，不动数据 --- #
    rc = rb.main(["--config", cfg_file, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "space-v2" in out
    assert "20 条记忆需要重嵌入" in out
    assert _snapshot_files(storage) == files_before

    # --- 实际重建 --- #
    rc = rb.main(["--config", cfg_file])
    assert rc == 0
    out = capsys.readouterr().out
    assert "重建完成" in out
    assert "Region" in out and "图边" in out

    b = _engine(tmp_path, name="engine.json",
                embedding={"hash_seed": 7, "revision": "space-v2"})
    assert b.load(storage)

    # 记忆数不变，非向量状态无损（时间戳/版本链/图边/权重全保留）
    assert len(b.memories) == 20
    for mid, old in before.items():
        m = b.memories[mid]
        assert m.text == old["text"]
        assert m.created_at == old["created_at"]
        assert m.last_hit == old["last_hit"]
        assert m.version == old["version"]
        assert m.importance == old["importance"]
        assert m.parent_id == old["parent_id"]
        assert m.children == old["children"]
        assert m.neighbors == set(old["neighbors"])
    assert sorted(
        (e.source, e.target, e.kind, e.weight, e.note) for e in b.graph_edges()
    ) == old_edges

    # 向量全部更新：等于新 provider 直嵌，且不等于旧向量
    for mid in list(before)[:6]:
        fresh = b.embeddings.embed_one(before[mid]["text"])
        assert np.allclose(b.memories[mid].embedding, fresh, atol=1e-12)
        assert not np.allclose(b.memories[mid].embedding, old_vectors[mid])

    # 新 revision 已写进快照；旧引擎 revision 视角再 load 会得到告警
    snap = load_snapshot(storage)
    assert snap.config.embedding.revision_stamp == embedding_revision(b.config)

    # 检索正常（重建后的库能查出相关内容）
    hits = b.search("记忆条目 3 主题", top_k=5)
    assert hits
    assert any("记忆条目 3" in h.memory.text for h in hits)


def test_rebuild_noop_when_revision_matches(tmp_path, capsys):
    storage = str(tmp_path / "e.json")
    a = _engine(tmp_path)
    a.add("stable memory")
    a.save()
    cfg_file = _write_config(tmp_path / "sme.config.json", storage)
    rc = rb.main(["--config", cfg_file])
    assert rc == 0
    out = capsys.readouterr().out
    assert "向量空间一致，无需重建" in out


def test_rebuild_missing_config_and_snapshot(tmp_path, capsys):
    rc = rb.main(["--config", str(tmp_path / "nope.json")])
    assert rc == 2
    cfg_file = _write_config(tmp_path / "sme.config.json", str(tmp_path / "no-snap.json"))
    rc = rb.main(["--config", cfg_file])
    assert rc == 2
    capsys.readouterr()


def test_rebuild_checkpoints_pending_wal(tmp_path, capsys):
    """快照之后还有 WAL 未落盘的写入：重建前先 checkpoint，一条不丢。"""
    storage = str(tmp_path / "wal.json")
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.storage.path = storage
    cfg.persistence.enabled = True
    cfg.persistence.checkpoint_every = 100  # 避免 add 过程中自动 checkpoint
    a = SpatialMemoryEngine(cfg)
    a.add("base memory")
    a.save()  # 生成基线快照并清空 WAL
    a.add("wal memory one")
    a.add("wal memory two")
    assert os.path.exists(storage + ".wal")
    assert os.path.getsize(storage + ".wal") > 0
    # 模拟旧引擎进程退出（Windows 下句柄不释放，重建工具删不掉 WAL 文件）
    a.wal.close()

    cfg_file = _write_config(
        tmp_path / "sme.config.json", storage,
        embedding={"hash_seed": 7, "revision": "space-v2"},
    )
    rc = rb.main(["--config", cfg_file])
    assert rc == 0
    out = capsys.readouterr().out
    assert "已将未落盘的 WAL 写入 checkpoint" in out

    b = _engine(tmp_path, name="wal.json",
                embedding={"hash_seed": 7, "revision": "space-v2"})
    assert b.load(storage)
    texts = {m.text for m in b.memories.values()}
    assert texts == {"base memory", "wal memory one", "wal memory two"}
    # 重建后 WAL 已被最终 save 清空
    assert not os.path.exists(storage + ".wal") or \
        os.path.getsize(storage + ".wal") == 0
