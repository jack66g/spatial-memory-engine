"""embedding 向量重建工具（模型无关化基建）。

embedding 模型是可替换耗材，但换模型（或开关 MRL 截维）= 向量空间整体作废：
Qwen3 和 bge-m3 都是 1024 维，同维度换模型不会触发维度报错，只会静默产生
垃圾相似度。本工具是官方重建路径：

    1. 读取旧快照（不动磁盘）；
    2. 用【新配置】的 provider 批量（embedding.batch_size）重嵌全部记忆；
    3. 在新引擎里重建空间索引（Region 重聚类）；
    4. 保存新快照（config 内写入新 revision 标记）。

图边 / 版本链 / 时间戳 / 权重 / 归档状态等非向量状态经
``Memory.from_dict`` 全字段还原，原样保留。失败安全：全部向量先在内存中
重嵌完成才开始导入，中途失败不落任何半成品快照。

用法::

    python -m sme.rebuild_embeddings --config data/sme.config.json
    python -m sme.rebuild_embeddings --config <path> --storage <path> --dry-run

流程约定：先在 Web 配置中心（或手改配置文件）把 embedding.provider/model/
dim/mrl_dim/revision 改成新值，再运行本工具；--dry-run 只对比新旧 revision
并报告影响面，不改动数据。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from sme.config import SMEConfig
from sme.config_items import load_config, validate_config
from sme.engine import SpatialMemoryEngine, embedding_revision
from sme.models import Memory
from sme.storage.backends import build_storage_backend
from sme.storage.wal import default_wal_path


def _say(msg: str) -> None:
    print(msg, flush=True)


def _load_snapshot_for(path: str):
    """按文件签名选后端（json/sqlite）加载快照；不存在返回 None。"""
    backend = build_storage_backend(SpatialMemoryEngine._detect_backend(path))
    return backend.load(path)


def _planned_effective_dim(config: SMEConfig) -> int:
    """dry-run 用：不构建 provider 的情况下推算向量维度（预计值）。"""
    emb = config.embedding
    if 0 < emb.mrl_dim < emb.dim:
        return emb.mrl_dim
    return emb.dim


def _snapshot_vector_dim(snapshot) -> int:
    for m in snapshot.memories:
        if m.embedding is not None:
            return int(len(m.embedding))
    return 0


def _checkpoint_pending_wal(snapshot, path: str, backend_name: str) -> str:
    """把旧快照未 checkpoint 的 WAL 先落成完整快照，避免重建丢写入。

    返回 "clean"（无待落盘 ops）/ "checkpointed"（已补落盘，需重读快照）/
    "failed"（调用方中止重建，数据保持原状）。重建不经过 engine.load（那会
    把引擎配置回绑成旧快照的配置），因此必须先用【快照自己的配置】起一个
    临时引擎 replay+save 一次。
    """
    old = snapshot.config
    if not old.persistence.enabled:
        return "clean"
    if backend_name != "sqlite":
        wal_file = old.persistence.wal_path or default_wal_path(
            old.storage.path or path
        )
        if not (os.path.exists(wal_file) and os.path.getsize(wal_file) > 0):
            return "clean"  # 没有待重放的 ops
    try:
        # from_dict(to_dict()) 做一份深拷贝，临时引擎的回绑不污染 snapshot
        tmp = SpatialMemoryEngine(SMEConfig.from_dict(old.to_dict()))
        tmp.config.storage.path = old.storage.path or path
        tmp.load(old.storage.path or path)  # replay WAL 进内存
        tmp.save()                          # checkpoint（含 WAL 清空）
        _say("[rebuild] 已将未落盘的 WAL 写入 checkpoint，开始重建")
        return "checkpointed"
    except Exception as exc:  # noqa: BLE001 - 任何失败都中止，绝不带缺数据重建
        _say(
            f"[rebuild] 错误：检测到未 checkpoint 的 WAL，且用旧配置回放失败"
            f"（{exc}）。请先用原配置启动一次引擎（python -m sme.api）完成"
            "落盘后再重建，否则窗口内的写入会丢失"
        )
        return "failed"


def main(argv: Optional[list[str]] = None) -> int:
    # Windows 控制台兜底：编码不动，只把无法编码的字符替换掉而不是崩溃
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - 老版本 python 没有 reconfigure
            pass

    parser = argparse.ArgumentParser(
        prog="python -m sme.rebuild_embeddings",
        description="用当前配置的 embedding provider 重嵌全部记忆并重建空间索引"
                    "（换模型 / 开关 MRL 截维后必须执行一次）",
    )
    parser.add_argument(
        "--config", default="data/sme.config.json",
        help="新配置文件路径（与 Web 配置中心同格式的 JSON；默认 data/sme.config.json）",
    )
    parser.add_argument(
        "--storage", default=None,
        help="引擎快照路径（默认取配置里的 storage.path）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只对比新旧向量空间 revision 并报告影响面，不改动任何数据",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        _say(f"[rebuild] 配置文件不存在：{args.config}（用 --config 指定新配置）")
        return 2

    raw = load_config(args.config)
    for line in validate_config(raw):
        _say(f"[rebuild] 配置告警：{line}")
    config = SMEConfig.from_dict(raw.get("sme", raw))
    # 与 REST 服务同口径：配置文件留空时读环境变量里的 embedding 密钥
    if not config.embedding.api_key:
        config.embedding.api_key = os.environ.get("SME_EMBEDDING_API_KEY", "")
    if args.storage:
        config.storage.path = args.storage
    storage_path = config.storage.path
    if not storage_path:
        _say("[rebuild] 未确定快照路径（--storage 或配置 storage.path）")
        return 2

    snapshot = _load_snapshot_for(storage_path)
    if snapshot is None:
        _say(f"[rebuild] 快照不存在：{storage_path}（重建需要一个已有的记忆库）")
        return 2

    # ---------------- 对比新旧向量空间 ---------------- #
    old_revision = (
        snapshot.config.embedding.revision_stamp
        or embedding_revision(snapshot.config)
    )
    new_revision = embedding_revision(config)
    total = len(snapshot.memories)
    archived = sum(1 for m in snapshot.memories if m.archived)
    active = total - archived
    old_dim = _snapshot_vector_dim(snapshot)
    new_dim = _planned_effective_dim(config)

    _say(f"快照: {storage_path}")
    _say(f"记忆数: {total}（活跃 {active} / 归档 {archived}）")
    _say(f"旧 revision: {old_revision}")
    _say(f"新 revision: {new_revision}")
    _say(f"向量维度: {old_dim or '（空库）'} → {new_dim}（预计）")

    if old_revision == new_revision:
        _say("结论: 向量空间一致，无需重建")
        return 0

    if args.dry_run:
        _say(
            f"结论: 向量空间不一致，{total} 条记忆需要重嵌入；"
            "去掉 --dry-run 重新运行即可执行重建"
        )
        return 0

    # ---------------- 实际重建 ---------------- #
    backend_name = SpatialMemoryEngine._detect_backend(storage_path)
    wal_state = _checkpoint_pending_wal(snapshot, storage_path, backend_name)
    if wal_state == "failed":
        return 1
    if wal_state == "checkpointed":
        # checkpoint 之后重读：快照里现在包含了 WAL 重放的写入
        snap2 = _load_snapshot_for(storage_path)
        if snap2 is not None:
            snapshot = snap2

    t_start = time.perf_counter()
    try:
        engine = SpatialMemoryEngine(config)  # 新 provider，空库
    except Exception as exc:  # noqa: BLE001 - provider 起不来（模型缺失/密钥错）
        _say(f"[rebuild] 错误：新配置无法构建 embedding provider：{exc}")
        return 1

    # 重建期间禁用 autosave：导入中途的周期快照不该覆盖旧库
    engine.config.storage.autosave = False
    # 图边先恢复（import 不碰 graph；这样 WAL 模式下 import 触发的即时
    # checkpoint 也带着完整图边）
    engine.graph.load_dict(
        {"edges": [e.to_dict() for e in snapshot.memory_edges]}
    )

    # 批量重嵌：全部向量在内存中完成后才开始导入（失败安全）
    items = [m.to_dict(include_embedding=False) for m in snapshot.memories if m.text]
    memories = [Memory.from_dict(d) for d in items]
    texts = [m.text for m in memories]
    vectors = []
    batch = max(1, config.embedding.batch_size)
    t_embed = time.perf_counter()
    try:
        for start in range(0, len(texts), batch):
            vectors.extend(engine.embeddings.embed(texts[start : start + batch]))
    except Exception as exc:  # noqa: BLE001 - 重嵌失败：旧快照未动，直接退出
        _say(f"[rebuild] 错误：重嵌入失败（已完成 {len(vectors)}/{len(texts)}）：{exc}")
        return 1
    embed_secs = time.perf_counter() - t_embed
    for memory, vec in zip(memories, vectors):
        memory.embedding = vec

    # 全字段还原登记：版本链/时间戳/权重/parent/children/归档均原样保留，
    # 活跃记忆经 _register 重新进空间索引 + 检索索引（Region 重聚类）
    count = engine.import_memories([m.to_dict() for m in memories])

    # 非向量状态：演化/融合/压缩计数器按旧库回填（与 REST 热重建同口径）
    engine.region_manager.split_count = snapshot.counters.get("splits", 0)
    engine.region_manager.merge_count = snapshot.counters.get("merges", 0)
    engine.consolidation.consolidation_count = snapshot.counters.get(
        "consolidations", 0
    )
    engine.compression.compression_count = snapshot.counters.get(
        "compressions", 0
    )

    saved = engine.save()  # 写入新快照 + 新 revision 标记，并清空 WAL
    elapsed = time.perf_counter() - t_start
    throughput = (count / embed_secs) if embed_secs > 0 and count else 0.0

    _say("重建完成：")
    _say(f"  记忆数: {count}（重嵌入 {count} 条，批量大小 {batch}）")
    _say(f"  向量维度: {old_dim or '（空库）'} → {engine.embeddings.effective_dim}")
    _say(f"  Region: 重新聚类为 {len(engine.space.regions)} 个")
    _say(f"  图边: 保留 {len(engine.graph)} 条")
    _say(f"  重嵌耗时: {embed_secs:.2f}s（吞吐 {throughput:.1f} 条/秒）")
    _say(f"  总耗时: {elapsed:.2f}s")
    _say(f"  新快照: {saved}（revision={new_revision}）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
