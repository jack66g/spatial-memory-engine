"""SpatialMemoryEngine - the facade of the whole system.

Wires together: embedding engine, spatial memory space, region manager,
memory manager, memory graph, two-stage retriever, ranker, policy,
reinforcement, decay, consolidation, compression, archive, visualization,
benchmark and optional LLM client.

Usage:
    engine = SpatialMemoryEngine()          # offline hashing embeddings
    engine.add("user likes apples")
    hits = engine.search("what fruits does the user like?")
"""

from __future__ import annotations

import json
import os
import threading
import time
import weakref
from typing import Any, Iterable, Optional

import numpy as np

from sme.dynamics.archive import ArchiveManager
from sme.config import SMEConfig
from sme.dynamics.consolidation import ConsolidationEngine
from sme.dynamics.decay import MemoryDecay
from sme.embedding import build_embedding_provider
from sme.modules.memory_graph import MemoryGraph
from sme.llm import LLMClient
from sme.memory_manager import MemoryManager
from sme.models import Memory, MemoryStats, RegionStats, SearchHit
from sme.dynamics.policy import MemoryPolicy
from sme.retrieval.ranking import MemoryRanker
from sme.dynamics.reinforcement import EbbinghausReinforcement
from sme.retrieval import SearchQuery, TwoStageRetriever
from sme.space import RegionManager, SpatialMemorySpace
from sme.storage.snapshot import EngineSnapshot
from sme.utils import logger, now

# v2 modules (v2 模块设计) - all disabled by default, no-op when off
from sme.modules.extraction import ExtractionEngine
from sme.modules.factversion import FactVersion
from sme.modules.noise import NoiseScorer
from sme.modules.qapair import QAPairStore
from sme.modules.factgraph import FactGraph, FactGraphExtractor
from sme.modules.profile import UserProfile
from sme.storage.wal import WriteAheadLog
from sme.modules.observability import MemoryTelemetry
from sme.modules.context import ContextManager
from sme.modules.namespaces import Namespaces, NS_KEY
from sme.storage.backends import build_storage_backend
from sme.modules.bridge import V2Bridge


def embedding_revision(config: SMEConfig) -> str:
    """当前配置的向量空间 revision（换 embedding = 换向量空间的身份标记）。

    由 provider/model/dim/mrl_dim/hash_seed + 用户手填的 revision 标签组成；
    engine.save() 时随快照 config 落盘，load() 时与当前引擎对比——不一致
    说明快照里的向量与当前 provider 不是同一空间（同维度换模型不会报维度
    错，只能靠它发现），需要 ``python -m sme.rebuild_embeddings`` 重建。
    """
    emb = config.embedding
    parts = [emb.provider or "hashing", emb.model or "", str(emb.dim)]
    if emb.mrl_dim and emb.mrl_dim > 0:
        parts.append(f"mrl{emb.mrl_dim}")
    if emb.hash_seed:
        parts.append(f"seed{emb.hash_seed}")
    tag = (emb.revision or "").strip()
    if tag:
        parts.append(tag)
    return "|".join(parts)


class SpatialMemoryEngine:
    def __init__(
        self,
        config: Optional[SMEConfig] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path:
            with open(config_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            config = SMEConfig.from_dict(loaded.get("sme", loaded))
        self.config = config or SMEConfig()
        # 用户是否显式传过配置：load 的向量空间告警只对"显式配置与快照
        # 不一致"生效（默认构造的引擎本就没有向量空间意图，load 后整体
        # 采纳快照配置即自洽，不告警）
        self._explicit_config = config is not None or bool(config_path)

        # --- subsystems -------------------------------------------------- #
        logger.info("engine init: provider=%s model=%s dim=%d",
                    self.config.embedding.provider,
                    self.config.embedding.model, self.config.embedding.dim)
        self.embeddings = build_embedding_provider(self.config.embedding)
        self._calibrate_region_threshold()
        self.llm = LLMClient(self.config.llm)
        # 空间维度取 provider 的 effective_dim：MRL 截维（mrl_dim>0 且小于
        # 原生维度）时实际产出的向量是 mrl_dim 维，Region 几何必须与之对齐
        self.space = SpatialMemorySpace(self.config.region, self.embeddings.effective_dim)
        self.region_manager: RegionManager = self.space.manager
        self.policy = MemoryPolicy(self.config.policy)
        self.reinforcement = EbbinghausReinforcement(self.config.reinforcement)
        self.decay = MemoryDecay(
            self.config.decay, enabled=self.config.policy.decay_enabled
        )
        self.archive_manager = ArchiveManager(
            cold_path=(
                os.path.join(
                    os.path.dirname(self.config.storage.path), "cold_archive.json"
                )
                if self.config.storage.path
                else None
            )
        )
        self.graph = MemoryGraph()
        self.ranker = MemoryRanker(self.config.ranking)
        self.retriever = TwoStageRetriever(self.config.retrieval, self.ranker)
        self.memory_manager = MemoryManager(
            self.config,
            self.space,
            self.embeddings,
            self.graph,
            self.policy,
            self.reinforcement,
            self.decay,
            self.archive_manager,
            on_upsert=self.retriever.index_memory,
            on_delete=self.retriever.drop_memory,
        )
        self.consolidation = ConsolidationEngine(self.config.consolidation, self.llm)
        self.compression = CompressionEngineProxy(self.config, self.llm)
        self.space.set_membership_hook(self._sync_memory_region)
        self._autosave_counter = 0
        self._lock = threading.RLock()
        # --- sleep-time compute：空闲期后台整理（maintenance.background） --
        # 写活动时间戳（空闲判定基准）；维护线程状态；观测量。
        # background=False（默认）时只有这几个轻量属性，零行为变化。
        self._last_write_ts = now()
        self._maint_lock = threading.Lock()   # 只保护线程的启停，不跨任何 IO
        self._maint_stop = threading.Event()
        self._maint_thread: Optional[threading.Thread] = None
        self._maint_finalizer = None
        self._maint_runs = 0
        self._maint_last_run_at: Optional[float] = None
        self._maint_last_duration_s: Optional[float] = None
        self._maint_last_error: Optional[str] = None

        # --- v2 modules (v2 模块设计); all disabled => v1 behavior ---- #
        self.extraction = ExtractionEngine(self.config.extraction, llm=self.llm)
        self.factversion = FactVersion(self.config.factversion)
        self.noise = NoiseScorer(self.config.noise)
        self.qapair = QAPairStore(self.config.qapair)
        self.factgraph = FactGraph(self.config.factgraph)
        self.factgraph_extractor = FactGraphExtractor(self.config.factgraph, llm=self.llm)
        self.profile = UserProfile(self.config.profile)
        self.profile.bind(self)
        self.wal = WriteAheadLog(
            self.config.persistence, self.config.storage,
            sqlite=self.config.storage.backend == "sqlite",
        )
        self.telemetry = MemoryTelemetry(self.config.observability)
        self.context = ContextManager(self.config.context)
        self.namespaces = Namespaces(self.config.namespaces)
        self.storage_backend = build_storage_backend(self.config.storage.backend)
        self._v2 = V2Bridge(self)
        # module 13 (optional): cross-encoder re-ranking, lazy-loaded
        self.reranker = None
        self._rerank_enabled = self.config.rerank.enabled
        # sleep-time compute：全部子系统就绪后才启动维护线程（线程绝不能
        # 看到半初始化的引擎）；background=False 时是 no-op（零漂移）
        self._ensure_maintenance_thread()

    # ------------------------------------------------------------------ #
    # v2 sidecar files (module state lives outside the v1 snapshot so the
    # snapshot schema stays byte-compatible when modules are disabled)
    # ------------------------------------------------------------------ #
    def _sidecar_path(self, name: str) -> str:
        base = self.config.storage.path
        if base.lower().endswith(".gz"):
            base = base[:-3]
        if base.lower().endswith(".json"):
            base = base[:-5]
        return base + f".{name}.json"

    def _save_sidecars(self) -> None:
        """Persist enabled v2 module state to separate files."""
        if self.qapair.enabled and self.qapair.count():
            self._atomic_write(self._sidecar_path("qapairs"), self.qapair.to_dict())
        if self.factgraph.enabled and self.factgraph.entities:
            self._atomic_write(self._sidecar_path("factgraph"), self.factgraph.to_dict())
        if self.profile.enabled and (self.profile.profile_facts or self.profile.snapshots):
            self._atomic_write(self._sidecar_path("profile"), self.profile.to_dict())

    def _load_sidecars(self) -> None:
        import json as _json

        for name, target in (
            ("qapairs", self.qapair),
            ("factgraph", self.factgraph),
            ("profile", self.profile),
        ):
            path = self._sidecar_path(name)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    target.load_dict(_json.load(fh))
            except Exception:  # noqa: BLE001 - a broken sidecar never bricks loading
                pass

    @staticmethod
    def _detect_backend(path: str) -> str:
        """Sniff the snapshot file: sqlite magic header -> 'sqlite', else 'json'.

        A missing file stays 'json' so the json backend can return None and
        the caller keeps the "nothing to load" semantics.
        """
        try:
            with open(path, "rb") as fh:
                head = fh.read(16)
        except OSError:
            return "json"
        return "sqlite" if head.startswith(b"SQLite format 3\x00") else "json"

    @staticmethod
    def _atomic_write(path: str, payload: dict) -> None:
        import json as _json

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)

    def _sync_memory_region(self, memory_id: str, region_id: Optional[str]) -> None:
        """Keep Memory.region_id in sync with the spatial membership."""
        memory = self.memories.get(memory_id)
        if memory is not None:
            memory.region_id = region_id

    def _calibrate_region_threshold(self) -> None:
        """Adapt the region join threshold to the embedding provider.

        Deterministic hashing embeddings score 0.8+ within a topic but their
        cross-topic cosine can reach ~0.63, so a strict threshold (0.70)
        works there. Real embedding models (bge / OpenAI-compatible) produce
        lower within-topic cosine for short sentences (~0.55-0.9), so the
        threshold relaxes to 0.55 when the default was not overridden.
        """
        # only the untouched default is calibrated; an explicit 0.70 written
        # by the user (config file / Web config UI / code) is honored as-is
        if "region.min_join_cosine" in self.config._explicit_keys:
            return
        if self.config.region.min_join_cosine != 0.70:
            return  # a programmatic non-default value is respected too
        if self.embeddings.name in ("hashing",):
            return
        self.config.region.min_join_cosine = 0.55

    # ------------------------------------------------------------------ #
    # convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def memories(self) -> dict[str, Memory]:
        return self.memory_manager.memories

    # ------------------------------------------------------------------ #
    # writing
    # ------------------------------------------------------------------ #
    def add(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        importance: float = 0.5,
        source: str = "user",
        link_to: Optional[str] = None,
        link_kind: str = "reference",
        embedding: Optional[Any] = None,
        ns: Optional[str] = None,
    ) -> Memory:
        with self._lock:
            self._touch_write()
            metadata = dict(metadata or {})
            if ns is not None:
                metadata[NS_KEY] = ns
            if self._v2.active():
                memory = self._v2.add(
                    text=text, metadata=metadata, tags=tags, importance=importance,
                    source=source, link_to=link_to, link_kind=link_kind,
                    embedding=embedding,
                )
                if memory is None:
                    # nothing worth storing (e.g. bare question, w/o module 02):
                    # return a lightweight stub so callers can keep linking ids
                    memory = Memory(
                        text=text, metadata=dict(metadata or {}),
                        tags=list(tags or []), source=source,
                        embedding=embedding,
                    )
            else:
                memory = self.memory_manager.add_memory(
                    text=text,
                    metadata=metadata,
                    tags=tags,
                    importance=importance,
                    source=source,
                    link_to=link_to,
                    link_kind=link_kind,
                    embedding=embedding,
                )
            if memory.id in self.memories:
                # only real stores enter the WAL - a dropped utterance (v2
                # pipeline) must never resurrect after a crash
                self._record_write({
                    "op": "add",
                    "mid": memory.id,
                    "text": memory.text,
                    "source": memory.source,
                    "metadata": memory.metadata,
                    "tags": memory.tags,
                    "importance": memory.importance,
                })
                self.telemetry.record("add", memory_id=memory.id,
                                      text=memory.text[:60])
            else:
                self.telemetry.record("drop", text=memory.text[:60])
            self._autosave()
            return memory

    def add_many(self, texts: list[str], **kwargs) -> list[Memory]:
        with self._lock:
            self._touch_write()
            if self._v2.active():
                # pipeline mode: route each text individually (extraction is
                # per-message; the bulk optimization would bypass it)
                out = []
                for text in texts:
                    mem = self.add(text, **kwargs)
                    out.append(mem)
                return out
            self.space.set_bulk(True)
            try:
                memories = self.memory_manager.add_many(texts, **kwargs)
            finally:
                self.space.set_bulk(False)
            if self.config.region.auto_evolve and self.space.write_ops > 0:
                self.space.manager.evolution_pass(self.space)
            self._autosave()
            return memories

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def update(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        importance: Optional[float] = None,
        weight: Optional[float] = None,
        summary: Optional[str] = None,
    ) -> Memory:
        with self._lock:
            self._touch_write()
            memory = self.memory_manager.update_memory(
                memory_id, text, metadata, tags, importance, weight, summary
            )
            self._record_write({
                "op": "update", "mid": memory_id,
                "text": text, "metadata": metadata, "tags": tags,
                "importance": importance, "weight": weight, "summary": summary,
            })
            self._autosave()
            return memory

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            self._touch_write()
            ok = self.memory_manager.delete_memory(memory_id)
            if ok:
                self._record_write({"op": "delete", "mid": memory_id})
                self._autosave()
            return ok

    def get(self, memory_id: str) -> Optional[Memory]:
        return self.memory_manager.get(memory_id)

    def list_memories(
        self,
        tag: Optional[str] = None,
        source: Optional[str] = None,
        include_archived: bool = False,
    ) -> list[Memory]:
        return self.memory_manager.list_memories(
            tag=tag, source=source, include_archived=include_archived
        )

    # ------------------------------------------------------------------ #
    # archive
    # ------------------------------------------------------------------ #
    def archive(self, memory_id: str) -> bool:
        with self._lock:
            self._touch_write()
            ok = self.memory_manager.archive_memory(memory_id)
            if ok:
                self._record_write({"op": "archive", "mid": memory_id})
                self._autosave()
            return ok

    def restore(self, memory_id: str) -> bool:
        with self._lock:
            self._touch_write()
            ok = self.memory_manager.restore_memory(memory_id)
            if ok:
                self._record_write({"op": "restore", "mid": memory_id})
                self._autosave()
            return ok

    def archived_count(self) -> int:
        return self.archive_manager.cold_count()

    # ------------------------------------------------------------------ #
    # reinforcement / decay
    # ------------------------------------------------------------------ #
    def reinforce(self, memory_id: str) -> Optional[dict]:
        with self._lock:
            result = self.memory_manager.reinforce(memory_id)
            if result is not None:
                self.telemetry.record("reinforce", memory_id=memory_id)
            return result

    def apply_decay(self, reference: Optional[float] = None) -> int:
        with self._lock:
            return self.memory_manager.apply_decay_all(reference)

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: SearchQuery | str,
        top_k: int = 10,
        top_regions: Optional[int] = None,
        metadata_filters: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        region_retrieval: Optional[str] = None,
        graph_expand: Optional[int] = None,
        ns: Optional[str] = None,
        include_archived: bool = False,
    ) -> list[SearchHit]:
        if isinstance(query, str):
            query = SearchQuery(
                text=query,
                top_k=top_k,
                top_regions=top_regions,
                metadata_filters=dict(metadata_filters or {}),
                tags=tags,
                region_retrieval=region_retrieval,
                graph_expand=graph_expand or 0,
                include_archived=include_archived,
            )
        elif graph_expand is not None:
            query.graph_expand = graph_expand
        if include_archived:
            query.include_archived = True
        if ns is not None:
            query.metadata_filters = dict(query.metadata_filters or {})
            query.metadata_filters[NS_KEY] = ns
        with self._lock:
            self.decay.enabled = self.policy.decay_enabled
            t0 = now()
            direct = self._v2.search_pre(query)  # module 02: QA replay
            hits = self.retriever.search(self, query)
            if direct:
                hits = direct + hits
            hits = self._v2.search_post(query, hits)  # modules 03/04/05/06
            if self._rerank_enabled and hits:
                # module 13: optional cross-encoder precision pass
                from sme.retrieval.rerank import Reranker

                if self.reranker is None:
                    self.reranker = Reranker(self.config.rerank)
                hits = self.reranker.rerank(query.text, hits)
            self.telemetry.search(
                query.text, query.top_k, len(hits), (now() - t0) * 1000.0,
                [h.score for h in hits],
            )
            return hits

    def search_regions(self, text: str, top_k: int = 5) -> list:
        with self._lock:
            return self.retriever.search_regions(self, text, top_k)

    # ------------------------------------------------------------------ #
    # consolidation / compression
    # ------------------------------------------------------------------ #
    def consolidate(self) -> list[Memory]:
        with self._lock:
            created = self.consolidation.consolidate(self)
            self._autosave()
            return created

    def compress(self) -> list[Memory]:
        with self._lock:
            created = self.compression.compress(self)
            self._autosave()
            return created

    # ------------------------------------------------------------------ #
    # graph
    # ------------------------------------------------------------------ #
    def link(self, a: str, b: str, kind: str = "reference", weight: float = 1.0, note: str = "") -> bool:
        return self.memory_manager.link(a, b, kind, weight, note)

    def graph_edges(self) -> list:
        return list(self.graph.edges)

    def traverse_graph(self, start: str, kinds: Optional[list[str]] = None, mode: str = "bfs", max_depth: int = 3) -> list[str]:
        return self.graph.traverse(start, kinds, mode, max_depth)

    def build_neighbor_edges(self, k: int = 5, threshold: float = 0.55) -> int:
        vectors = {mid: m.embedding for mid, m in self.memories.items() if m.embedding is not None}
        added = self.graph.add_auto_neighbors(list(vectors), vectors, k=k, threshold=threshold)
        # keep the legacy Memory.neighbors field in sync with the graph
        for edge in self.graph.edges:
            if edge.kind == "neighbor":
                a = self.memories.get(edge.source)
                b = self.memories.get(edge.target)
                if a is not None:
                    a.neighbors.add(edge.target)
                if b is not None:
                    b.neighbors.add(edge.source)
        return added

    # ------------------------------------------------------------------ #
    # stats
    # ------------------------------------------------------------------ #
    def memory_stats(self) -> MemoryStats:
        return self.memory_manager.stats()

    def region_stats(self) -> RegionStats:
        return self.region_manager.region_stats(self.space)

    def region_history(self) -> list:
        return [
            {"kind": e.kind, "region_id": e.region_id, "detail": e.detail}
            for e in self.region_manager.history[-200:]
        ]

    def engine_stats(self) -> dict[str, Any]:
        stats = {
            "memories": self.memory_stats().to_dict(),
            "regions": self.region_stats().to_dict(),
            "splits": self.region_manager.split_count,
            "merges": self.region_manager.merge_count,
            "consolidations": self.consolidation.consolidation_count,
            "compressions": self.compression.compression_count,
            "graph_edges": len(self.graph),
            "provider": self.embeddings.model_name,
            "policy": self.policy.to_dict(),
        }
        # v2 module stats (absent when disabled => no v1 diff)
        if self.extraction.enabled:
            stats["extraction"] = self.extraction.to_dict()
        if self.qapair.enabled:
            stats["qapairs"] = self.qapair.count()
        if self.factgraph.enabled:
            stats["factgraph"] = self.factgraph.stats()
        if self.profile.enabled:
            stats["profile"] = self.profile.stats()
        if self.wal.enabled:
            stats["wal"] = self.wal.stats()
        # sleep-time compute：仅 background=True 时展示（关 = 无 v1 diff）
        if self.config.maintenance.background:
            stats["maintenance"] = {
                "enabled": True,
                "thread_alive": bool(
                    self._maint_thread and self._maint_thread.is_alive()
                ),
                "idle_after_s": self.config.maintenance.idle_after_s,
                "every_s": self.config.maintenance.every_s,
                "last_run_at": self._maint_last_run_at,
                "runs": self._maint_runs,
                "last_duration_s": self._maint_last_duration_s,
                "last_error": self._maint_last_error,
            }
        if self.telemetry.enabled:
            stats["telemetry"] = self.telemetry.summary()
        if self.namespaces.enabled:
            stats["namespaces"] = self.namespaces.stats()
        return stats

    # ------------------------------------------------------------------ #
    # visualization
    # ------------------------------------------------------------------ #
    def visualize(self, output_path: str = "memory_space.png", show_graph: bool = True) -> str:
        from sme.visualization import visualize as _viz

        return _viz(self, output_path, config=self.config.visualization, show_graph=show_graph)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Optional[str] = None) -> str:
        path = path or self.config.storage.path
        self.space.sync_geometry()  # snapshot carries exact region geometry
        # revision 标记随快照落盘：provider/model/dim/mrl_dim 的组合身份，
        # load 时与当前引擎对比，不一致即提示重建（SDK 直连软防护）
        self.config.embedding.revision_stamp = embedding_revision(self.config)
        # derive the backend live: engine.config.storage.backend may have
        # been changed after construction
        self.storage_backend = build_storage_backend(self.config.storage.backend)
        snapshot = EngineSnapshot(
            memories=list(self.memories.values()),
            regions=list(self.space.regions.values()),
            region_edges=list(self.space.region_edges.values()),
            memory_edges=list(self.graph.edges),
            counters={
                "splits": self.region_manager.split_count,
                "merges": self.region_manager.merge_count,
                "consolidations": self.consolidation.consolidation_count,
                "compressions": self.compression.compression_count,
            },
            config=self.config,
        )
        saved = self.storage_backend.save(
            path, snapshot, compress=self.config.storage.compress
        )
        self._save_sidecars()
        self._autosave_counter = 0
        self.wal.checkpointed += 1
        self.wal.reset()  # everything is in the snapshot now
        self.telemetry.record("save", path=path, memories=len(self.memories))
        logger.info("engine saved: %s (%d memories)", saved, len(self.memories))
        return saved

    def load(self, path: Optional[str] = None) -> bool:
        with self._lock:
            path = path or self.config.storage.path
            # cross-backend loading: sniff the file signature and switch the
            # backend before reading, so a sqlite snapshot can be loaded by a
            # json-configured engine (and vice versa) instead of failing with
            # a confusing JSON parse error on the binary file.
            detected = self._detect_backend(path)
            if detected != self.config.storage.backend:
                self.config.storage.backend = detected
                self.storage_backend = build_storage_backend(detected)
            snapshot = self.storage_backend.load(path)
            if snapshot is None:
                return False
            # 向量空间身份检查（SDK 直连软防护；REST PUT /config 已有硬闸门）：
            # 快照 revision 与当前引擎 revision 不一致且库非空 => 磁盘向量与
            # 当前 provider 不是同一空间（同维度换模型不报维度错，静默垃圾
            # 相似度），明确告警让用户跑官方重建工具
            if snapshot.memories and self._explicit_config:
                disk_revision = (
                    snapshot.config.embedding.revision_stamp
                    or embedding_revision(snapshot.config)
                )
                engine_revision = embedding_revision(self.config)
                if disk_revision != engine_revision:
                    logger.warning(
                        "embedding 向量空间不匹配：快照 revision=%s，当前引擎 "
                        "revision=%s，检索质量不可信，请运行 "
                        "python -m sme.rebuild_embeddings 重建向量",
                        disk_revision, engine_revision,
                    )
            self.config = snapshot.config
            # re-bind EVERY subsystem config (v1 + v2) to the snapshot config
            # so the restored engine behaves exactly like the saved one
            self.space.config = self.config.region
            self.region_manager.config = self.config.region
            self.policy.config = self.config.policy
            self.reinforcement.config = self.config.reinforcement
            self.decay.config = self.config.decay
            self.ranker.config = self.config.ranking
            self.retriever.config = self.config.retrieval
            self.consolidation.config = self.config.consolidation
            self.compression._config = self.config
            if self.compression._engine is not None:
                self.compression._engine.config = self.config.compression
            self.extraction.config = self.config.extraction
            self.factversion.config = self.config.factversion
            self.noise.config = self.config.noise
            self.qapair.config = self.config.qapair
            self.factgraph.config = self.config.factgraph
            self.factgraph_extractor.config = self.config.factgraph
            self.profile.config = self.config.profile
            self.wal.config = self.config.persistence
            # 回绑 WAL 捕获的 StorageConfig 引用：WAL 构造时持有的是旧 config.storage
            # 对象，快照可能带着不同的存储路径/后端；不同步的话 WAL 会继续写旧路径。
            # 先关旧句柄（文件/sqlite），下次 append 时按新路径重新打开。
            self.wal.close()
            self.wal._storage = self.config.storage
            self.telemetry.config = self.config.observability
            self.context.config = self.config.context
            self.namespaces.config = self.config.namespaces
            self.storage_backend = build_storage_backend(self.config.storage.backend)
            self._rerank_enabled = self.config.rerank.enabled
            if self.reranker is not None:
                self.reranker.config = self.config.rerank
            # rebuild the embedding provider / LLM client when the snapshot
            # was saved with a different provider (dim mismatch is fatal)
            if (
                self.embeddings.name != self.config.embedding.provider
                or self.embeddings.model_name != self.config.embedding.model
                or self.embeddings.dim != self.config.embedding.dim
                or self.embeddings.mrl_dim != self.config.embedding.mrl_dim
            ):
                self.embeddings = build_embedding_provider(self.config.embedding)
                self._calibrate_region_threshold()
                self.llm = LLMClient(self.config.llm)
                self.consolidation.llm = self.llm
                self.compression._llm = self.llm
                self.extraction.llm = self.llm
                self.factgraph_extractor.llm = self.llm
            else:
                # provider unchanged: keep the live client but re-point its
                # config (temperature/max_tokens/...) to the snapshot values
                self.llm.config = self.config.llm
                if self.llm.base_url != self.config.llm.base_url.rstrip("/"):
                    self.llm = LLMClient(self.config.llm)
                    self.consolidation.llm = self.llm
                    self.compression._llm = self.llm
                    self.extraction.llm = self.llm
                    self.factgraph_extractor.llm = self.llm
            self.memory_manager.load_state(
                snapshot.memories,
                {"edges": [e.to_dict() for e in snapshot.memory_edges]},
            )
            # rebuild the spatial index from the loaded memories; the space
            # dim follows the (possibly rebuilt) provider's effective dim so
            # MRL-truncated vectors load into geometry of the right size
            self.space.load_state(
                {"dim": self.embeddings.effective_dim, "write_ops": 0, "membership": {}, "vectors": {}},
                snapshot.regions,
                snapshot.region_edges,
            )
            # archived memories never re-enter the spatial space
            vectors: dict[str, np.ndarray] = {}
            for mid, memory in self.memories.items():
                if memory.archived:
                    memory.region_id = None
                elif memory.embedding is not None:
                    vectors[mid] = memory.embedding
            self.space.vectors = vectors
            # membership is rebuilt from the region snapshot (the regions'
            # member sets are authoritative; memory.region_id may be stale)
            vector_ids = set(vectors)
            membership: dict[str, Optional[str]] = {}
            pruned: list[str] = []
            to_rebuild = [
                region
                for region in self.space.regions.values()
                if region.member_ids & vector_ids
            ]
            # parallel exact geometry rebuild (numpy releases the GIL;
            # iteration 2.6 - 100k regions/memories load well under budget)
            from concurrent.futures import ThreadPoolExecutor

            if len(to_rebuild) > 8:
                with ThreadPoolExecutor(max_workers=min(8, len(to_rebuild))) as pool:
                    list(pool.map(
                        lambda r: r.update_geometry(vectors, self.space.dim),
                        to_rebuild,
                    ))
            else:
                for region in to_rebuild:
                    region.update_geometry(vectors, self.space.dim)
            for region in self.space.regions.values():
                region.member_ids &= vector_ids
                if region.member_ids:
                    for mid in region.member_ids:
                        membership[mid] = region.id
                else:
                    # collect first, delete after iteration (mutating the
                    # dict while iterating raises RuntimeError)
                    pruned.append(region.id)
            for rid in pruned:
                self.space.regions.pop(rid, None)
            self.space._membership = membership
            for mid, memory in self.memories.items():
                if not memory.archived:
                    memory.region_id = membership.get(mid)
            # drop region-graph edges that point to pruned regions
            for key, edge in list(self.space.region_edges.items()):
                if edge.source not in self.space.regions or edge.target not in self.space.regions:
                    self.space.region_edges.pop(key, None)
            self.space._centroids_dirty = True
            self.decay.enabled = self.policy.decay_enabled
            # re-sync the cold archive with the loaded snapshot: archived
            # memories live inside the snapshot, and the cold store must
            # point at the loaded path (a stale cold dict would corrupt
            # stats and could be flushed over another path's archive)
            cold_path = (
                os.path.join(
                    os.path.dirname(self.config.storage.path), "cold_archive.json"
                )
                if self.config.storage.path
                else None
            )
            self.archive_manager.cold_path = cold_path
            self.archive_manager.load_state(
                {m.id: m.to_dict() for m in snapshot.memories if m.archived}
            )
            self.region_manager.split_count = snapshot.counters.get("splits", 0)
            self.region_manager.merge_count = snapshot.counters.get("merges", 0)
            self.consolidation.consolidation_count = snapshot.counters.get("consolidations", 0)
            self.compression.compression_count = snapshot.counters.get("compressions", 0)
            self.retriever.refresh_keywords(self)
            # v2: sidecar module state + pending WAL replay (module 07)
            self._load_sidecars()
            if self.wal.enabled and os.path.exists(self.wal.path):
                self.wal.replay(self)
            # sleep-time compute：load 重绑快照配置后线程读新配置（间隔
            # 实时读）；快照翻开后台整理则（重）启线程，翻关由线程下跳自退。
            # load/replay 本身视为一次写活动：空闲计时从现在开始。
            self._touch_write()
            self._ensure_maintenance_thread()
            logger.info("engine loaded: %s (%d memories)", path, len(self.memories))
            return True

    def export_json(self, path: str) -> str:
        """Export all memories (with embeddings) as plain JSON."""
        # 全程持引擎锁：迭代 memories.values() 与并发写互斥，
        # 否则会 RuntimeError: dictionary changed size during iteration
        with self._lock:
            data = {
                "exported_at": now(),
                "memories": [m.to_dict() for m in self.memories.values()],
                "graph": self.graph.to_dict(),
                "stats": self.engine_stats(),
                # 演化/融合/压缩计数器：热重建迁移后回填新引擎用
                "counters": {
                    "splits": self.region_manager.split_count,
                    "merges": self.region_manager.merge_count,
                    "consolidations": self.consolidation.consolidation_count,
                    "compressions": self.compression.compression_count,
                },
            }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        return path

    def import_json(self, path: str) -> int:
        """Import memories from an export file (used by the API too).

        Also restores the memory graph edges (``export_json`` includes them);
        memory ids are preserved so the edges stay valid.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        count = self.import_memories(data.get("memories", []))
        graph_data = data.get("graph")
        if graph_data:
            self.graph.load_dict(graph_data)
        return count

    def import_documents(
        self,
        text_or_path: str,
        title: str = "",
        source: str = "document",
        summary_text: Optional[str] = None,
    ) -> list[Memory]:
        """Import a document as chunk memories + one summary parent.

        Knowledge-base scenario (iteration 1.3): chunks carry
        ``来源/文档名/条款号`` metadata and the summary memory links to its
        chunks via parent/children + summary edges. Retrieval hits the short
        summary vector; follow ``memory.children`` for the original text.
        """
        from sme.import_docs import import_documents as _import_documents

        with self._lock:
            self._touch_write()
            created = _import_documents(
                self, text_or_path, title=title, source=source,
                summary_text=summary_text,
            )
            if self.wal.enabled and created:
                # bulk import: checkpoint immediately (WAL does not carry the
                # parent/children summary structure)
                self.save()
            else:
                self._autosave()
            return created

    def import_memories(self, items: list[dict]) -> int:
        """Import memories from export-shaped dicts, losslessly.

        用 ``Memory.from_dict`` 完整还原全部字段（version/weight/hit_count/
        last_hit/created_at/summary/parent_id/children/archived...），导出携带
        的向量原样恢复、绝不重新编码。archived 记忆不进空间索引与检索索引，
        只登记进冷存档（与快照 load 的口径一致）——"导出-导入"无损。
        """
        with self._lock:
            self._touch_write()
            self.space.set_bulk(True)
            count = 0
            try:
                for item in items:
                    if not item.get("text"):
                        continue
                    memory = Memory.from_dict(item)
                    if memory.embedding is None:
                        # 旧格式导出 / 手写条目可能没有向量：才走当前 provider 兜底
                        memory.embedding = self.embeddings.embed_one(memory.text)
                    if memory.archived:
                        # 归档记忆：不进空间索引 / 不进检索索引，只登记冷存档
                        memory.region_id = None
                        self.memory_manager.memories[memory.id] = memory
                        self.archive_manager.archive(memory)
                    else:
                        # 活跃记忆：走 MemoryManager 的标准登记路径
                        # （空间插入 + memories 注册 + 检索索引 upsert）
                        self.memory_manager._register(memory)
                    count += 1
            finally:
                self.space.set_bulk(False)
            if self.config.region.auto_evolve and self.space.write_ops > 0:
                self.space.manager.evolution_pass(self.space)
            if self.wal.enabled:
                # bulk imports bypass the WAL; checkpoint right away so a
                # crash after the import never loses the batch
                self.save()
            else:
                self._autosave()
            return count

    # ------------------------------------------------------------------ #
    def _record_write(self, op: dict[str, Any]) -> None:
        """Module 07: append the write op to the WAL (no-op when disabled)."""
        if self.wal.enabled:
            self.wal.append(op)

    def _autosave(self) -> None:
        if not self.config.storage.autosave:
            return
        if self.wal.enabled:
            # WAL mode: checkpoint the full snapshot periodically only
            if self.wal.ops_since_checkpoint() >= self.config.persistence.checkpoint_every:
                self.save()
            return
        self._autosave_counter += 1
        if self._autosave_counter >= self.config.storage.autosave_interval:
            self.save()

    # ------------------------------------------------------------------ #
    # sleep-time compute: idle-period background maintenance
    # (空闲期整理，Letta/Anthropic 路线——维护挪出写路径，空闲后台做)
    # ------------------------------------------------------------------ #
    def _touch_write(self) -> None:
        """刷新写活动时间戳（空闲判定的基准）。

        所有会改库的引擎入口（add/update/delete/archive/restore/import）
        与 load/replay 都会调用；维护线程自己触发的 autosave 不算写活动，
        否则每轮整理都会把空闲窗口推开、饿死下一轮。
        """
        self._last_write_ts = now()

    @staticmethod
    def _maintenance_stop_finalizer(stop: threading.Event) -> None:
        """引擎被 GC / 热重建丢弃时停掉维护线程（weakref.finalize 回调）。"""
        stop.set()

    def _ensure_maintenance_thread(self) -> None:
        """（重新）启动空闲维护线程；background=False 时是 no-op。

        线程目标只持 ``weakref.ref(self)`` + 共享 stop Event——绝不持引擎
        强引用，否则线程活着引擎就永远不可达，GC 收尾永不触发（同 WAL
        flusher 的约束）。engine.load 重绑快照配置后再次调用：配置翻开则
        重启线程，翻关由线程下一跳读配置后自行退出。finalizer 每次重启
        前先 detach 旧的，避免重复触发。
        """
        if not self.config.maintenance.background:
            return
        with self._maint_lock:
            thread = self._maint_thread
            if thread is not None and thread.is_alive():
                return
            if self._maint_finalizer is not None:
                self._maint_finalizer.detach()
            self._maint_stop.clear()
            self._maint_finalizer = weakref.finalize(
                self,
                SpatialMemoryEngine._maintenance_stop_finalizer,
                self._maint_stop,
            )
            self._maint_thread = threading.Thread(
                target=SpatialMemoryEngine._maintenance_main,
                args=(weakref.ref(self), self._maint_stop),
                name="sme-maintenance",
                daemon=True,
            )
            self._maint_thread.start()

    @staticmethod
    def _maintenance_main(
        engine_ref: "weakref.ReferenceType[SpatialMemoryEngine]",
        stop: threading.Event,
    ) -> None:
        """空闲维护循环（daemon 线程）。

        每 every_s 整理一次（周期实时读配置：分片等待 ≤1s，热调 every_s /
        翻关 / 停线程都在下一分片生效，无需重启）。阻塞等待期间必须不持
        任何引擎强引用（局部 ``engine`` 读完配置即删）——否则等待中的局部
        变量会把引擎钉在内存里，GC 收尾永不触发。单轮异常只记日志不杀
        线程，按整周期退避后重试。
        """
        last_tick = time.monotonic()
        while not stop.is_set():
            engine = engine_ref()
            if engine is None:
                return
            try:
                cfg = engine.config.maintenance
                if not cfg.background:
                    return  # 配置翻关：线程退出（load 翻开由 _ensure 重启）
                due = last_tick + max(0.05, float(cfg.every_s))
                del engine, cfg  # 关键：阻塞等待期间允许引擎被 GC
                now_m = time.monotonic()
                if now_m < due:
                    if stop.wait(min(1.0, due - now_m)):
                        return
                    continue  # 醒来重读配置再算 due（热调生效）
                engine = engine_ref()
                if engine is None:
                    return
                engine._maintenance_tick()
                last_tick = time.monotonic()
                del engine
            except Exception:  # noqa: BLE001 - 维护线程不能被静默杀死
                logger.warning("空闲整理线程单轮失败（下一周期重试）", exc_info=True)
                last_tick = time.monotonic()  # 整周期退避，失败不刷屏
                time.sleep(0.5)

    def _maintenance_steps(self) -> dict[str, bool]:
        """各维护步骤是否使能；全 False 时本轮不进锁、不空转。"""
        active = sum(
            1 for m in self.memories.values()
            if not m.archived and m.source != "summary"
        )
        return {
            "decay": bool(self.policy.decay_enabled and self.decay.enabled),
            "consolidate": active >= self.config.consolidation.min_group_size,
            "compress": active >= self.config.compression.min_region_compact,
        }

    def _maintenance_tick(self) -> None:
        """一轮空闲整理：decay -> consolidate -> compress（锁内、逐步容错）。

        拿锁前做两道闸：写活动静默不足 idle_after_s 直接跳过；维护项全关
        不空转。拿锁后再确认一次（等锁期间可能有新写入 / 配置翻转）。
        每步独立 try/except：单步失败记日志与 last_error，不中断其他步骤。
        """
        cfg = self.config.maintenance
        if not cfg.background:
            return
        if now() - self._last_write_ts < cfg.idle_after_s:
            return  # 写活动尚未静默：本轮跳过，不推迟下一轮
        if not any(self._maintenance_steps().values()):
            return  # 维护项全关：不进锁不空转，也不计入 runs
        t0 = now()
        data: dict[str, Any] = {}
        errors: list[str] = []
        with self._lock:
            cfg = self.config.maintenance  # 等锁期间配置可能已被 load 重绑
            if (not cfg.background
                    or now() - self._last_write_ts < cfg.idle_after_s):
                return
            if self.policy.decay_enabled and self.decay.enabled:
                try:
                    data["decayed"] = self.apply_decay()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"decay: {exc}")
                    logger.warning("空闲整理 apply_decay 失败", exc_info=True)
            try:
                data["consolidated"] = len(self.consolidate())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"consolidate: {exc}")
                logger.warning("空闲整理 consolidate 失败", exc_info=True)
            try:
                data["compressed"] = len(self.compress())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"compress: {exc}")
                logger.warning("空闲整理 compress 失败", exc_info=True)
        self._maint_runs += 1
        self._maint_last_run_at = now()
        self._maint_last_duration_s = now() - t0
        self._maint_last_error = "; ".join(errors) if errors else None
        # telemetry.record 在 observability.enabled=False 时是 no-op（零开销）
        self.telemetry.record(
            "maintenance", runs=self._maint_runs,
            duration_s=round(self._maint_last_duration_s, 4),
            errors=len(errors), **data,
        )

    def __repr__(self) -> str:
        return (
            f"<SpatialMemoryEngine memories={len(self.memories)} "
            f"regions={len(self.space.regions)} provider={self.embeddings.model_name}>"
        )


class CompressionEngineProxy:
    """Thin proxy to keep imports cheap in the engine facade."""

    def __init__(self, config: SMEConfig, llm: LLMClient) -> None:
        self._config = config
        self._llm = llm
        self.compression_count = 0
        self._engine = None

    def _get(self):
        if self._engine is None:
            from sme.dynamics.compression import CompressionEngine

            self._engine = CompressionEngine(self._config.compression, self._llm)
        return self._engine

    def compress(self, engine: object) -> list:
        created = self._get().compress(engine)
        self.compression_count = self._get().compression_count
        return created
