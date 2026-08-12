"""SME 基线：按内置预设实例化引擎（聊天/知识库动/静/机器人/全关）。

预设的会话层语义在此复刻：reinforce_on → 命中强化；graph_expand →
检索图扩展；consolidate_every/compress_every → 周期融合/压缩（off 永不）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # SME 插件根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.base import BaseAdapter, SearchResult  # noqa: E402

OFF_PERIOD = 10 ** 9


def _build_engine(preset: str, workspace: str, model: str):
    from sme.config import SMEConfig
    from sme.config_items import PRESET_BY_KEY, apply_preset, defaults_config
    from sme.engine import SpatialMemoryEngine

    cfg_dict = defaults_config()
    apply_preset(cfg_dict, PRESET_BY_KEY[preset])
    cfg = SMEConfig.from_dict(
        {k: v for k, v in cfg_dict.items() if k != "_help"}
    )
    # 公平性：统一本地 BGE；状态文件放工作区
    cfg.embedding.provider = "sentence-transformers"
    cfg.embedding.model = model
    cfg.embedding.dim = 512
    # 公平性：LLM 提取与 mem0 等基线用同一个 DeepSeek（key 走环境变量）
    if cfg.extraction.enabled and cfg.extraction.mode == "llm":
        import os

        cfg.llm.base_url = "https://api.deepseek.com/v1"
        cfg.llm.api_key = os.environ.get("SME_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
        cfg.llm.model = "deepseek-v4-flash"
        cfg.llm.reasoning_effort = "none"
        cfg.llm.max_tokens = 600
        cfg.llm.temperature = 0.1
    cfg.storage.path = str(Path(workspace) / f"sme_{preset}.json")
    cfg.storage.autosave = False
    return SpatialMemoryEngine(cfg), cfg_dict


class SMEAdapter(BaseAdapter):
    def __init__(self, preset: str, workspace: str, embedder: Any = None,
                 model: str = "BAAI/bge-small-zh-v1.5") -> None:
        super().__init__(workspace, embedder)
        self.preset = preset
        self.name = f"sme_{preset}"
        self.engine, self.cfg = _build_engine(preset, workspace, model)
        self.reinforce_on = bool(self.cfg.get("memory.reinforce_on", True))
        self.graph_expand = int(self.cfg.get("memory.graph_expand", 0))
        self.consolidate_every = int(self.cfg.get("memory.consolidate_every", OFF_PERIOD))
        self.compress_every = int(self.cfg.get("memory.compress_every", OFF_PERIOD))
        self._writes = 0

    # ------------------------------------------------------------------ #
    def _store(self, text: str, role: str) -> None:
        if role == "assistant" and self.engine.extraction.enabled:
            # 与预设语义一致：提取模块开启时助手回答不入库
            if not self.engine.extraction.config.store_assistant:
                return
        self.engine.add(text, source=role)
        self._writes += 1
        # 周期融合/压缩（会话层语义；off=OFF_PERIOD 时永不触发）
        if self._writes and self._writes % max(1, self.consolidate_every) == 0:
            self.engine.consolidate()
        if self._writes and self._writes % max(1, self.compress_every) == 0:
            self.engine.compress()

    def _search(self, query: str, top_k: int) -> list[SearchResult]:
        from sme.retrieval import SearchQuery

        hits = self.engine.search(
            SearchQuery(text=query, top_k=top_k, graph_expand=self.graph_expand)
        )
        out = [SearchResult(text=h.memory.text, score=float(h.score))
               for h in hits]
        if out and self.reinforce_on:
            self.engine.reinforce(hits[0].memory.id)
        return out

    def stats(self) -> dict[str, Any]:
        s = super().stats()
        s["memories"] = len(self.engine.memories)
        s["regions"] = len(self.engine.space.regions)
        s["qapairs"] = self.engine.qapair.count()
        return s
