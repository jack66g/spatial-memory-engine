"""mem0 基线（mem0 2.x）：市面主流记忆库（可选，未安装时自动跳过）。

公平性：
- embedder: huggingface provider + 与其它基线相同的本地 BGE 模型；
- llm: deepseek provider + 与 SME 相同的 deepseek-v4-flash（记忆提取）；
- API Key 只从环境变量读取，不落盘。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # SME 插件根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.base import BaseAdapter, SearchResult  # noqa: E402

try:
    from mem0 import Memory as Mem0Memory
except Exception:  # noqa: BLE001 - mem0 可选依赖
    Mem0Memory = None  # type: ignore


class Mem0Adapter(BaseAdapter):
    def __init__(self, workspace: str, embedder: Any = None,
                 model: str = "BAAI/bge-small-zh-v1.5",
                 llm_model: str = "deepseek-v4-flash") -> None:
        super().__init__(workspace, embedder)
        self.name = "mem0"
        if Mem0Memory is None:
            raise ImportError("mem0 未安装（pip install mem0ai）；跳过该基线")
        api_key = os.environ.get("SME_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
        config = {
            "embedder": {
                "provider": "huggingface",
                "config": {"model": model},
            },
            "llm": {
                "provider": "deepseek",
                "config": {"model": llm_model, "api_key": api_key},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "battle",
                    "path": str(Path(workspace) / "mem0_qdrant"),
                    "embedding_model_dims": 512,  # BGE-small-zh
                },
            },
        }
        self.mem0 = Mem0Memory.from_config(config)
        # 公平性：与 SME 提取对齐 —— 保持用户语言（中文），主语归一为"用户"
        try:
            self.mem0.custom_instructions = (
                "Extract memories in the SAME language as the user's message "
                "(Chinese). Write each memory as one short factual statement "
                "starting with the subject 用户 (e.g. 用户叫小林). Do not "
                "translate to English."
            )
        except Exception:  # noqa: BLE001
            pass
        self._memories: list[str] = []

    def store_raw(self, text: str) -> None:
        """知识库导入：infer=False 直存原文（绕过 LLM 提取，公平可完成）。"""
        text = text.strip()
        if not text:
            return
        self.mem0.add(text, user_id="xiaolin", infer=False)
        self._memories.append(text)
        self.store_count += 1

    def _store(self, text: str, role: str) -> None:
        if role != "user":
            return
        self.mem0.add(text, user_id="xiaolin")
        self._memories.append(text)

    def _search(self, query: str, top_k: int) -> list[SearchResult]:
        try:
            # mem0 2.x: user_id 需放入 filters；返回 {'results': [...]}
            raw = self.mem0.search(
                query, limit=top_k, filters={"user_id": "xiaolin"})
        except Exception:  # noqa: BLE001
            return []
        if isinstance(raw, dict):
            results = raw.get("results", [])
        else:
            results = raw or []
        out: list[SearchResult] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            memory = r.get("memory") or r.get("text") or ""
            if memory:
                out.append(SearchResult(text=memory, score=float(r.get("score", 0.0))))
        return out

    def stats(self) -> dict[str, Any]:
        s = super().stats()
        s["memories"] = len(self._memories)
        emb = getattr(self.mem0, "embedding_model", None)
        s["embedder"] = getattr(getattr(emb, "config", None), "model", None)
        s["llm"] = type(getattr(self.mem0, "llm", None)).__name__
        return s
