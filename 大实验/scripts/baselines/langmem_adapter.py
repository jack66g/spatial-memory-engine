"""LangMem 基线：LLM 提取记忆 + 统一 BGE 检索。

langmem（LangChain 官方记忆库）负责记忆提取（DeepSeek），
检索后端与其它基线统一用本地 BGE（langmem 本身不提供向量检索，
统一检索后端才能对比"提取策略"差异；此点在报告中注明）。
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path
from typing import Any

if not hasattr(typing, "NotRequired"):  # Python 3.10 兼容
    from typing_extensions import NotRequired
    typing.NotRequired = NotRequired

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # SME 插件根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.base import BaseAdapter, SearchResult  # noqa: E402


class LangMemAdapter(BaseAdapter):
    def __init__(self, workspace: str, embedder: Any = None,
                 llm_model: str = "deepseek-v4-flash") -> None:
        super().__init__(workspace, embedder)
        self.name = "langmem"
        try:
            import os

            from langchain_openai import ChatOpenAI
            from langmem import create_memory_manager
            from langmem.knowledge.extraction import Memory
        except Exception as exc:  # noqa: BLE001
            raise ImportError(f"langmem 不可用（{exc}）") from exc
        api_key = os.environ.get("SME_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
        self._llm = ChatOpenAI(
            model=llm_model, api_key=api_key,
            base_url="https://api.deepseek.com/v1", temperature=0.1,
            max_tokens=600,
        )
        self.manager = create_memory_manager(
            self._llm, schemas=[Memory],
            instructions=(
                "用与用户消息相同的语言（中文）提取长期记忆，"
                "每条记忆是一句确定的事实陈述，主语归一为「用户」。"
                "忽略问候、闲聊与问题。宁缺毋滥。"
            ),
        )
        self._memories: list[str] = []
        self._vecs: list[list[float]] = []

    def store_raw(self, text: str) -> None:
        """知识库导入：直存原文（同 RAG，统一 BGE 嵌入）。"""
        text = text.strip()
        if not text:
            return
        self._memories.append(text)
        self._vecs.append(self.embedder.embed_one(text))
        self.store_count += 1

    def _store(self, text: str, role: str) -> None:
        if role != "user":
            return
        try:
            result = self.manager.invoke({"messages": [{"role": "user",
                                                        "content": text}]})
        except Exception:  # noqa: BLE001
            return
        for mem in getattr(result, "memories", result) or []:
            content = getattr(mem, "content", None) or getattr(mem, "text", None)
            if content and str(content).strip():
                content = str(content).strip()
                self._memories.append(content)
                self._vecs.append(self.embedder.embed_one(content))

    def _search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self._vecs:
            return []
        import numpy as np

        q = self.embedder.embed_one(query)
        mat = np.asarray(self._vecs, dtype=np.float64)
        scores = mat @ np.asarray(q)
        order = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(text=self._memories[int(i)], score=float(scores[int(i)]))
                for i in order]

    def stats(self) -> dict[str, Any]:
        s = super().stats()
        s["memories"] = len(self._memories)
        return s
