"""裸 RAG 基线：同 BGE 向量库 + 余弦 top-k，无任何记忆动力学。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # SME 插件根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.base import BaseAdapter, SearchResult  # noqa: E402


class RagAdapter(BaseAdapter):
    def __init__(self, workspace: str, embedder: Any = None) -> None:
        super().__init__(workspace, embedder)
        self.name = "rag"
        self._texts: list[str] = []
        self._vecs: list[list[float]] = []

    def _store(self, text: str, role: str) -> None:
        if role != "user":
            return  # 裸 RAG 只存用户消息
        self._texts.append(text)
        self._vecs.append(self.embedder.embed_one(text))

    def _search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self._vecs:
            return []
        q = self.embedder.embed_one(query)
        import numpy as np

        mat = np.asarray(self._vecs, dtype=np.float64)
        scores = mat @ np.asarray(q)
        order = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(text=self._texts[int(i)], score=float(scores[int(i)]))
                for i in order]

    def stats(self) -> dict[str, Any]:
        s = super().stats()
        s["memories"] = len(self._texts)
        return s
