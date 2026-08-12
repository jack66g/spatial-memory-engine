"""BM25 全文检索基线：纯关键词倒排（借用 sme 的 BM25Index 作为库）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # SME 插件根
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.base import BaseAdapter, SearchResult  # noqa: E402


class BM25Adapter(BaseAdapter):
    def __init__(self, workspace: str, embedder: Any = None) -> None:
        super().__init__(workspace, embedder)
        self.name = "bm25"
        from sme.retrieval.retriever import BM25Index

        self._index = BM25Index(cjk_bigram=True)
        self._texts: list[str] = []

    def _store(self, text: str, role: str) -> None:
        if role != "user":
            return
        self._texts.append(text)
        self._index.add_document(f"m{len(self._texts) - 1}", text)

    def _search(self, query: str, top_k: int) -> list[SearchResult]:
        from sme.utils import tokenize

        scores = self._index.scores(tokenize(query, cjk_bigram=True),
                                    [f"m{i}" for i in range(len(self._texts))])
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [SearchResult(text=self._texts[int(k[1:])], score=float(v))
                for k, v in ranked if v > 0]

    def stats(self) -> dict[str, Any]:
        s = super().stats()
        s["memories"] = len(self._texts)
        return s
