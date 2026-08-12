"""基线统一接口。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    text: str
    score: float


class BaseAdapter:
    """所有记忆基线实现同一接口，保证跑批公平。"""

    name: str = "base"
    needs_llm: bool = False

    def __init__(self, workspace: str, embedder: Any = None) -> None:
        self.workspace = workspace
        self.embedder = embedder  # 共享的 BGE 嵌入器（公平性）
        self.store_ms: list[float] = []
        self.search_ms: list[float] = []
        self.store_count = 0

    # ------------------------------------------------------------------ #
    def store(self, text: str, role: str = "user") -> None:
        """存储一轮消息（user 或 assistant）。"""
        t0 = time.perf_counter()
        self._store(text, role)
        self.store_ms.append((time.perf_counter() - t0) * 1000.0)
        self.store_count += 1

    def store_raw(self, text: str) -> None:
        """直存原文（知识库导入用，绕过 LLM 提取——公平且可完成）。"""
        self.store(text, role="user")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        t0 = time.perf_counter()
        hits = self._search(query, top_k)
        self.search_ms.append((time.perf_counter() - t0) * 1000.0)
        return hits

    def on_answer(self, query: str, top_hit: SearchResult | None) -> None:
        """检索命中后的强化钩子（各基线按自身设计决定是否实现）。"""

    def stats(self) -> dict[str, Any]:
        def avg(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 2) if xs else 0.0

        return {
            "store_count": self.store_count,
            "store_avg_ms": avg(self.store_ms),
            "search_avg_ms": avg(self.search_ms),
        }

    # 子类实现 ----------------------------------------------------------- #
    def _store(self, text: str, role: str) -> None:
        raise NotImplementedError

    def _search(self, query: str, top_k: int) -> list[SearchResult]:
        raise NotImplementedError
