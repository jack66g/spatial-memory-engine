"""裸 RAG 选手：Qwen3 向量 + 余弦 top-k（无任何记忆管理，对照组）。

- add 原文直存；search 全量点积取 top_k（向量已归一化，点积=余弦）。
- 用 common.embed_texts 的进程级单例（与 graphiti 共享同一个 Qwen3 实例）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Contestant


class RagContestant(Contestant):
    name = "rag_qwen3"

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.reinit_cost_s = 0.0
        self.reset()

    def reset(self) -> None:
        from .common import get_embedder

        t0 = time.time()
        get_embedder()  # 预热（单例，首次计 reinit 成本）
        self.reinit_cost_s = time.time() - t0
        self._texts: list[str] = []
        self._mat = np.zeros((0, 0), dtype=np.float32)

    def add(self, text: str, ns: Optional[str] = None) -> None:
        from .common import embed_texts

        vec = np.asarray(embed_texts([text])[0], dtype=np.float32)
        if self._mat.size == 0:
            self._mat = vec.reshape(1, -1)
        else:
            self._mat = np.vstack([self._mat, vec])
        self._texts.append(text)

    def search(self, q: str, top_k: int = 5) -> list[tuple[str, float]]:
        from .common import embed_texts

        if not self._texts:
            return []
        qv = np.asarray(embed_texts([q])[0], dtype=np.float32)
        scores = self._mat @ qv
        order = np.argsort(-scores)[:top_k]
        return [(self._texts[i], float(scores[i])) for i in order]

    def count(self) -> int:
        return len(self._texts)


def make_rag(workspace: Path) -> Contestant:
    return RagContestant(workspace)
