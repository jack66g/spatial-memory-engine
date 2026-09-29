"""BM25 选手：rank_bm25 + sme.utils.tokenize 中文 bigram（零向量对照组）。

- 只 import sme 的分词工具（铁律允许：import 不改）。
- 分数是 BM25 原始分（量纲与余弦不同，judge 只看文本）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from sme.utils import tokenize

from .base import Contestant


class Bm25Contestant(Contestant):
    name = "bm25_bigram"

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.reinit_cost_s = 0.0
        self.reset()

    def reset(self) -> None:
        import time

        t0 = time.time()
        self._corpus: list[str] = []
        self._toks: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self.reinit_cost_s = time.time() - t0

    def add(self, text: str, ns: Optional[str] = None) -> None:
        self._corpus.append(text)
        self._toks.append(tokenize(text, cjk_bigram=True))
        self._bm25 = None  # 惰性重建

    def _ensure(self) -> BM25Okapi:
        if self._bm25 is None:
            if not self._toks:
                raise RuntimeError("empty corpus")
            self._bm25 = BM25Okapi(self._toks)
        return self._bm25

    def search(self, q: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._corpus:
            return []
        scores = self._ensure().get_scores(tokenize(q, cjk_bigram=True))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self._corpus[i], float(scores[i])) for i in order if scores[i] > 0]

    def count(self) -> int:
        return len(self._corpus)


def make_bm25(workspace: Path) -> Contestant:
    return Bm25Contestant(workspace)
