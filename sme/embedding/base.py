"""Unified embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from sme.utils import normalize


class EmbeddingProvider(ABC):
    """All embedding engines implement this interface.

    Providers embed one or many texts into fixed-dimension vectors. The
    vectors are L2-normalized by default (config.normalize), which makes
    cosine similarity equal to the dot product.

    MRL (Matryoshka Representation Learning, e.g. Qwen3-Embedding): when
    ``mrl_dim`` is set to a value strictly between 0 and the native dim,
    vectors are truncated to their first ``mrl_dim`` components and
    re-normalized — the effective output dimension becomes ``mrl_dim``
    (``effective_dim``), trading a little retrieval quality for storage.
    """

    name: str = "base"
    dim: int = 0
    mrl_dim: int = 0
    normalize_output: bool = True
    model_name: str = "base"

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed a batch of texts; returns one vector per text."""

    def embed_one(self, text: str) -> np.ndarray:
        vectors = self.embed([text])
        return vectors[0]

    @property
    def effective_dim(self) -> int:
        """Dimension actually emitted: mrl_dim when MRL truncation is active."""
        if 0 < self.mrl_dim < self.dim:
            return self.mrl_dim
        return self.dim

    # ------------------------------------------------------------------ #
    def _post(self, vectors: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        target = self.effective_dim
        for v in vectors:
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            # MRL truncation FIRST, normalization AFTER (先截再归一)：截断后
            # 前缀的 L2 范数必然 < 1，不重归一的话点积不再是余弦相似度
            if 0 < self.mrl_dim < arr.shape[0]:
                arr = arr[: self.mrl_dim]
            if target and arr.shape[0] != target:
                # pad/truncate to the effective dimension
                if arr.shape[0] < target:
                    arr = np.pad(arr, (0, target - arr.shape[0]))
                else:
                    arr = arr[: target]
            if self.normalize_output:
                arr = normalize(arr)
            out.append(arr)
        return out
