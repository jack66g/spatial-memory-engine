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
    """

    name: str = "base"
    dim: int = 0
    normalize_output: bool = True
    model_name: str = "base"

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed a batch of texts; returns one vector per text."""

    def embed_one(self, text: str) -> np.ndarray:
        vectors = self.embed([text])
        return vectors[0]

    # ------------------------------------------------------------------ #
    def _post(self, vectors: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for v in vectors:
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if self.dim and arr.shape[0] != self.dim:
                # pad/truncate to the configured dimension
                if arr.shape[0] < self.dim:
                    arr = np.pad(arr, (0, self.dim - arr.shape[0]))
                else:
                    arr = arr[: self.dim]
            if self.normalize_output:
                arr = normalize(arr)
            out.append(arr)
        return out
