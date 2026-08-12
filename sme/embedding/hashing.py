"""Deterministic hashing embedding provider.

Produces stable pseudo-embeddings purely from text (no network, no models)
using the classic "hashing trick" with character n-grams (fastText-style):

    - every substring of length 1..window of the lowercased text becomes a
      feature;
    - each feature is hashed (signed) into the embedding vector;
    - longer substrings get slightly more weight (they carry more semantics);

Texts sharing stems/substrings (e.g. "apple"/"apples", CJK characters) get
similar vectors, while unrelated texts are far apart. The embedding is fully
deterministic for a fixed seed, which keeps tests and benchmarks
reproducible offline.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from sme.embedding.base import EmbeddingProvider


def _hash_int(parts: str, digest_bits: int = 8) -> int:
    return int.from_bytes(
        hashlib.blake2b(parts.encode("utf-8"), digest_size=digest_bits).digest(),
        byteorder="little",
    )


class HashingEmbeddingProvider(EmbeddingProvider):
    name = "hashing"

    def __init__(
        self,
        dim: int = 64,
        factors: int = 2,
        window: int = 3,
        normalize_output: bool = True,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.factors = factors
        self.window = max(1, window)
        self.normalize_output = normalize_output
        self._seed = seed
        self.model_name = f"hashing-d{dim}-w{self.window}"

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        out = [self._embed_one(text) for text in texts]
        return self._post(out)

    def _embed_one(self, text: str) -> np.ndarray:
        lowered = (text or "").lower()
        if not lowered.strip():
            lowered = "_empty_"
        n = len(lowered)
        vector = np.zeros(self.dim, dtype=np.float64)
        for i in range(n):
            for w in range(1, self.window + 1):
                if i + w > n:
                    continue
                feat = lowered[i : i + w]
                for f in range(self.factors):
                    h = _hash_int(f"{self._seed}:{f}:{feat}")
                    idx = h % self.dim
                    sign = 1.0 if (h >> 8) & 1 else -1.0
                    vector[idx] += sign * (1.0 + 0.5 * w)
        # normalization is handled uniformly by _post() (respects
        # normalize_output, so normalize=False really disables it)
        return vector
