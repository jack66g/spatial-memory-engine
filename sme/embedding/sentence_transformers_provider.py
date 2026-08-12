"""Local SentenceTransformers embedding provider.

Loads any model from HuggingFace hub or a local path, e.g.:
    BAAI/bge-small-zh-v1.5, BAAI/bge-m3, nomic-ai/nomic-embed-text-v1.5,
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, ...

The dependency is imported lazily so the rest of SME works without it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from sme.embedding.base import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    name = "sentence-transformers"

    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize_output: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "sentence-transformers is not installed. "
                "pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model, device=device)
        self.model = model
        self.batch_size = batch_size
        self.normalize_output = normalize_output
        self.model_name = model
        getter = getattr(
            self._model, "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self.dim = int(getter())

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        raw = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_output,
            convert_to_numpy=True,
        )
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        return [np.asarray(v, dtype=np.float64).reshape(-1) for v in raw]
