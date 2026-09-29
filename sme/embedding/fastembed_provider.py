"""Local ONNX embedding provider backed by Qdrant ``fastembed``.

Runs quantized (int8) ONNX models on CPU via onnxruntime — typically 2x+
faster than the torch fp32 SentenceTransformers baseline, with a small
quantization-only quality delta.

The default model is Qdrant's official ONNX export of Qwen3-Embedding-0.6B
(the ``-Q`` tag selects ``onnx/model_quantized.onnx``, int8 weights). Decoder
style embedding models such as Qwen3 MUST use last-token pooling; Qdrant's
export implements this correctly, so mean-pooled community exports (which
produce useless vectors) are avoided by construction.

fastembed L2-normalizes the pooled vector itself; the base class ``_post()``
re-normalizing an already-unit vector is an idempotent no-op, so behavior
stays consistent with the other providers (pad/truncate + normalize flag).

The dependency is imported lazily so the rest of SME works without it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from sme.embedding.base import EmbeddingProvider

#: int8 (quantized) variant of Qdrant's official Qwen3-Embedding-0.6B ONNX export.
DEFAULT_QWEN3_INT8_MODEL = "Qwen/Qwen3-Embedding-0.6B-Q"


class FastEmbedProvider(EmbeddingProvider):
    """ONNX embeddings via fastembed (onnxruntime, CPU int8 by default).

    ``model`` accepts any fastembed-supported model name, e.g.
    ``Qwen/Qwen3-Embedding-0.6B-Q`` (int8), ``Qwen/Qwen3-Embedding-0.6B``
    (fp32 ONNX), or ``BAAI/bge-small-zh-v1.5``.
    """

    name = "fastembed"

    def __init__(
        self,
        model: str = DEFAULT_QWEN3_INT8_MODEL,
        cache_dir: str | None = None,
        threads: int | None = None,
        batch_size: int = 32,
        normalize_output: bool = True,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "fastembed 未安装。请先安装：pip install fastembed "
                '（或 pip install -e ".[onnx]"；int8 量化模型还需 '
                "onnxruntime>=1.23，安装 fastembed 最新版即可自带）"
            ) from exc
        self._model = TextEmbedding(
            model_name=model,
            cache_dir=cache_dir,
            threads=threads,
        )
        self.model = model
        self.batch_size = batch_size
        self.normalize_output = normalize_output
        self.model_name = model
        self.dim = int(self._model.embedding_size)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        # fastembed's embed() is a lazy generator; list() forces evaluation so
        # errors (model loading, tokenize) surface here, not at consumption.
        raw = list(self._model.embed(list(texts), batch_size=self.batch_size))
        return self._post([np.asarray(v).reshape(-1) for v in raw])
