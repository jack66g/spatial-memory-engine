"""共享嵌入器：所有向量基线使用同一个本地 BGE 模型（公平性核心）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # SME 插件根

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


class SharedEmbedder:
    """懒加载的 sentence-transformers 编码器（与 SME 本地 provider 同模型）。"""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        raw = self._get().encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        return [v.tolist() for v in raw]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


_shared: Any = None


def get_embedder(model: str = DEFAULT_MODEL) -> SharedEmbedder:
    global _shared
    if _shared is None or _shared.model != model:
        _shared = SharedEmbedder(model)
    return _shared
