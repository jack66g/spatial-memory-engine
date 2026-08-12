"""OpenAI-compatible embedding provider.

Works with ANY service exposing the OpenAI ``/embeddings`` REST contract:

    POST {base_url}/embeddings
    {"model": ..., "input": [ ... ]}
    -> {"data": [{"embedding": [...]}], "model": "..."}

Supported out of the box: OpenAI, DeepSeek, Qwen (DashScope compatible mode),
SiliconFlow, OpenRouter (embeddings), Jina, BGE via API, vLLM, LM Studio,
Ollama (``/v1/embeddings``), etc. Just swap base_url / api_key / model.
"""

from __future__ import annotations

from typing import Sequence

import httpx
import numpy as np

from sme.embedding.base import EmbeddingProvider


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 0,
        batch_size: int = 32,
        timeout: float = 60.0,
        normalize_output: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.normalize_output = normalize_output
        self.extra_headers = extra_headers or {}
        self.model_name = model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        items = list(texts)
        with httpx.Client(timeout=self.timeout) as client:
            for start in range(0, len(items), self.batch_size):
                batch = items[start : start + self.batch_size]
                resp = client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers(),
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                ordered = sorted(
                    data["data"], key=lambda item: int(item.get("index", 0))
                )
                for item in ordered:
                    vectors.append(np.asarray(item["embedding"], dtype=np.float64))
        if self.dim == 0 and vectors:
            self.dim = vectors[0].shape[0]
        return self._post(vectors)
