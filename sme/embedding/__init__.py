"""Embedding Engine: pluggable embedding providers with a unified interface.

Supported providers:
    - hashing               : deterministic offline pseudo-embeddings (tests/demo)
    - openai                : any OpenAI-compatible embedding API
                              (OpenAI, SiliconFlow, Jina, BGE via API, vLLM, ...)
    - sentence-transformers : local models (BGE, Nomic, MiniLM, ...)
    - fastembed             : local ONNX models via onnxruntime (int8 Qwen3
                              by default; pip install -e ".[onnx]")
"""

from sme.embedding.base import EmbeddingProvider
from sme.embedding.factory import build_embedding_provider

__all__ = [
    "EmbeddingProvider",
    "build_embedding_provider",
]
