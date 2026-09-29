"""Embedding provider factory: build a provider from configuration."""

from __future__ import annotations

from sme.config import EmbeddingConfig
from sme.embedding.base import EmbeddingProvider
from sme.embedding.fastembed_provider import (
    DEFAULT_QWEN3_INT8_MODEL,
    FastEmbedProvider,
)
from sme.embedding.hashing import HashingEmbeddingProvider
from sme.embedding.openai_compat import OpenAICompatibleEmbeddingProvider
from sme.embedding.sentence_transformers_provider import SentenceTransformersProvider


def build_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Instantiate the provider named by ``config.provider``.

    ``openai`` uses the OpenAI-compatible REST endpoint configured in
    ``config.base_url / api_key / model``. Any string that maps to
    SentenceTransformers loads a local torch model. ``fastembed`` (aliases
    ``onnx`` / ``onnx-int8``) loads a local ONNX model via fastembed /
    onnxruntime — int8 CPU inference, markedly faster than torch fp32; the
    model defaults to Qdrant's quantized Qwen3-Embedding-0.6B export unless
    ``config.model`` names another fastembed-supported model.
    """
    key = (config.provider or "hashing").strip().lower()
    if key in ("openai", "openai-compatible"):
        return OpenAICompatibleEmbeddingProvider(
            base_url=config.base_url or "https://api.openai.com/v1",
            api_key=config.api_key,
            model=config.model,
            dim=config.dim,
            batch_size=config.batch_size,
            normalize_output=config.normalize,
        )
    if key in (
        "sentence-transformers",
        "sentence_transformers",
        "sentence_transformer",
        "local",
        "bge",
        "nomic",
    ):
        model = config.model
        if key == "bge" and model.startswith("text-embedding"):
            model = "BAAI/bge-small-zh-v1.5"
        return SentenceTransformersProvider(
            model=model,
            normalize_output=config.normalize,
        )
    if key in ("fastembed", "onnx", "onnx-int8"):
        model = config.model
        # the global default model name is an OpenAI API model; fastembed
        # can't load it — fall back to the quantized Qwen3 export instead
        if not model or model.startswith("text-embedding"):
            model = DEFAULT_QWEN3_INT8_MODEL
        return FastEmbedProvider(
            model=model,
            batch_size=config.batch_size,
            normalize_output=config.normalize,
        )
    # default & explicit hashing
    return HashingEmbeddingProvider(
        dim=config.dim,
        factors=config.hash_factors,
        window=config.hash_window,
        normalize_output=config.normalize,
    )
