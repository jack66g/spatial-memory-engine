"""Smoke tests for the fastembed (ONNX int8) embedding provider.

The model-backed tests skip themselves unless fastembed is installed AND the
quantized Qwen3 model is already in the local fastembed cache — the suite
never triggers a ~1.1GB download.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sme.config import EmbeddingConfig
from sme.embedding.factory import build_embedding_provider
from sme.embedding.fastembed_provider import (
    DEFAULT_QWEN3_INT8_MODEL,
    FastEmbedProvider,
)


def _int8_model_cached() -> bool:
    """True when the quantized ONNX model is in fastembed's default cache."""
    cache = Path(tempfile.gettempdir()) / "fastembed_cache"
    return cache.exists() and any(
        p.name == "model_quantized.onnx" and p.stat().st_size > 500_000_000
        for p in cache.rglob("*.onnx")
    )


def test_defaults_reference_quantized_export():
    # the "-Q" tag must select onnx/model_quantized.onnx (int8), and the
    # provider must be reachable under its registered factory name
    assert FastEmbedProvider.name == "fastembed"
    assert DEFAULT_QWEN3_INT8_MODEL == "Qwen/Qwen3-Embedding-0.6B-Q"


def test_missing_dependency_hint(monkeypatch):
    # None in sys.modules makes "from fastembed import ..." raise ImportError
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ImportError) as excinfo:
        FastEmbedProvider()
    msg = str(excinfo.value)
    assert "fastembed" in msg
    assert "pip install" in msg


@pytest.mark.skipif(
    not _int8_model_cached(),
    reason="fastembed int8 model not in local cache (offline smoke)",
)
class TestFastEmbedProviderWithModel:
    """Model-backed smoke: dim, normalization, determinism, similarity."""

    def test_factory_builds_and_embeds(self, zh):
        provider = build_embedding_provider(
            EmbeddingConfig(provider="fastembed")  # model defaults to an
        )  # OpenAI name -> falls back to the int8 Qwen3 export
        assert isinstance(provider, FastEmbedProvider)
        assert provider.model_name == DEFAULT_QWEN3_INT8_MODEL
        assert provider.dim == 1024

        vectors = provider.embed([zh["likes_coffee"], zh["lives_beijing"], "hello"])
        assert len(vectors) == 3
        for v in vectors:
            assert isinstance(v, np.ndarray)
            assert v.shape == (1024,)
            assert np.isclose(float(np.linalg.norm(v)), 1.0, atol=1e-6)

    def test_embed_one_matches_embed(self, zh):
        provider = FastEmbedProvider()
        one = provider.embed_one(zh["q_name"])
        batch = provider.embed([zh["q_name"]])
        assert np.allclose(one, batch[0], atol=1e-9)

    def test_similarity_ordering(self, zh):
        provider = FastEmbedProvider()
        v = provider.embed(
            [zh["likes_coffee"], zh["lives_beijing"], zh["works_company"]]
        )
        sim = lambda a, b: float(np.dot(a, b))  # normalized: dot == cosine
        assert sim(v[0], v[1]) > sim(v[0], v[2]) or sim(v[0], v[1]) > 0.3

    def test_alias_onnx(self):
        provider = build_embedding_provider(EmbeddingConfig(provider="onnx"))
        assert isinstance(provider, FastEmbedProvider)
