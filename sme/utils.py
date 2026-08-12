"""Shared math and time utilities for the Spatial Memory Engine."""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger("sme")


# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #
def now() -> float:
    """Current unix timestamp in seconds."""
    return time.time()


def age_days(then: float, reference: float | None = None) -> float:
    """Age of an event in days, relative to `reference` (default: now)."""
    ref = time.time() if reference is None else reference
    return max(0.0, (ref - then) / 86400.0)


# --------------------------------------------------------------------------- #
# vector math
# --------------------------------------------------------------------------- #
def to_array(v: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def normalize(v: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = to_array(v)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return arr
    return arr / n


def cosine_similarity(
    a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray
) -> float:
    """Cosine similarity in [-1, 1]."""
    va, vb = to_array(a), to_array(b)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def euclidean_distance(
    a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray
) -> float:
    return float(np.linalg.norm(to_array(a) - to_array(b)))


def vector_mean(vectors: Iterable[Sequence[float] | np.ndarray]) -> np.ndarray:
    """Element-wise mean of vectors (zero vector if empty)."""
    arrs = [to_array(v) for v in vectors]
    if not arrs:
        return np.zeros(0, dtype=np.float64)
    return np.mean(np.stack(arrs), axis=0)


def bbox_of(vectors: Iterable[Sequence[float] | np.ndarray], dim: int):
    """Compute (min, max) axis-aligned bounding box of vectors.

    Returns arrays of size `dim`. With fewer than 2 vectors the bbox is
    expanded by a small epsilon so volume is never exactly zero.
    """
    arrs = [to_array(v) for v in vectors]
    if not arrs:
        return np.zeros(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64)
    mat = np.stack(arrs)
    bmin = np.min(mat, axis=0)
    bmax = np.max(mat, axis=0)
    span = bmax - bmin
    # expand degenerate axes
    tiny = 1e-6
    expand = np.where(span < tiny, tiny, 0.0)
    return bmin - expand, bmax + expand


# --------------------------------------------------------------------------- #
# Ebbinghaus-style retention
# --------------------------------------------------------------------------- #
def ebbinghaus_retention(
    age_d: float,
    half_life_days: float = 7.0,
    power: float = 1.0,
) -> float:
    """Retention probability following an Ebbinghaus-like forgetting curve.

    retention = 1 / (1 + (age / half_life)) ** power
    """
    if age_d <= 0.0:
        return 1.0
    return 1.0 / (1.0 + (age_d / max(half_life_days, 1e-9)) ** max(power, 0.1))


def exponential_decay(age_d: float, half_life_days: float = 30.0) -> float:
    """Exponential decay factor in [0, 1]."""
    if age_d <= 0.0:
        return 1.0
    return math.exp(-math.log(2.0) * age_d / max(half_life_days, 1e-9))


# --------------------------------------------------------------------------- #
# tokenization (lightweight, language-agnostic)
# --------------------------------------------------------------------------- #
_ALNUM_RUN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str, cjk_bigram: bool = False) -> list[str]:
    """Split text into lowercase alphanumeric tokens.

    Latin words stay whole; CJK runs become unigrams (``cjk_bigram=False``,
    the historical behavior) or 1-2 grams (``cjk_bigram=True``, used by the
    BM25 keyword channel - "违约金" -> 违/约/金/违约/约金, which cuts the
    single-char inverted-index noise for Chinese terms).
    """
    if not text:
        return []
    lowered = text.lower()
    result: list[str] = []
    has_cjk = _CJK_CHAR.search
    for tok in _ALNUM_RUN_RE.findall(lowered):
        if len(tok) > 1 and has_cjk(tok):
            result.extend(tok)
            if cjk_bigram:
                for i in range(len(tok) - 1):
                    result.append(tok[i : i + 2])
        else:
            result.append(tok)
    return result


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
