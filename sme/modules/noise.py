"""Module 06 - NoiseControl: retrieval-side noise suppression.

Suppresses three kinds of noise that crowd out real memories:

* duplication  - the same sentence stored many times (dataset template
                 sentences in the v1 stress test)
* templating   - short generic sentences whose n-gram fingerprints cover
                 many other memories
* low density  - sentences with almost no information payload (no digits,
                 no entities, little vocabulary)

``apply(hits)`` re-ranks a search result list, multiplying each hit by a
noise factor. Disabled => the ranking is returned untouched (v1 behavior).
"""

from __future__ import annotations

from typing import Any

from sme.config import NoiseConfig
from sme.utils import tokenize


class NoiseScorer:
    def __init__(self, config: NoiseConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------ #
    def _duplicate_count(self, engine: Any, text: str) -> int:
        """How many times this exact text appears among recent memories."""
        n = 0
        for mid, mem in list(engine.memories.items())[-self.config.dup_window :]:
            if mem.archived:
                continue
            if mem.text == text:
                n += 1
        return max(n - 1, 0)

    def _template_degree(self, engine: Any, text: str) -> float:
        """n-gram coverage of the memory corpus (0..1)."""
        toks = tokenize(text)
        if len(toks) < self.config.ngram_window:
            return 0.0
        n = self.config.ngram_window
        grams = {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}
        if not grams:
            return 0.0
        covered = 0
        corpus = [m.text for m in list(engine.memories.values())[-self.config.dup_window :]]
        for gram in grams:
            target = "".join(gram)
            if any(target in c for c in corpus):
                covered += 1
        return covered / len(grams)

    def _info_density(self, text: str) -> float:
        """Payload ratio: digits + rare tokens / total characters."""
        if not text:
            return 0.0
        digits = sum(1 for ch in text if ch.isdigit())
        toks = tokenize(text)
        if not toks:
            return 0.0
        # CJK unigrams repeat a lot; density = (unique tokens + digits) / len
        unique = len(set(toks))
        return (unique + digits) / max(len(text), 1)

    # ------------------------------------------------------------------ #
    def factor(self, engine: Any, memory: Any) -> float:
        """Noise multiplier in (0, 1] for one memory."""
        if not self.enabled:
            return 1.0
        cfg = self.config
        if memory.metadata.get("fact_kind") == "fact":
            return 1.0  # extracted facts are clean by construction
        text = memory.text
        f = 1.0
        dup = self._duplicate_count(engine, text)
        if dup >= 1:
            f *= max(0.05, 1.0 - cfg.dup_penalty * min(dup, 5) / 5.0)
        tmpl = self._template_degree(engine, text)
        if tmpl >= 0.5:
            f *= max(0.1, 1.0 - cfg.template_penalty * tmpl)
        density = self._info_density(text)
        if density < cfg.min_density:
            f *= max(0.15, 1.0 - cfg.template_penalty)
        return f

    def apply(self, hits: list[Any], engine: Any) -> list[Any]:
        """Re-rank hits with the noise factor applied to each score."""
        if not self.enabled or not hits:
            return hits
        out = []
        for hit in hits:
            f = self.factor(engine, hit.memory)
            if f < 1.0:
                hit.score *= f
                if hit.breakdown is not None:
                    hit.breakdown.final = round(hit.score, 4)
            out.append(hit)
        out.sort(key=lambda h: h.score, reverse=True)
        return out
