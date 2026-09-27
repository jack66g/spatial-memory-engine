"""Memory Ranking.

The final score of a memory is a weighted combination of many signals:

    semantic    - cosine similarity to the query (from the embedding space)
    importance  - declared/learned importance of the memory
    freshness   - how recently the memory was created or updated
    weight      - reinforcement weight (hits build it up)
    decay       - time decay factor (Ebbinghaus/exponential forgetting)
    hit_count   - total historical hits (long-term reinforcement)
    recency     - time since the last hit
    region      - the score of the region the memory belongs to

    FinalScore = sum(weight_i * signal_i)   (weights sum to 1)

The ranking is fully extensible: custom scorers can be registered.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from sme.config import RankingConfig
from sme.models import Memory, ScoreBreakdown
from sme.utils import age_days, clamp, cosine_similarity, now

Scorer = Callable[[Memory, float, float, object], float]


class _RankingEngine:
    """Duck-typed engine reference (avoids circular imports)."""

    space: object
    decay: object
    policy: object


class MemoryRanker:
    def __init__(self, config: RankingConfig) -> None:
        self.config = config
        self._custom_scorers: dict[str, Scorer] = {}

    # ------------------------------------------------------------------ #
    def register_scorer(self, name: str, scorer: Scorer) -> None:
        """Register an extra normalized [0,1] scorer for extensibility."""
        self._custom_scorers[name] = scorer

    def weights(self) -> dict[str, float]:
        return {
            "semantic": self.config.semantic,
            "importance": self.config.importance,
            "freshness": self.config.freshness,
            "weight": self.config.weight,
            "decay": self.config.decay,
            "hit_count": self.config.hit_count,
            "recency": self.config.recency,
            "region": self.config.region,
        }

    # ------------------------------------------------------------------ #
    def score(
        self,
        memory: Memory,
        query_vector: np.ndarray,
        region_score: float,
        engine: object,
        reference: float | None = None,
        semantic: float | None = None,
        detailed: bool = True,
    ) -> tuple[float, ScoreBreakdown]:
        """Compute the final score (and optionally its breakdown).

        ``semantic`` may be pre-computed by the caller (batch vectorized).
        With ``detailed=False`` the per-signal rounding/clamping is skipped
        so thousands of candidates can be scored in a fast pass; the
        breakdown is then computed only for the final top-k.
        """
        ref = now() if reference is None else reference
        cfg = self.config

        if semantic is None:
            semantic = 0.0
            if memory.embedding is not None:
                semantic = (
                    cosine_similarity(query_vector, memory.embedding) + 1.0
                ) / 2.0

        importance = engine.decay.effective_importance(memory, ref)
        # freshness decays from the last touch/hit along the decay
        # half-life: recently touched memories are fresher (the stored
        # ``Memory.freshness`` field is legacy and always 1.0)
        half_life = getattr(engine.decay.config, "half_life_days", 30.0)
        freshness = memory.freshness / (
            1.0 + age_days(memory.last_hit, ref) / max(half_life, 1e-9)
        )
        eff_weight = engine.decay.effective_weight(memory, ref)
        weight = eff_weight / (1.0 + eff_weight)
        decay = engine.decay.factor(memory, ref)
        hit_count = memory.hit_count / (1.0 + memory.hit_count)
        recency = 1.0 / (1.0 + age_days(memory.last_hit, ref) / 7.0)
        region = region_score

        final = (
            cfg.semantic * semantic
            + cfg.importance * importance
            + cfg.freshness * freshness
            + cfg.weight * weight
            + cfg.decay * decay
            + cfg.hit_count * hit_count
            + cfg.recency * recency
            + cfg.region * region
        )
        for name, scorer in self._custom_scorers.items():
            final += 0.1 * clamp(scorer(memory, semantic, region_score, engine))

        if not detailed:
            return float(final), ScoreBreakdown(final=round(final, 4))

        breakdown = ScoreBreakdown(
            semantic=round(clamp(semantic), 4),
            importance=round(clamp(importance), 4),
            freshness=round(clamp(freshness), 4),
            weight=round(clamp(weight), 4),
            decay=round(clamp(decay), 4),
            hit_count=round(clamp(hit_count), 4),
            recency=round(clamp(recency), 4),
            region=round(clamp(region), 4),
            final=round(final, 4),
        )
        return final, breakdown
