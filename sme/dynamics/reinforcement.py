"""Memory Reinforcement (Ebbinghaus-style).

Every time a memory is hit (retrieved and used), it is reinforced:

    hit_count += 1
    weight      += delta * retention(age)     (older memories gain less)
    importance  += delta * retention(age)
    last_hit / freshness update

The retention follows an Ebbinghaus forgetting curve:
    retention = 1 / (1 + (age / half_life)) ** power

This gives long-term reinforcement: memories that are used repeatedly
(especially used recently) become progressively stronger and easier to
retrieve, while rarely-used memories fade.
"""

from __future__ import annotations

from sme.config import ReinforcementConfig
from sme.models import Memory
from sme.utils import age_days, clamp, ebbinghaus_retention, now


class EbbinghausReinforcement:
    def __init__(self, config: ReinforcementConfig) -> None:
        self.config = config

    def retention(self, memory: Memory, reference: float | None = None) -> float:
        """Ebbinghaus retention for the memory given its last-hit age."""
        age_d = age_days(memory.last_hit, reference)
        return ebbinghaus_retention(
            age_d,
            half_life_days=self.config.ebbinghaus_half_life_days,
            power=self.config.ebbinghaus_power,
        )

    def on_hit(self, memory: Memory, reference: float | None = None) -> dict:
        """Apply reinforcement to a memory hit. Returns the delta summary."""
        ref = now() if reference is None else reference
        retention = self.retention(memory, ref)
        weight_delta = self.config.hit_weight_delta * retention
        importance_delta = self.config.hit_importance_delta * retention

        memory.hit_count += 1
        memory.weight = clamp(
            memory.weight + weight_delta,
            0.0,
            self.config.max_weight,
        )
        memory.importance = clamp(
            memory.importance + importance_delta,
            self.config.min_importance,
            self.config.max_importance,
        )
        memory.last_hit = ref
        memory.freshness = 1.0
        memory.version += 1

        return {
            "memory_id": memory.id,
            "retention": round(retention, 4),
            "weight_delta": round(weight_delta, 4),
            "importance_delta": round(importance_delta, 4),
            "hit_count": memory.hit_count,
        }
