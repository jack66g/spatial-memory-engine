"""Memory Decay.

Memories are NEVER deleted. Instead they decay: as time passes since the
last reinforcement, the effective weight, importance and retrieval
probability all decrease smoothly along an exponential forgetting curve.

Effective values used at retrieval time:

    factor(age)    = floor + (1 - floor) * exp(-ln2 * age / half_life)
    eff_weight     = weight      * factor
    eff_importance = importance  * factor

Because factor() >= floor > 0, a memory can always be retrieved again -
it just becomes harder and harder to hit. A single new hit fully restores
freshness (see reinforcement module).
"""

from __future__ import annotations

from sme.config import DecayConfig
from sme.models import Memory
from sme.utils import age_days, exponential_decay, now


class MemoryDecay:
    def __init__(self, config: DecayConfig, enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled

    def factor(self, memory: Memory, reference: float | None = None) -> float:
        """Current decay multiplier in [floor, 1] (weight floor)."""
        return self._factor(memory, reference, 1.0 - self.config.max_weight_decay)

    def _factor(self, memory: Memory, reference: float | None,
                floor: float) -> float:
        if not self.enabled:
            return 1.0
        age_d = age_days(memory.last_hit, reference)
        if age_d < 1e-6:
            return 1.0
        retention = exponential_decay(age_d, self.config.half_life_days)
        return floor + (1.0 - floor) * retention

    def effective_weight(self, memory: Memory, reference: float | None = None) -> float:
        return memory.weight * self.factor(memory, reference)

    def effective_importance(
        self, memory: Memory, reference: float | None = None
    ) -> float:
        return memory.importance * self._factor(
            memory, reference, 1.0 - self.config.max_importance_decay
        )

    def apply(self, memory: Memory, reference: float | None = None) -> float:
        """Persist the current decay factor onto the memory record.

        This makes the decay visible in exports and stats. The base weight
        and importance themselves are left untouched so reinforcement always
        builds on top of history.
        """
        ref = now() if reference is None else reference
        memory.decay_factor = self.factor(memory, ref)
        return memory.decay_factor
