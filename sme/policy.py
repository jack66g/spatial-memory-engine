"""Memory Policy: the two global switches of the engine.

    1. full_memory  - True: keep and retrieve everything, never filter.
                      False: filter by importance (low-importance memories
                      become hard/impossible to retrieve but are never deleted).
    2. decay_enabled- True: weight/importance decay with time.
                      False: weights stay constant forever.
"""

from __future__ import annotations

from sme.config import PolicyConfig
from sme.models import Memory


class MemoryPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    @property
    def full_memory(self) -> bool:
        return self.config.full_memory

    @full_memory.setter
    def full_memory(self, value: bool) -> None:
        self.config.full_memory = bool(value)

    @property
    def decay_enabled(self) -> bool:
        return self.config.decay_enabled

    @decay_enabled.setter
    def decay_enabled(self, value: bool) -> None:
        self.config.decay_enabled = bool(value)

    # ------------------------------------------------------------------ #
    def allows_retrieval(self, memory: Memory) -> bool:
        """Whether a memory may participate in retrieval."""
        if memory.archived:
            return False
        if self.config.full_memory:
            return True
        return memory.importance >= self.config.importance_threshold

    def to_dict(self) -> dict:
        return {
            "full_memory": self.config.full_memory,
            "importance_threshold": self.config.importance_threshold,
            "decay_enabled": self.config.decay_enabled,
        }
