"""Module 04 - UserProfile: per-user profile facts and snapshots.

Aggregates profile-grade facts per user and periodically emits a compact
profile *snapshot* (ChatGPT-memory style). During retrieval, profile facts
receive a boost so "总结一下我" surfaces the profile instead of template
noise.

Disabled => no-op; no profile facts, no snapshot, no ranking boost.
"""

from __future__ import annotations

from typing import Any

from sme.config import ProfileConfig
from sme.models import Fact
from sme.utils import now

PROFILE_TAG = "profile_fact"
SNAPSHOT_TAG = "profile_snapshot"

# profile-grade fact kinds (keywords that indicate stable personal info)
_PROFILE_HINTS = [
    "喜欢", "爱好", "工作", "职业", "年龄", "生日", "名字", "姓名", "叫",
    "住在", "住址", "家在", "老家", "公司", "同事", "朋友", "家人", "父母",
    "学历", "专业", "毕业", "习惯", "日常", "每周", "每天", "经常", "不喜",
    "讨厌", "最", "计划", "目标", "养了", "养猫", "养狗", "有孩子", "结婚了",
    "单身", "对象", "妻子", "丈夫", "孩子", "血型", "星座", "身高", "体重",
]


class UserProfile:
    def __init__(self, config: ProfileConfig) -> None:
        self.config = config
        self.profile_facts: list[str] = []      # memory ids tagged as profile facts
        self.snapshots: dict[str, dict] = {}    # user_id -> {ts, facts}
        self._write_count = 0
        self._engine: Any = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def bind(self, engine: Any) -> None:
        """Bind the owning engine (for snapshot/query access to memories)."""
        self._engine = engine

    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_profile_grade(fact: Fact) -> bool:
        if fact.kind in ("correction",):
            return True
        return any(h in fact.text for h in _PROFILE_HINTS)

    def upsert(self, memory: Any) -> bool:
        """Register a stored memory as a profile fact (if grade qualifies)."""
        if not self.enabled:
            return False
        fact = memory.metadata.get("fact_kind") == "fact"
        if not fact:
            return False
        if self._is_profile_grade(
            Fact(text=memory.text, kind=memory.metadata.get("fact_kind", "fact"))
        ):
            if memory.id not in self.profile_facts:
                self.profile_facts.append(memory.id)
            memory.tags = list(set(memory.tags + [PROFILE_TAG]))
            self._write_count += 1
            if self._write_count >= self.config.snapshot_every:
                self.snapshot("default")
                self._write_count = 0
            return True
        return False

    # ------------------------------------------------------------------ #
    def snapshot(self, user_id: str = "default") -> dict:
        """Emit a profile snapshot: the top-N profile facts for one user."""
        top = self.config.top_facts
        mems = self.profile_memories({})
        facts = [m.text for m in mems if m.source != "summary"][:top]
        snap = {"ts": now(), "facts": facts, "count": len(facts)}
        self.snapshots[user_id] = snap
        return snap

    def profile_memories(self, engine_memories: dict) -> list:
        """Current profile memories (ordered by importance)."""
        if self._engine is not None and not engine_memories:
            engine_memories = self._engine.memories
        out = []
        for mid in self.profile_facts:
            mem = engine_memories.get(mid)
            if mem is not None and not mem.archived:
                out.append(mem)
        out.sort(key=lambda m: m.importance, reverse=True)
        return out

    # ------------------------------------------------------------------ #
    def boost(self, hits: list[Any]) -> list[Any]:
        """Boost profile facts during retrieval (module 04 ranking signal)."""
        if not self.enabled or not hits:
            return hits
        weight = self.config.boost_weight
        for hit in hits:
            if PROFILE_TAG in hit.memory.tags:
                hit.score *= 1.0 + weight
                if hit.breakdown is not None:
                    hit.breakdown.final = round(hit.score, 4)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def stats(self) -> dict[str, Any]:
        return {
            "profile_facts": len(self.profile_facts),
            "snapshots": {k: v["count"] for k, v in self.snapshots.items()},
        }

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_facts": list(self.profile_facts),
            "snapshots": {
                k: {"ts": v["ts"], "facts": v["facts"], "count": v["count"]}
                for k, v in self.snapshots.items()
            },
            "write_count": self._write_count,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self.profile_facts = list(data.get("profile_facts", []))
        self.snapshots = {
            k: {"ts": v.get("ts", now()), "facts": v.get("facts", []),
                "count": v.get("count", 0)}
            for k, v in data.get("snapshots", {}).items()
        }
        self._write_count = int(data.get("write_count", 0))
