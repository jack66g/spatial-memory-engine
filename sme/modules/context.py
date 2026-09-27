"""Module 11 - ContextManager: layered context management.

Letta/MemGPT-style two layers:

* in-context  - always-injected: user profile facts (reserve_profile tokens)
                + the most recent dialogue rounds that fit the budget
* external    - on-demand retrieval: relevant memories (already produced by
                the engine's search pipeline)

``build()`` assembles the final LLM message list under a token budget.
Disabled => the caller (ai/chat.py) keeps the v1 prompt assembly unchanged.
"""

from __future__ import annotations

import re
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK ~1 token/char, latin ~1 token/4 chars."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
    other = len(text) - cjk
    return cjk + max(1, int(other / 4))


class ContextManager:
    def __init__(self, config: Any) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------ #
    def _profile_block(self, engine: Any) -> str:
        """In-context profile facts (module 04) or nothing.

        ``reserve_profile`` is a TOKEN budget: profile facts are included in
        importance order until the budget is exhausted (no arbitrary
        fact-count math).
        """
        profile = getattr(engine, "profile", None)
        if profile is None or not profile.enabled:
            return ""
        budget = int(getattr(self.config, "reserve_profile", 500))
        lines: list[str] = []
        used = 0
        for mem in profile.profile_memories(engine.memories):
            cost = estimate_tokens(mem.text) + 4
            if used + cost > budget:
                break
            used += cost
            lines.append(f"- {mem.text}")
        if not lines:
            return ""
        return f"【用户画像】\n" + "\n".join(lines)

    def _history_block(self, history: Any, rounds: int) -> str:
        conv: list[str] = []
        used = 0
        for role, text in reversed(list(history)[-rounds * 2 :]):
            cost = estimate_tokens(text) + 8
            if used + cost > self.config.budget_tokens:
                continue  # keep the newest rounds, drop the oldest
            used += cost
            conv.append(f"{'用户' if role == 'user' else '助手'}: {text}")
        conv.reverse()
        return "\n".join(conv) if conv else "（无）"

    # ------------------------------------------------------------------ #
    def build(
        self,
        engine: Any,
        history: Any,
        question: str,
        memories: list[Any],
        window_rounds: int = 20,
        max_memories: int = 6,
    ) -> list[dict]:
        """Assemble the system/user prompt under the token budget.

        Returns the message list (system + user). Disabled callers keep the
        v1 prompt; enabled callers get the layered version.
        """
        profile_block = self._profile_block(engine)
        memory_lines = []
        for i, hit in enumerate(memories[:max_memories], 1):
            tag = f"[{hit.region_id[:6]}]" if hit.region_id else ""
            memory_lines.append(f"{i}. (来源{tag} 相关度{hit.score:.2f}) {hit.memory.text}")
        memory_block = "\n".join(memory_lines) if memory_lines else "（暂无相关记忆）"
        conv_block = self._history_block(history, window_rounds)

        system = (
            "你是用户的 AI 助手，拥有长期记忆能力。\n"
            "【记忆】是用户过去说过的内容（按相关度排序），回答时可参考；\n"
            "如果记忆与当前问题无关就忽略，不要机械复述。\n"
            "用户可能重复提到同一话题，请保持前后一致，并以最近的说法为准。"
        )
        user_parts = [f"【相关记忆】\n{memory_block}"]
        if profile_block:
            user_parts.append(profile_block)
        user_parts.append(f"【最近对话（{window_rounds} 轮窗口）】\n{conv_block}")
        user_parts.append(f"【当前问题】\n{question}")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def budget_used(self, messages: list[dict]) -> int:
        return sum(estimate_tokens(m.get("content", "")) for m in messages)
