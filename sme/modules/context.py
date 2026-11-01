"""Scope-aware context assembly with a budget for the complete message list."""
from __future__ import annotations

import re
from typing import Any, Callable


def estimate_tokens(text: str) -> int:
    """Reproducible estimate, not a model tokenizer: CJK~1, Latin~4 chars."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
    return cjk + max(1, int((len(text) - cjk) / 4))


class ContextManager:
    def __init__(self, config: Any) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def build(self, engine: Any, history: Any, question: str, memories: list[Any],
              window_rounds: int = 20, max_memories: int = 6,
              ns: str | None = None,
              token_counter: Callable[[str], int] | None = None) -> list[dict]:
        """Keep required instructions/question, then whole evidence and rounds.

        ns=None preserves the global view. No fact or question is truncated
        in the middle of a negation/value. An optional counter counts rendered
        strings; it does not implicitly change the tokenizer used by an LLM.
        """
        system = (
            "你是用户的 AI 助手，拥有长期记忆能力。\n"
            "【记忆】是用户过去说过的内容（按相关度排序），回答时可参考；\n"
            "如果记忆与当前问题无关就忽略，不要机械复述。\n"
            "用户可能重复提到同一话题，请保持前后一致，并以最近的说法为准。"
        )
        evidence: list[str] = []
        profile_lines: list[str] = []
        rounds: list[list[tuple[str, str]]] = []

        def render() -> list[dict]:
            parts = ["【相关记忆】\n" + ("\n".join(evidence) or "（暂无相关记忆）")]
            if profile_lines:
                parts.append("【用户画像】\n" + "\n".join(profile_lines))
            conv = "\n".join(f"{'用户' if role == 'user' else '助手'}: {text}"
                             for group in rounds for role, text in group)
            parts += [f"【最近对话（{window_rounds} 轮窗口）】\n" + (conv or "（无）"),
                      f"【当前问题】\n{question}"]
            return [{"role": "system", "content": system},
                    {"role": "user", "content": "\n\n".join(parts)}]

        budget = int(self.config.budget_tokens)
        def fits() -> bool:
            return self.budget_used(render(), token_counter) <= budget
        if not fits():
            raise ValueError("context budget is too small for required instructions and question")
        for hit in memories[:max_memories]:
            m = hit.memory
            if ns is not None and m.metadata.get("ns") != ns:
                continue
            tag = f"[{hit.region_id[:6]}]" if hit.region_id else ""
            evidence.append(f"{len(evidence)+1}. (来源{tag} 相关度{hit.score:.2f}) {m.text}")
            if not fits():
                evidence.pop()
        profile = getattr(engine, "profile", None)
        if profile is not None and profile.enabled:
            count = token_counter or estimate_tokens
            for m in profile.profile_memories(engine.memories, ns=ns):
                profile_lines.append(f"- {m.text}")
                if (count("【用户画像】\n" + "\n".join(profile_lines)) > self.config.reserve_profile
                        or not fits()):
                    profile_lines.pop()
        groups: list[list[tuple[str, str]]] = []
        for role, text in list(history):
            if role == "user" or not groups:
                groups.append([])
            groups[-1].append((role, text))
        selected = groups[-window_rounds:] if window_rounds > 0 else []
        for group in reversed(selected):
            rounds.insert(0, group)
            if not fits():
                rounds.pop(0)
        return render()

    def budget_used(self, messages: list[dict],
                    token_counter: Callable[[str], int] | None = None) -> int:
        count = token_counter or estimate_tokens
        return 2 + sum(4 + count(m.get("role", "")) + count(m.get("content", ""))
                       for m in messages)
