"""Module 01 - ExtractionEngine: LLM/rules fact extraction on the write path.

The first gate on the write side: only clean *facts* enter the memory store.
Questions / chit-chat / AI answers are dropped by default (breaking the
hallucination feedback loop where the model's own words get re-memorized).

Flow (v2 模块设计 module 01)::

    user text -> extract(texts) -> dedup(facts) -> store(engine, facts)

* LLM mode: a strict prompt returns a JSON array of definitive facts.
* rules mode (offline fallback): non-question + length >= 6 + no question
  words + not a pure-emotion sentence.
* dedup: each fact is embedded and compared to the top-5 most similar
  memories; cosine >= dedup_threshold => skipped.

Disabled (or mode="off") => extraction is a no-op and the engine stores the
raw text exactly like v1.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import numpy as np

from sme.config import ExtractionConfig
from sme.models import Fact
from sme.utils import tokenize

QUESTION_WORDS = [
    "什么", "怎么", "为什么", "为啥", "哪", "谁", "几", "多少", "吗",
    "呢", "是不是", "有没有", "如何", "是否", "啥", "咋", "何时", "哪里",
    "哪儿", "哪个", "哪些", "干嘛", "为啥", "为什么",
]

QUESTION_END = ("？", "?", "吗？")

EMOTION_WORDS = {
    "哈哈", "呵呵", "嘿嘿", "嗯嗯", "呜呜", "啊啊", "啧啧", "唉", "哎",
    "嗨", "哦哦", "好的", "好哒", "okk", "ok", "嗯", "哦", "呵呵呵",
    "哈哈哈", "嘿嘿嘿", "呜呜呜", "啊哈哈", "好耶", "哇塞", "天了噜",
}

EXTRACTION_PROMPT = (
    "你是记忆提取引擎。从用户的聊天消息中提取【确定的事实】，输出 JSON 数组。\n"
    "规则：\n"
    "1. 只提取用户亲口说出的确定事实（个人信息、偏好、计划、经历、关系、观点结论等）\n"
    "2. 主语统一归一为“用户”，如“我叫小林”→“用户叫小林”\n"
    "3. 忽略问题、反问、寒暄、语气词、与 AI 的对话过程本身\n"
    "4. 识别纠正语气（其实/不对/更正/说错/搞错），此类事实 kind=\"correction\"\n"
    "5. 识别疑问句，kind=\"question\"（只保留值得回放的问题本身）\n"
    "6. 宁缺毋滥：不确定、猜测、情绪化的话不要提取\n"
    "7. 用与原文相同的语言（中文）输出\n"
    "输出格式（严格 JSON 数组，不要任何其他内容）：\n"
    '[{{"text": "事实文本", "kind": "fact|correction|question", "subject": "用户"}}]\n'
    "如果没有可提取的事实，输出 []\n\n"
    "用户消息：\n{text}"
)


def is_question(text: str) -> bool:
    """Heuristic question detection (shared by extraction and qapair)."""
    t = text.strip()
    if not t:
        return False
    if t.endswith(QUESTION_END):
        return True
    if any(w in t for w in QUESTION_WORDS):
        return True
    if re.search(r"\d+\s*(个|条|次)?\s*[？?]", t):
        return True
    return False


def is_pure_emotion(text: str) -> bool:
    t = text.strip()
    if len(t) <= 4 and t in EMOTION_WORDS:
        return True
    if len(t) <= 8 and all(ch in "哈哈嘿嘿嗯嗯哦哦啊啊呜呜呵呵" for ch in t):
        return True
    return False


TRAILING_PARTICLES = ("了吗", "嘛呢", "吧", "了", "呢", "吗")  # longest first


def _strip_particles(text: str) -> str:
    """Normalize trailing modal particles (iteration 2.5).

    "用户不喜欢喝咖啡了" -> "用户不喜欢喝咖啡" so the dedup/versioning
    machinery matches it to the particle-free statement instead of splitting
    the entity ("喜欢喝咖啡了" vs "喜欢喝咖啡").
    """
    t = text.strip()
    changed = True
    while changed and len(t) > 4:
        changed = False
        for p in TRAILING_PARTICLES:
            if t.endswith(p):
                t = t[: -len(p)].rstrip()
                changed = True
                break
    return t


def _rules_extract(text: str) -> list[Fact]:
    """Offline deterministic extraction (rules mode / LLM fallback)."""
    t = text.strip()
    if not t:
        return []
    if is_question(t):
        # questions only become QA pairs when module 02 is enabled; the
        # extraction itself never stores a bare question as a fact.
        return [Fact(text=t, kind="question", subject="用户")]
    if is_pure_emotion(t):
        return []
    if len(t) < 6:
        return []
    if _looks_like_chat(t):
        return []
    return [Fact(text=_strip_particles(t), kind="fact", subject="用户")]


def _looks_like_chat(text: str) -> bool:
    """Chit-chat detection for rules mode: no fact-ish payload."""
    t = text
    if re.match(r"^(嗯|哦|好的|好|行|ok|可以|对|是|知道|明白|哈哈|嘿嘿)", t, re.I):
        return True
    if len(tokenize(t)) <= 2:
        return True
    return False


def parse_llm_facts(raw: str) -> list[Fact]:
    """Parse the LLM's JSON array output defensively."""
    raw = raw.strip()
    # strip code fences / prose if the model was chatty
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    facts: list[Fact] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        kind = str(item.get("kind", "fact")).strip().lower()
        if kind not in ("fact", "correction", "question"):
            kind = "fact"
        subject = str(item.get("subject", "用户")).strip() or "用户"
        facts.append(Fact(text=text, kind=kind, subject=subject))
    return facts


class ExtractionEngine:
    def __init__(self, config: ExtractionConfig, llm: Optional[Any] = None) -> None:
        self.config = config
        self.llm = llm
        self.extract_calls = 0
        self.facts_extracted = 0
        self.facts_dropped = 0
        self.dedup_skipped = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.mode != "off"

    # ------------------------------------------------------------------ #
    def extract(self, text: str, assistant: bool = False) -> list[Fact]:
        """Extract clean facts from one raw text."""
        if not self.enabled:
            return []
        if assistant and not self.config.store_assistant:
            return []  # AI answers never enter the fact store (module 01)
        t = text.strip()
        if not t:
            return []
        facts: list[Fact] = []
        if self.config.mode == "llm" and self.llm is not None and self.llm.configured:
            try:
                prompt = EXTRACTION_PROMPT.format(text=t[:600])
                raw = self.llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=600,
                )
                facts = parse_llm_facts(raw)
                self.extract_calls += 1
            except Exception:  # noqa: BLE001 - fall back to rules
                facts = []
        if not facts and (self.config.mode == "rules" or (
            self.config.mode == "llm" and self.config.fallback_rules
        )):
            facts = _rules_extract(t)
        # drop chit-chat kind / empty subjects
        kept: list[Fact] = []
        for fact in facts:
            if fact.kind == "chat":
                self.facts_dropped += 1
                continue
            kept.append(fact)
        self.facts_extracted += len(kept)
        return kept

    # ------------------------------------------------------------------ #
    def dedup(self, facts: list[Fact], engine: Any) -> list[Fact]:
        """Embed each fact and drop those too similar to existing memories.

        Cosine >= dedup_threshold against any of the top-5 most similar
        existing memories => skip (the fact is already remembered).
        """
        if not self.enabled or not facts:
            return facts
        threshold = self.config.dedup_threshold
        out: list[Fact] = []
        for fact in facts:
            if fact.kind == "correction":
                # correction statements belong to module 05; never dedup them
                out.append(fact)
                continue
            vec = engine.embeddings.embed_one(fact.text)
            sims = self._top_similar(engine, vec, k=5)
            if sims and sims[0] >= threshold:
                self.dedup_skipped += 1
                continue
            out.append(fact)
        return out

    @staticmethod
    def _top_similar(engine: Any, vec: np.ndarray, k: int = 5) -> list[float]:
        """Cosine of the top-k most similar *unarchived* memories (bounded)."""
        try:
            regions = engine.space.query_regions(vec, 2)
        except Exception:  # noqa: BLE001 - no regions yet
            regions = []
        ids: set[str] = set()
        for rhit in regions:
            ids.update(engine.space.candidates_in_region(rhit.region.id))
            if len(ids) >= 400:
                break
        if not ids:
            return []
        mems = [
            m for m in (engine.memories.get(mid) for mid in ids)
            if m is not None and not m.archived
        ]
        if not mems:
            return []
        mat = np.stack([m.embedding for m in mems if m.embedding is not None])
        if mat.size == 0:
            return []
        v = vec / max(float(np.linalg.norm(vec)), 1e-12)
        mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
        sims = (mat @ v).reshape(-1)
        sims = np.sort(sims)[::-1][:k]
        return [float(s) for s in sims if s > 0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "mode": self.config.mode,
            "extract_calls": self.extract_calls,
            "facts_extracted": self.facts_extracted,
            "facts_dropped": self.facts_dropped,
            "dedup_skipped": self.dedup_skipped,
        }
