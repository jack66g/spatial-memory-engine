"""FactGraph extractor: LLM entity/relation extraction with rules fallback."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from sme.config import FactGraphConfig

EXTRACT_PROMPT = (
    "你是知识图谱提取引擎。从一条用户事实中提取实体和关系，输出 JSON。\n"
    "规则：\n"
    "1. 实体：人名（含称呼如朋友/同事/老板）、地点、物品、组织、其他；用户本人统一为“用户”\n"
    "2. 关系：实体之间的谓词（如 friend / works_at / lives_in / likes / owns）\n"
    "3. 宁缺毋滥：只提取明确提到的实体与关系\n"
    "输出格式（严格 JSON，不要其他内容）：\n"
    '{{"entities": [{{"name": "实体名", "kind": "person|place|item|org|other"}}],\n'
    ' "relations": [{{"source": "实体名", "target": "实体名", "predicate": "谓词"}}]}}\n\n'
    "用户事实：\n{text}"
)

# rules-mode entity kinds by marker
_MARKERS_ENTITY = [
    ("朋友", "person"), ("同事", "person"), ("老板", "person"), ("客户", "person"),
    ("经理", "person"), ("同学", "person"), ("老师", "person"),
    ("咖啡师", "person"), ("理发师", "person"), ("邻居", "person"),
    ("麻辣烫店", "place"), ("水果店", "place"), ("咖啡店", "place"), ("健身房", "place"),
    ("公司", "org"), ("大学", "org"), ("学校", "org"),
]


def _rules_extract(text: str):
    """Deterministic fallback: anchor on '用户' plus known noun markers."""
    entities: list[tuple[str, str]] = [("用户", "person")]
    relations: list[tuple[str, str, str]] = []
    for marker, kind in _MARKERS_ENTITY:
        if marker in text:
            name = marker
            entities.append((name, kind))
            relations.append(("用户", name, f"has_{kind}"))
            break  # keep it simple: one anchored relation per fact
    # try to capture the relation target: X -> is -> thing
    m = re.search(r"(用户|我)\s*(?:的)?([\u4e00-\u9fff]{2,8}?)\s*是\s*([\u4e00-\u9fff]{2,12})", text)
    if m and m.group(2) and m.group(3):
        src = m.group(2)
        dst = m.group(3)
        if src != "用户":
            entities.append((src, "person"))
        entities.append((dst, "other"))
        relations.append((src, dst, "is"))
    # 用户喜欢X / 用户不喜欢X
    m = re.search(r"用户(不)?喜欢([\u4e00-\u9fff]{2,12})", text)
    if m:
        item = m.group(2)
        entities.append((item, "item"))
        relations.append(("用户", item, "dislikes" if m.group(1) else "likes"))
    # X在Y上班/工作/居住
    m = re.search(r"([\u4e00-\u9fff]{2,8})(?:在)([\u4e00-\u9fff]{2,12})(?:上班|工作|居住|住)", text)
    if m:
        who, where = m.group(1), m.group(2)
        if who != "用户":
            entities.append((who, "person"))
        entities.append((where, "place" if ("住" in text) else "org"))
        relations.append((who, where, "works_at" if ("上班" in text or "工作" in text) else "lives_in"))
    if text.startswith("用户"):
        rest = text[2:]
        m2 = re.match(r"([\u4e00-\u9fff]{2,10})[\u4e00-\u9fff]?(?:\s*(?:了|到|在|去))?([\u4e00-\u9fff]{2,16})", rest)
        if m2:
            entities.append((m2.group(1), "item"))
            relations.append(("用户", m2.group(1), "likes"))
    return entities, relations


class FactGraphExtractor:
    def __init__(self, config: FactGraphConfig, llm: Optional[Any] = None) -> None:
        self.config = config
        self.llm = llm
        self.extract_calls = 0
        self.fallbacks = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.extract_mode != "off"

    # ------------------------------------------------------------------ #
    def extract(self, text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        """Extract (entities, relations) from one fact text."""
        if not self.enabled:
            return [], []
        if self.config.extract_mode == "llm" and self.llm is not None and self.llm.configured:
            try:
                raw = self.llm.chat(
                    [{"role": "user", "content": EXTRACT_PROMPT.format(text=text[:400])}],
                    temperature=0.1,
                    max_tokens=400,
                )
                data = self._parse(raw)
                if data:
                    self.extract_calls += 1
                    return data
            except Exception:  # noqa: BLE001
                pass
        self.fallbacks += 1
        return _rules_extract(text)

    @staticmethod
    def _parse(raw: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]] | None:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        entities = []
        for item in data.get("entities", []):
            if isinstance(item, dict) and item.get("name"):
                entities.append((str(item["name"]).strip(), str(item.get("kind", "other")).strip() or "other"))
        relations = []
        for item in data.get("relations", []):
            if isinstance(item, dict) and item.get("source") and item.get("target"):
                relations.append((
                    str(item["source"]).strip(),
                    str(item["target"]).strip(),
                    str(item.get("predicate", "relates_to")).strip() or "relates_to",
                ))
        return entities, relations
