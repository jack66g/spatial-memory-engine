"""Memory Compression: long-term summarization.

Older memories within a region are compressed into a Summary Node that
participates in retrieval (it carries a summary embedding). Historical
memories are kept in the space - compression never deletes anything.

A region is compressed at most once per "generation" (tracked via the
summary memory's metadata), so repeated runs do not duplicate nodes.
"""

from __future__ import annotations

from typing import Optional

from sme.config import CompressionConfig
from sme.llm import LLMClient
from sme.utils import age_days, now


class CompressionEngine:
    def __init__(
        self,
        config: CompressionConfig,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.compression_count = 0

    # ------------------------------------------------------------------ #
    def find_compressible(self, engine: object, max_regions: int = 10) -> list[list]:
        """Regions -> candidate memory lists (old, un-summarized memories)."""
        out: list[list] = []
        for region in engine.space.regions.values():
            if region.size < self.config.min_region_compact:
                continue
            if self._region_compressed(engine, region.id):
                continue
            candidates = [
                engine.memories[mid]
                for mid in region.member_ids
                if mid in engine.memories
                and not engine.memories[mid].archived
                and engine.memories[mid].source != "summary"
                and age_days(engine.memories[mid].last_hit) >= self.config.age_days_threshold
            ]
            if len(candidates) >= self.config.min_region_compact:
                out.append(candidates)
            if len(out) >= max_regions:
                break
        return out

    # ------------------------------------------------------------------ #
    def compress(self, engine: object) -> list:
        """Generate a Summary Memory for each compressible region."""
        created: list = []
        groups = self.find_compressible(engine)
        for candidates in groups:
            region_id = engine.space.region_for(candidates[0].id)
            summary_text = self._summarize(candidates)
            summary = engine.memory_manager.summary_memory(
                text=summary_text,
                member_ids=[m.id for m in candidates],
                importance=0.7,
            )
            summary.metadata["compresses_region"] = region_id
            summary.metadata["compressed_at"] = now()
            summary.metadata["cover_count"] = len(candidates)
            created.append(summary)
            self.compression_count += 1
        return created

    # ------------------------------------------------------------------ #
    def _summarize(self, candidates: list) -> str:
        if self.config.summary_source == "llm" and self.llm is not None:
            return self.llm.summarize_memories(candidates)
        preview = "；".join(m.text for m in candidates[:5])
        more = f" 等{len(candidates)}条" if len(candidates) > 5 else ""
        return f"长期记忆摘要（{len(candidates)}条）{more}：{preview}"

    @staticmethod
    def _region_compressed(engine: object, region_id: str) -> bool:
        for memory in engine.memories.values():
            if memory.source == "summary" and memory.metadata.get("compresses_region") == region_id:
                return True
        return False
