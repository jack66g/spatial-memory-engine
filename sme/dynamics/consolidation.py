"""Memory Consolidation: automatic fusion of similar memories.

Example:
    today:    "user likes apples"
    tomorrow: "user likes iPhones"
    later:    "user likes Macs"
    ->  a Summary Memory "user likes the Apple ecosystem" is generated,
        wired as Parent of the three memories (children), and linked via
        memory-graph edges of kind `summary`.

Consolidation runs inside regions: memories that are spatially close
(cosine >= threshold) and share a theme are grouped greedily. History is
kept - nothing is deleted or rewritten.
"""

from __future__ import annotations

from typing import Optional

from sme.config import ConsolidationConfig
from sme.llm import LLMClient
from sme.utils import cosine_similarity, now


class ConsolidationEngine:
    def __init__(
        self,
        config: ConsolidationConfig,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.consolidation_count = 0
        self.last_run_at: float = 0.0

    # ------------------------------------------------------------------ #
    def find_groups(self, engine: object, max_groups: int = 20) -> list[list]:
        """Find groups of similar memories inside each region.

        Two mechanisms:
          1. fine-grained greedy clustering at the configured threshold;
          2. whole-region coherence: a small, dense region whose members are
             mutually similar (avg cosine >= threshold * 0.6) is fused into a
             single group - this is the "user likes apples today, iPhones
             tomorrow, Macs later -> apple ecosystem" auto-fusion.
        """
        threshold = self.config.similarity_threshold
        groups: list[list] = []
        for region in engine.space.regions.values():
            members = [
                engine.memories[mid]
                for mid in region.member_ids
                if mid in engine.memories
                and not engine.memories[mid].archived
                and engine.memories[mid].source != "summary"
            ]
            if len(members) < self.config.min_group_size:
                continue
            members.sort(key=lambda m: m.importance, reverse=True)

            # 1) fine-grained clustering
            clusters = self._greedy_clusters(members, threshold)
            covered: set[str] = set()
            for cluster in clusters:
                if len(cluster) >= self.config.min_group_size:
                    group = cluster[: self.config.max_group_size]
                    covered.update(m.id for m in group)
                    groups.append(group)
                    if len(groups) >= max_groups:
                        return groups

            # 2) whole-region coherence fallback
            if len(members) <= self.config.max_group_size:
                coverage = len(covered) / len(members)
                if coverage >= 0.6:
                    continue  # already covered by fine-grained clusters
                if self._region_coherent(members, threshold):
                    groups.append(members)
                    if len(groups) >= max_groups:
                        return groups
        return groups

    @staticmethod
    def _greedy_clusters(members: list, threshold: float) -> list[list]:
        used: set[str] = set()
        clusters: list[list] = []
        for anchor in members:
            if anchor.id in used:
                continue
            cluster = [anchor]
            for other in members:
                if other.id in used or other.id == anchor.id:
                    continue
                if anchor.embedding is not None and other.embedding is not None:
                    if (
                        cosine_similarity(anchor.embedding, other.embedding)
                        >= threshold
                    ):
                        cluster.append(other)
            if len(cluster) >= 2:
                for m in cluster:
                    used.add(m.id)
                clusters.append(cluster)
        return clusters

    @staticmethod
    def _region_coherent(members: list, threshold: float) -> bool:
        """Average pairwise cosine of a sample of the members."""
        sample = members[:30]
        if len(sample) < 2:
            return False
        total, pairs = 0.0, 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                a, b = sample[i].embedding, sample[j].embedding
                if a is None or b is None:
                    continue
                total += cosine_similarity(a, b)
                pairs += 1
                if pairs >= 200:
                    break
        if pairs == 0:
            return False
        return (total / pairs) >= threshold * 0.6

    # ------------------------------------------------------------------ #
    def consolidate(self, engine: object, groups: Optional[list[list]] = None) -> list:
        """Run consolidation: generate a summary memory per group."""
        groups = groups if groups is not None else self.find_groups(engine)
        created: list = []
        for group in groups:
            ids = [m.id for m in group]
            # skip if this exact set is already summarized
            if self._already_consolidated(engine, ids):
                continue
            if self.config.summary_source == "llm" and self.llm is not None:
                summary_text = self.llm.summarize_memories(group)
            else:
                summary_text = self._template_summary(group)
            summary = engine.memory_manager.summary_memory(
                text=summary_text,
                member_ids=ids,
                importance=0.75,
            )
            summary.metadata["covers"] = sorted(ids)
            summary.metadata["consolidated_at"] = now()
            created.append(summary)
            self.consolidation_count += 1
        self.last_run_at = now()
        return created

    # ------------------------------------------------------------------ #
    def _template_summary(self, group: list) -> str:
        topics = self._common_tokens(group)
        preview = "；".join(m.text for m in group[:3])
        more = f"，共{len(group)}条" if len(group) > 3 else ""
        if topics:
            return f"关于「{topics}」的 {len(group)} 条相关记忆{more}：{preview}"
        return f"{len(group)} 条相关记忆{more}：{preview}"

    @staticmethod
    def _common_tokens(group: list) -> str:
        """Content-word topic extraction for template summaries.

        Tokens must appear in >= 2 members and be at least 4 chars long;
        the top tokens are ranked by (frequency * length**2) so content
        words like "apple"/"language" beat filler like "user".
        """
        from collections import Counter

        from sme.utils import tokenize

        counter: Counter = Counter()
        for m in group:
            counter.update(set(tokenize(m.text)))
        candidates = [
            (tok, cnt)
            for tok, cnt in counter.items()
            if cnt >= 2 and len(tok) >= 4
        ]
        candidates.sort(key=lambda pair: (-pair[1] * len(pair[0]) ** 2, pair[0]))
        common = [tok for tok, _ in candidates[:4]]
        return "、".join(common) if common else ""

    @staticmethod
    def _already_consolidated(engine: object, ids: set[str]) -> bool:
        for memory in engine.memories.values():
            if memory.source == "summary" and memory.metadata.get("covers"):
                if set(memory.metadata["covers"]) == set(ids):
                    return True
        return False
