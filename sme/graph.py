"""Memory Graph.

Maintains typed edges between memories:

    reference   - memory A references memory B
    cause       - A caused B (timeline causation)
    conversation- A and B belong to the same conversation chain
    summary     - A is a summary covering B
    parent/child- consolidation hierarchy
    neighbor    - spatially close memories (auto-derived)

Supports graph traversal (BFS/DFS) and export.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

from sme.models import MemoryEdge

KIND_REFERENCE = "reference"
KIND_CAUSE = "cause"
KIND_CONVERSATION = "conversation"
KIND_SUMMARY = "summary"
KIND_PARENT = "parent"
KIND_NEIGHBOR = "neighbor"


class MemoryGraph:
    def __init__(self) -> None:
        self.edges: list[MemoryEdge] = []

    # ------------------------------------------------------------------ #
    def add_edge(
        self,
        source: str,
        target: str,
        kind: str,
        weight: float = 1.0,
        note: str = "",
    ) -> MemoryEdge:
        if source == target:
            raise ValueError("self-loops are not allowed in the memory graph")
        existing = self.find(source, target, kind)
        if existing is not None:
            existing.weight = weight
            existing.note = note or existing.note
            return existing
        edge = MemoryEdge(
            source=source, target=target, kind=kind, weight=weight, note=note
        )
        self.edges.append(edge)
        return edge

    def find(self, source: str, target: str, kind: str) -> Optional[MemoryEdge]:
        for edge in self.edges:
            if (
                edge.source == source
                and edge.target == target
                and edge.kind == kind
            ) or (
                edge.kind == kind
                and edge.source == target
                and edge.target == source
                and kind in (KIND_NEIGHBOR, KIND_CONVERSATION)
            ):
                return edge
        return None

    def remove_edges_for(self, memory_id: str) -> int:
        before = len(self.edges)
        self.edges = [
            e
            for e in self.edges
            if e.source != memory_id and e.target != memory_id
        ]
        return before - len(self.edges)

    # ------------------------------------------------------------------ #
    def neighbors_of(
        self,
        memory_id: str,
        kinds: Iterable[str] | None = None,
    ) -> set[str]:
        kinds = set(kinds) if kinds else None
        result: set[str] = set()
        for edge in self.edges:
            if kinds and edge.kind not in kinds:
                continue
            if edge.source == memory_id:
                result.add(edge.target)
            elif edge.target == memory_id:
                result.add(edge.source)
        return result

    def add_auto_neighbors(
        self,
        memory_ids: Iterable[str],
        vectors: dict,
        k: int = 5,
        threshold: float = 0.55,
    ) -> int:
        """Derive spatial neighbor edges among the given memories."""
        from sme.utils import cosine_similarity

        ids = [mid for mid in memory_ids if mid in vectors]
        added = 0
        for i, a in enumerate(ids):
            scored = [
                (b, cosine_similarity(vectors[a], vectors[b]))
                for b in ids[i + 1 :]
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            for b, sim in scored[:k]:
                if sim >= threshold:
                    self.add_edge(a, b, KIND_NEIGHBOR, weight=sim)
                    added += 1
        return added

    # ------------------------------------------------------------------ #
    def traverse(
        self,
        start: str,
        kinds: Iterable[str] | None = None,
        mode: str = "bfs",
        max_depth: int = 3,
    ) -> list[str]:
        """Traverse the graph from `start`. Returns visited ids in order."""
        kinds = set(kinds) if kinds else None
        visited: list[str] = []
        depth: dict[str, int] = {start: 0}
        if mode == "dfs":
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.append(node)
                if depth[node] >= max_depth:
                    continue
                for nb in sorted(self.neighbors_of(node, kinds)):
                    if nb not in depth:
                        depth[nb] = depth[node] + 1
                        stack.append(nb)
        else:
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.append(node)
                if depth[node] >= max_depth:
                    continue
                for nb in sorted(self.neighbors_of(node, kinds)):
                    if nb not in depth:
                        depth[nb] = depth[node] + 1
                        queue.append(nb)
        return visited

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {"edges": [e.to_dict() for e in self.edges]}

    def load_dict(self, data: dict) -> None:
        self.edges = [MemoryEdge.from_dict(e) for e in data.get("edges", [])]

    def __len__(self) -> int:
        return len(self.edges)
