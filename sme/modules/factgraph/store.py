"""FactGraph store: entities, temporal relations and multi-hop queries."""

from __future__ import annotations

from typing import Any, Optional

from sme.config import FactGraphConfig
from sme.models import Entity, Relation
from sme.utils import now


class FactGraph:
    def __init__(self, config: FactGraphConfig) -> None:
        self.config = config
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self._name_index: dict[str, list[str]] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------ #
    # write side
    # ------------------------------------------------------------------ #
    def upsert_entity(self, name: str, kind: str = "other",
                      memory_id: Optional[str] = None) -> Entity:
        name = (name or "").strip()
        if not name:
            raise ValueError("entity name must not be empty")
        for eid in self._name_index.get(name, []):
            ent = self.entities.get(eid)
            if ent is not None and ent.is_valid():
                if memory_id and memory_id not in ent.memory_ids:
                    ent.memory_ids.append(memory_id)
                return ent
        ent = Entity(name=name, kind=kind)
        if memory_id:
            ent.memory_ids.append(memory_id)
        self.entities[ent.id] = ent
        self._name_index.setdefault(name, []).append(ent.id)
        if len(self.entities) > self.config.max_entities * 4:
            self._prune()
        return ent

    def add_relation(self, src: str, dst: str, predicate: str,
                     memory_id: Optional[str] = None,
                     reference: Optional[float] = None) -> Relation:
        """Add a relation; a NEW predicate on the same entity pair
        invalidates the OLD one (latest statement wins)."""
        ref = now() if reference is None else reference
        for rel in self.relations:
            if not rel.is_valid(ref):
                continue
            if {rel.source, rel.target} == {src, dst}:
                if rel.predicate == predicate:
                    # same relation restated -> refresh timestamp
                    rel.valid_at = ref
                    if memory_id and memory_id not in rel.memory_ids:
                        rel.memory_ids.append(memory_id)
                    return rel
                # conflicting predicate on the same pair -> supersede
                rel.invalid_at = ref
        rel = Relation(
            source=src, target=dst, predicate=predicate, valid_at=ref
        )
        if memory_id:
            rel.memory_ids.append(memory_id)
        self.relations.append(rel)
        return rel

    def add_fact(self, entities: list[tuple[str, str]],
                 relations: list[tuple[str, str, str]],
                 memory_id: Optional[str] = None) -> None:
        """Ingest extracted entities/relations for one fact."""
        eid_of: dict[str, str] = {}
        for name, kind in entities:
            ent = self.upsert_entity(name, kind, memory_id)
            eid_of[name] = ent.id
        for src, dst, pred in relations:
            if src not in eid_of or dst not in eid_of:
                continue
            self.add_relation(eid_of[src], eid_of[dst], pred, memory_id)

    def _prune(self) -> None:
        """Drop entities with no valid relations and no memories (bounded)."""
        keep = max(self.config.max_entities, 1)
        used = {rel.source for rel in self.relations if rel.is_valid()} | {
            rel.target for rel in self.relations if rel.is_valid()
        }
        drop = [eid for eid, ent in self.entities.items()
                if eid not in used and not ent.memory_ids]
        for eid in drop[: max(len(drop) - keep, 0)]:
            ent = self.entities.pop(eid, None)
            if ent is not None:
                # remove every index entry for this name (multiple entities
                # may share one name - popping only the first would leak)
                eids = self._name_index.get(ent.name)
                if eids is not None:
                    if eid in eids:
                        eids.remove(eid)
                    if not eids:
                        self._name_index.pop(ent.name, None)

    # ------------------------------------------------------------------ #
    # read side
    # ------------------------------------------------------------------ #
    def find_entities(self, text: str) -> list[Entity]:
        """Entities whose name appears in the query text."""
        out = []
        for name, eids in self._name_index.items():
            if name and name in text:
                for eid in eids:
                    ent = self.entities.get(eid)
                    if ent is not None and ent.is_valid():
                        out.append(ent)
        return out

    def neighbors(self, entity_id: str, reference: Optional[float] = None) -> list[tuple[str, str]]:
        """(neighbor_entity_id, predicate) pairs via valid relations."""
        ref = now() if reference is None else reference
        out: list[tuple[str, str]] = []
        for rel in self.relations:
            if not rel.is_valid(ref):
                continue
            if rel.source == entity_id:
                out.append((rel.target, rel.predicate))
            elif rel.target == entity_id:
                out.append((rel.source, rel.predicate))
        return out

    def multi_hop(self, entity_ids: list[str], max_depth: int | None = None,
                  reference: Optional[float] = None) -> dict[str, int]:
        """BFS over valid relations; returns {entity_id: hop_depth}."""
        max_depth = max_depth or self.config.max_depth
        ref = now() if reference is None else reference
        visited: dict[str, int] = {eid: 0 for eid in entity_ids}
        frontier = list(entity_ids)
        for depth in range(1, max_depth + 1):
            if not frontier:
                break
            nxt: list[str] = []
            for eid in frontier:
                for nb, _pred in self.neighbors(eid, ref):
                    if nb in visited:
                        continue
                    visited[nb] = depth
                    nxt.append(nb)
            frontier = nxt
        return visited

    def memories_for(self, entity_ids: list[str]) -> list[tuple[str, int]]:
        """(memory_id, min_hop) pairs behind the given entities."""
        hops = self.multi_hop(entity_ids)
        out: dict[str, int] = {}
        for eid, depth in hops.items():
            ent = self.entities.get(eid)
            if ent is None:
                continue
            for mid in ent.memory_ids:
                if mid not in out or depth < out[mid]:
                    out[mid] = depth
        return [(mid, d) for mid, d in out.items()]

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        valid = sum(1 for r in self.relations if r.is_valid())
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "valid_relations": valid,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self.entities = {e.id: e for e in (Entity.from_dict(d) for d in data.get("entities", []))}
        self.relations = [Relation.from_dict(d) for d in data.get("relations", [])]
        self._name_index = {}
        for ent in self.entities.values():
            self._name_index.setdefault(ent.name, []).append(ent.id)
