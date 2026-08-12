"""Region Manager.

Owns the lifecycle of spatial regions:

    create / delete / split / merge / stats / density / centroid update
    / region graph maintenance

All operations work on a ``SpatialMemorySpace`` object. Split/merge are
density-driven, NOT fixed KMeans clustering:

- a region splits when it grows dense and large (its members form a coherent
  blob that is better represented by two blobs);
- two regions merge when their centroids drift close together;
- every region keeps a centroid, radius, bounding box and density; neighbors
  form a region graph that evolves as regions evolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from sme.config import RegionConfig
from sme.models import Region, RegionEdge, RegionStats, new_id
from sme.utils import cosine_similarity, euclidean_distance, logger, now

MAX_SPLITS_PER_PASS = 8
MAX_MERGES_PER_PASS = 8


@dataclass
class EvolutionEvent:
    """A record of a region evolution action (split / merge / delete)."""

    kind: str  # split | merge | region_created | region_deleted
    region_id: str
    detail: str = ""


class RegionManager:
    """Density-based region lifecycle manager."""

    def __init__(self, config: RegionConfig) -> None:
        self.config = config
        self.split_count = 0
        self.merge_count = 0
        self.history: list[EvolutionEvent] = []
        self._history_cap = 500

    def _note_event(self, event: EvolutionEvent) -> None:
        self.history.append(event)
        if len(self.history) > self._history_cap:
            self.history = self.history[-self._history_cap:]

    # ------------------------------------------------------------------ #
    # creation / deletion
    # ------------------------------------------------------------------ #
    def create_region(
        self,
        space: "SpatialMemorySpace",
        member_ids: list[str] | set[str] | None = None,
        parent: Optional[str] = None,
        region_id: Optional[str] = None,
    ) -> Region:
        members = set(member_ids or [])
        region = Region(
            id=region_id or new_id("r"),
            member_ids=members,
            parent_region=parent,
            created_at=now(),
        )
        region.update_geometry(space.vectors, space.dim)
        space.regions[region.id] = region
        space._touch(region.id)
        self.note_region(space, region.id)
        self._note_event(
            EvolutionEvent("region_created", region.id, f"{len(members)} members")
        )
        return region

    def delete_region(self, space: "SpatialMemorySpace", region_id: str) -> None:
        if region_id not in space.regions:
            return
        del space.regions[region_id]
        # drop the ANN index: count-based rebuilds cannot detect a
        # delete+create cycle that keeps the region count unchanged, so a
        # stale deleted-region centroid could otherwise be returned by
        # nearest_region (and crash the write path with a KeyError).
        space._region_ann = None
        # clear membership pointers of any orphans
        for mid in list(space.vectors):
            if mid not in space._membership:
                continue
            if space._membership[mid] == region_id:
                space._set_membership(mid, None)
        space._prune_edges_for(region_id)
        self.note_region(space, region_id)
        self._note_event(EvolutionEvent("region_deleted", region_id))

    # ------------------------------------------------------------------ #
    # geometry / density
    # ------------------------------------------------------------------ #
    def refresh_geometry(self, space: "SpatialMemorySpace", region_id: str) -> Region:
        """Recompute centroid, radius, bbox, density for one region."""
        region = space.regions[region_id]
        region.update_geometry(space.vectors, space.dim)
        space._touch(region_id)
        self.note_region(space, region_id)
        return region

    def region_stats(self, space: "SpatialMemorySpace") -> RegionStats:
        space.sync_geometry()  # stats need exact radius/density
        regions = list(space.regions.values())
        if not regions:
            return RegionStats(count=0)
        sizes = [r.size for r in regions]
        stats = RegionStats(
            count=len(regions),
            avg_size=float(np.mean(sizes)),
            avg_density=float(np.mean([r.density for r in regions])),
            max_size=max(sizes),
            min_size=min(sizes),
            avg_radius=float(np.mean([r.radius for r in regions])),
            edge_count=len(space.region_edges),
            avg_neighbors=float(np.mean([len(self.neighbors_of(space, r.id)) for r in regions]))
            if regions
            else 0.0,
            split_count=self.split_count,
            merge_count=self.merge_count,
        )
        return stats

    # ------------------------------------------------------------------ #
    # region graph
    # ------------------------------------------------------------------ #
    def neighbors_of(self, space: "SpatialMemorySpace", region_id: str) -> list[RegionEdge]:
        edges: list[RegionEdge] = []
        for edge in space.region_edges.values():
            if region_id in (edge.source, edge.target):
                edges.append(edge)
        return edges

    def update_edges_for(
        self, space: "SpatialMemorySpace", region_ids: set[str]
    ) -> None:
        """Remove stale edges and recompute neighbors for the given regions."""
        space.sync_geometry()  # edge thresholds use exact radius values
        keep = {}
        for key, edge in space.region_edges.items():
            if edge.source in region_ids or edge.target in region_ids:
                space._close_edge_keys.discard(key)
                continue
            keep[key] = edge
        space.region_edges = keep
        for rid in region_ids:
            self._recompute_edges(space, rid)

    def _recompute_edges(self, space: "SpatialMemorySpace", region_id: str) -> None:
        a = space.regions[region_id]
        if a.centroid is None:
            return
        for other_id, other in space.regions.items():
            if other_id == region_id or other.centroid is None:
                continue
            dist = euclidean_distance(a.centroid, other.centroid)
            threshold = (a.radius + other.radius) * self.config.neighbor_factor
            key = frozenset((region_id, other_id))
            if dist <= threshold:
                space.region_edges[key] = RegionEdge(
                    source=region_id,
                    target=other_id,
                    distance=dist,
                )
                if dist < self.config.merge_distance:
                    space._close_edge_keys.add(key)
                else:
                    space._close_edge_keys.discard(key)
            else:
                space.region_edges.pop(key, None)
                space._close_edge_keys.discard(key)

    # ------------------------------------------------------------------ #
    # split
    # ------------------------------------------------------------------ #
    def split_region(
        self, space: "SpatialMemorySpace", region_id: str
    ) -> tuple[Region, Region] | None:
        """Density-based 2-way split. Returns the two child regions or None."""
        region = space.regions.get(region_id)
        if region is None or region.size < 4 or region.centroid is None:
            return None
        members = list(region.member_ids)
        # seed selection: farthest from centroid, then farthest from that
        seed_a = max(members, key=lambda mid: euclidean_distance(
            space.vectors[mid], region.centroid
        ))
        seed_b = max(members, key=lambda mid: euclidean_distance(
            space.vectors[mid], space.vectors[seed_a]
        ))
        if seed_a == seed_b:
            return None
        va, vb = space.vectors[seed_a], space.vectors[seed_b]
        part_a, part_b = {seed_a}, {seed_b}
        for mid in members:
            if mid in (seed_a, seed_b):
                continue
            d_a = euclidean_distance(space.vectors[mid], va)
            d_b = euclidean_distance(space.vectors[mid], vb)
            (part_a if d_a <= d_b else part_b).add(mid)
        if not part_a or not part_b:
            return None
        # keep the original id for the bigger part to preserve references
        keep, other = (part_a, part_b) if len(part_a) >= len(part_b) else (part_b, part_a)
        child_keep = Region(
            id=region_id,
            member_ids=keep,
            parent_region=region.parent_region,
            created_at=region.created_at,
            generation=region.generation + 1,
        )
        child_new = Region(
            id=new_id("r"),
            member_ids=other,
            parent_region=region_id,
            generation=region.generation + 1,
        )
        child_keep.update_geometry(space.vectors, space.dim)
        child_new.update_geometry(space.vectors, space.dim)
        space.regions[region_id] = child_keep
        space.regions[child_new.id] = child_new
        # re-point the moved half so membership stays consistent after the
        # split (the other half keeps the original region id)
        for mid in other:
            space._set_membership(mid, child_new.id)
        space._touch(region_id)
        space._touch(child_new.id)
        self.note_region(space, region_id)
        self.note_region(space, child_new.id)
        self.update_edges_for(space, {region_id, child_new.id})
        self.split_count += 1
        logger.info("region split: %s -> %d/%d members",
                    region_id, len(keep), len(other))
        self._note_event(
            EvolutionEvent(
                "split",
                region_id,
                f"{len(keep)} -> {len(other)} members (density {region.density:.1f})",
            )
        )
        return child_keep, child_new

    # ------------------------------------------------------------------ #
    # merge
    # ------------------------------------------------------------------ #
    def merge_regions(
        self, space: "SpatialMemorySpace", a_id: str, b_id: str
    ) -> Region | None:
        """Absorb the smaller region into the larger one."""
        if a_id not in space.regions or b_id not in space.regions:
            return None
        a, b = space.regions[a_id], space.regions[b_id]
        keeper_id = a_id if a.size >= b.size else b_id
        absorbed_id = b_id if keeper_id == a_id else a_id
        keeper = space.regions[keeper_id]
        absorbed = space.regions[absorbed_id]
        merged_members = keeper.member_ids | absorbed.member_ids
        keeper.member_ids = merged_members
        keeper.update_geometry(space.vectors, space.dim)
        # re-point the absorbed members to the keeper so membership stays
        # consistent after the merge
        for mid in absorbed.member_ids:
            space._set_membership(mid, keeper_id)
        del space.regions[absorbed_id]
        space._touch(keeper_id)
        self.note_region(space, keeper_id)
        self.note_region(space, absorbed_id)
        self.update_edges_for(space, {keeper_id})
        # hysteresis: a region that was just merged should not be split again
        space._merge_ops[keeper_id] = space.write_ops
        space._merge_recently.add(keeper_id)
        self.merge_count += 1
        logger.info("region merge: %s absorbed by %s (%d members)",
                    absorbed_id, keeper_id, absorbed.size)
        self._note_event(
            EvolutionEvent(
                "merge",
                keeper_id,
                f"absorbed {absorbed_id} ({absorbed.size} members)",
            )
        )
        return keeper

    # ------------------------------------------------------------------ #
    # candidate tracking (keeps evolution passes cheap)
    # ------------------------------------------------------------------ #
    def note_region(self, space: "SpatialMemorySpace", region_id: str) -> None:
        """Register a region for split / absorption consideration."""
        region = space.regions.get(region_id)
        if region is None:
            space._split_candidates.discard(region_id)
            space._small_candidates.discard(region_id)
            return
        if region.size >= self.config.split_threshold:
            space._split_candidates.add(region_id)
        else:
            space._split_candidates.discard(region_id)
        if region.size < self.config.min_region_size:
            space._small_candidates.add(region_id)
        else:
            space._small_candidates.discard(region_id)

    # ------------------------------------------------------------------ #
    # evolution pass (density increase -> split, closeness -> merge)
    # ------------------------------------------------------------------ #
    def evolution_pass(self, space: "SpatialMemorySpace") -> list[EvolutionEvent]:
        """One pass of region evolution: splits first, then merges.

        Only candidate regions (tracked incrementally) are examined, so a
        pass is O(candidates) instead of O(all regions).
        """
        events: list[EvolutionEvent] = []
        cfg = self.config

        space.sync_geometry()  # split decisions read exact density values

        # --- 1) density-based splits ----------------------------------- #
        splits = 0
        for rid in list(space._split_candidates):
            if splits >= MAX_SPLITS_PER_PASS:
                break
            region = space.regions.get(rid)
            if region is None:
                continue
            if (
                region.size >= cfg.split_threshold
                and region.density >= cfg.max_density
                and space.write_ops
                - space._merge_ops.get(rid, -(cfg.evolve_interval * 4))
                >= cfg.evolve_interval * 4
            ):
                outcome = self.split_region(space, rid)
                if outcome is not None:
                    splits += 1
                    events.append(self.history[-1])
        space._split_candidates.clear()

        # --- 2) absorb tiny regions into their nearest neighbor -------- #
        merges = 0
        for rid in list(space._small_candidates):
            if merges >= MAX_MERGES_PER_PASS:
                break
            region = space.regions.get(rid)
            if region is None or region.size >= cfg.min_region_size:
                continue
            neighbor = self._closest_region(space, rid)
            if neighbor and neighbor.id != rid:
                outcome = self.merge_regions(space, rid, neighbor.id)
                if outcome is not None:
                    merges += 1
                    events.append(self.history[-1])
        space._small_candidates.clear()

        # --- 3) merge regions that drifted close together ------------- #
        for key in list(space._close_edge_keys):
            if merges >= MAX_MERGES_PER_PASS:
                break
            edge = space.region_edges.get(key)
            if edge is None:
                space._close_edge_keys.discard(key)
                continue
            a, b = edge.source, edge.target
            ra, rb = space.regions.get(a), space.regions.get(b)
            if ra is None or rb is None:
                space._close_edge_keys.discard(key)
                continue
            combined = ra.size + rb.size
            if (
                combined >= cfg.min_region_size
                and combined <= cfg.split_threshold * 2
            ):
                outcome = self.merge_regions(space, a, b)
                if outcome is not None:
                    merges += 1
                    events.append(self.history[-1])
        # keep the merge-hysteresis bookkeeping bounded
        if len(space._merge_ops) > 64:
            cutoff = space.write_ops - cfg.evolve_interval * 8
            space._merge_ops = {
                k: v for k, v in space._merge_ops.items() if v >= cutoff
            }
        return events

    def _closest_region(self, space: "SpatialMemorySpace", region_id: str) -> Optional[Region]:
        region = space.regions.get(region_id)
        if region is None or region.centroid is None:
            return None
        ids, matrix = space._centroid_matrix()
        if not ids:
            return None
        diffs = matrix - region.centroid.reshape(1, -1)
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)
        for idx in order:
            other_id = ids[int(idx)]
            if other_id == region_id:
                continue
            return space.regions[other_id]
        return None

    # ------------------------------------------------------------------ #
    # helpers used by the space during insert/remove
    # ------------------------------------------------------------------ #
    def absorb_member(self, space: "SpatialMemorySpace", memory_id: str, vector: np.ndarray) -> str:
        """Attach a memory to the best existing region, or create a new one.

        Membership is density-based: the memory joins the nearest region
        when it is close enough to the region centroid (cosine >= the
        configured join threshold); otherwise it starts a new region.
        """
        if not space.regions:
            region = self.create_region(space, [memory_id])
            return region.id
        best_id, _best_dist = space.nearest_region(vector)
        region = space.regions[best_id]
        if region.centroid is not None:
            join_cos = cosine_similarity(vector, region.centroid)
            if join_cos >= self.config.min_join_cosine:
                # exact incremental centroid: (n*c + v) / (n+1) == mean of
                # the new member set (float drift ~1e-17, bounded by the
                # periodic full recompute at evolution passes)
                n = region.size
                region.member_ids.add(memory_id)
                region.centroid = (n * region.centroid + vector) / (n + 1)
                region.mark_stale()
                space._touch(best_id)
                self.note_region(space, best_id)
                return best_id
        new_region = self.create_region(space, [memory_id])
        return new_region.id
