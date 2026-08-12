"""Spatial Memory Space.

The embedding space of the whole memory system. Maintains:

    - memory nodes (memory_id -> embedding vector)
    - regions (density-based, dynamic, irregular)
    - region graph (neighbor edges)
    - membership bookkeeping and automatic evolution triggers

Inserting a memory either joins the nearest compatible region or spawns a
new region. Regions evolve automatically (split / merge) as the space
grows, so dense topics become well-separated regions over time.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from sme.config import RegionConfig
from sme.index import ANNIndex
from sme.models import Region, RegionEdge, RegionHit
from sme.space.region_manager import RegionManager
from sme.utils import euclidean_distance


class SpatialMemorySpace:
    """The continuous embedding space where all memories live."""

    def __init__(self, config: RegionConfig, dim: int = 64) -> None:
        self.config = config
        self.dim = dim
        self.vectors: dict[str, np.ndarray] = {}
        self.regions: dict[str, Region] = {}
        self.region_edges: dict[frozenset, RegionEdge] = {}
        self._membership: dict[str, Optional[str]] = {}
        # optional hook fired on every membership change so callers can keep
        # their own region_id copies in sync (e.g. Memory.region_id)
        self._membership_hook: Optional[Callable[[str, Optional[str]], None]] = None
        self._dirty: set[str] = set()
        self._centroids: dict[str, np.ndarray] = {}
        self._centroids_dirty = True
        # evolution candidate tracking (keeps evolution passes cheap)
        self._split_candidates: set[str] = set()
        self._small_candidates: set[str] = set()
        self._close_edge_keys: set[frozenset] = set()
        self._merge_recently: set[str] = set()
        self._merge_ops: dict[str, int] = {}
        self.manager = RegionManager(config)
        self.write_ops = 0
        self._bulk = False
        self._region_ann: Optional[ANNIndex] = None

    def set_bulk(self, bulk: bool) -> None:
        """Suspend/resume auto-evolution (used for batch writes)."""
        self._bulk = bool(bulk)

    def set_membership_hook(
        self, hook: Optional[Callable[[str, Optional[str]], None]]
    ) -> None:
        """Register a callback fired on every membership change."""
        self._membership_hook = hook

    def _set_membership(self, memory_id: str, region_id: Optional[str]) -> None:
        self._membership[memory_id] = region_id
        if self._membership_hook is not None:
            self._membership_hook(memory_id, region_id)

    # ------------------------------------------------------------------ #
    # basic state
    # ------------------------------------------------------------------ #
    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def node_count(self) -> int:
        return len(self.vectors)

    def region_for(self, memory_id: str) -> Optional[str]:
        return self._membership.get(memory_id)

    def _touch(self, region_id: str) -> None:
        self._dirty.add(region_id)
        self._centroids_dirty = True

    # ------------------------------------------------------------------ #
    def _centroid_matrix(self) -> tuple[list[str], np.ndarray]:
        """(region ids, matrix of centroids) - cached until regions change."""
        if self._centroids_dirty or len(self._centroids) != len(self.regions):
            self._centroids = {
                rid: r.centroid
                for rid, r in self.regions.items()
                if r.centroid is not None
            }
            self._centroids_dirty = False
        if not self._centroids:
            return [], np.zeros((0, self.dim))
        return list(self._centroids), np.stack(list(self._centroids.values()))

    # ------------------------------------------------------------------ #
    def nearest_region(self, vector: np.ndarray) -> tuple[str, float]:
        """(region_id, distance) of the region nearest to `vector`.

        Uses the hnswlib ANN index once the space grows past
        ``ann_min_regions`` (with lazy centroid sync). To keep the result
        exact, the ANN returns a small beam of candidates (top-8) and they
        are re-ranked with exact distances - a few microseconds of extra
        work that eliminates approximate top-1 errors. Below the threshold
        a pure vectorized exact scan is used.
        """
        ann = self._ensure_region_ann()
        if ann is not None:
            beam = ann.query(vector, 8)
            if beam:
                # every beam entry (including the ANN top-1) is validated for
                # existence and re-ranked with the same exact distance metric:
                # the ANN may return a stale/deleted region id (count-based
                # rebuilds miss delete+create cycles) and its cosine-space
                # distance must never be compared with the euclidean one.
                best_key, best_dist = None, float("inf")
                for key, _ in beam:
                    region = self.regions.get(key)
                    if region is None or region.centroid is None:
                        continue
                    dist = euclidean_distance(vector, region.centroid)
                    if dist < best_dist:
                        best_dist = dist
                        best_key = key
                if best_key is not None:
                    return best_key, best_dist
        ids, matrix = self._centroid_matrix()
        if not ids:
            return "", float("inf")
        diffs = matrix - vector.reshape(1, -1)
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        return ids[idx], float(dists[idx])

    def _ensure_region_ann(self) -> Optional[ANNIndex]:
        """Build / sync the ANN index over region centroids (lazy)."""
        cfg = self.config
        if not cfg.ann_enabled or self.region_count < cfg.ann_min_regions:
            return None
        if self._region_ann is None:
            self._region_ann = ANNIndex(dim=self.dim, metric="cosine")
        if len(self._region_ann) != self.region_count:
            # structural change (creates/deletes) - full sync is cheapest
            self._region_ann.rebuild(
                {
                    rid: r.centroid
                    for rid, r in self.regions.items()
                    if r.centroid is not None
                }
            )
            self._dirty.clear()
        elif self._dirty:
            for rid in self._dirty:
                region = self.regions.get(rid)
                if region is not None and region.centroid is not None:
                    self._region_ann.add(rid, region.centroid)
            self._dirty.clear()
        return self._region_ann

    def _prune_edges_for(self, region_id: str) -> None:
        self.region_edges = {
            k: e
            for k, e in self.region_edges.items()
            if region_id not in (e.source, e.target)
        }
        self._close_edge_keys = {
            k for k in self._close_edge_keys if region_id not in k
        }

    # ------------------------------------------------------------------ #
    # insert / remove
    # ------------------------------------------------------------------ #
    def insert(self, memory_id: str, vector: np.ndarray) -> str:
        """Add a memory node to the space. Returns its region id."""
        arr = np.asarray(vector, dtype=np.float64).reshape(-1)
        self.vectors[memory_id] = arr
        region_id = self.manager.absorb_member(self, memory_id, arr)
        self._set_membership(memory_id, region_id)
        self.write_ops += 1
        self._maybe_evolve()
        return region_id

    def remove(self, memory_id: str) -> None:
        """Remove a memory node from the space (regions stay consistent)."""
        if memory_id not in self.vectors:
            return
        del self.vectors[memory_id]
        region_id = self._membership.pop(memory_id, None)
        if region_id is None:
            return
        if self._membership_hook is not None:
            self._membership_hook(memory_id, None)
        region = self.regions.get(region_id)
        if region is None:
            return
        region.member_ids.discard(memory_id)
        if region.size == 0:
            self.manager.delete_region(self, region_id)
        else:
            self.manager.refresh_geometry(self, region_id)
            self._maybe_evolve()

    def _maybe_evolve(self) -> None:
        if self._bulk:
            return
        if not self.config.auto_evolve:
            return
        if self.write_ops % self.config.evolve_interval == 0:
            self.manager.evolution_pass(self)

    # ------------------------------------------------------------------ #
    def sync_geometry(self) -> None:
        """Exact recompute of deferred region geometry (radius/density/bbox).

        The write path keeps centroids exact incrementally and defers the
        rest; every consumer that needs radius/density/bbox calls this once
        so the values are exactly what a full recompute would produce.
        """
        for region in self.regions.values():
            region.refresh_if_stale(self.vectors, self.dim)

    # ------------------------------------------------------------------ #
    # region retrieval (stage 1)
    # ------------------------------------------------------------------ #
    def query_regions(self, query_vec: np.ndarray, top_k: int) -> list[RegionHit]:
        """Rank regions by relevance to the query vector.

        Two-phase ranking so that small, focused regions are not starved by
        large regions whose centroids act like an "average gravity":

        phase A - coarse: rank ALL regions by cosine(query, centroid);
        phase B - refine: re-rank the top (top_k*4) regions by their
                  maximum member cosine, which catches a lone memory that
                  matches the query even inside a mediocre region.
        """
        q = query_vec.reshape(1, -1)
        q = q / np.clip(np.linalg.norm(q), 1e-12, None)

        coarse: list[RegionHit] = []
        for region in self.regions.values():
            if region.centroid is None or region.size == 0:
                continue
            c = region.centroid.reshape(1, -1)
            cosine = float(
                (c @ q.T).item()
                / max(np.linalg.norm(c) * np.linalg.norm(q), 1e-12)
            )
            distance = euclidean_distance(query_vec, region.centroid)
            score = 0.6 * (cosine + 1.0) / 2.0 + 0.4 * (1.0 / (1.0 + distance))
            coarse.append(RegionHit(region=region, score=score, distance=distance))
        coarse.sort(key=lambda hit: hit.score, reverse=True)

        refine_n = max(top_k * 4, 12)
        refine_n = min(refine_n, len(coarse))
        refined: list[RegionHit] = []
        for hit in coarse[:refine_n]:
            region = hit.region
            member_vecs = [
                self.vectors[mid]
                for mid in region.member_ids
                if mid in self.vectors
            ]
            max_cos = hit.score  # centroid-based score is the fallback
            if member_vecs:
                mat = np.stack(member_vecs)
                mat = mat / np.clip(
                    np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None
                )
                max_cos = float((mat @ q.T).reshape(-1).max())
            # blend centroid relevance with best-member relevance
            final_score = 0.4 * hit.score + 0.6 * max_cos
            refined.append(
                RegionHit(
                    region=region,
                    score=final_score,
                    distance=hit.distance,
                )
            )
        refined.sort(key=lambda hit: hit.score, reverse=True)
        return refined[:top_k]

    def candidates_in_region(
        self, region_id: str, exclude: set[str] | None = None
    ) -> list[str]:
        region = self.regions.get(region_id)
        if region is None:
            return []
        exclude = exclude or set()
        return [mid for mid in region.member_ids if mid not in exclude]

    # ------------------------------------------------------------------ #
    # serialization
    # ------------------------------------------------------------------ #
    def load_state(self, state: dict[str, Any], regions: list[Region], edges: list[RegionEdge]) -> None:
        self.dim = state.get("dim", self.dim)
        self.write_ops = state.get("write_ops", 0)
        self._membership = dict(state.get("membership", {}))
        self.vectors = {
            k: np.asarray(v, dtype=np.float64) for k, v in state.get("vectors", {}).items()
        }
        self.regions = {r.id: r for r in regions}
        self.region_edges = {frozenset((e.source, e.target)): e for e in edges}
        self._dirty = set()
        self._split_candidates = set()
        self._small_candidates = set()
        self._close_edge_keys = set()
        self._merge_recently = set()
        self._merge_ops: dict[str, int] = {}
        self._centroids_dirty = True
        self._region_ann = None
