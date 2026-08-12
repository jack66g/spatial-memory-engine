"""Memory Visualization.

Projects the high-dimensional embedding space to 2D and draws:

    - every memory node (colored by region)
    - region centroids (large markers)
    - region boundaries (convex hulls)
    - region neighbor lines
    - optional memory-graph edges

Projection methods: pca (default), random. Saves a PNG file.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sme.config import VisualizationConfig
from sme.utils import to_array

try:  # matplotlib is optional at import time
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
except ImportError:  # pragma: no cover
    plt = None
    MplPolygon = None


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def project_2d(
    vectors: list[np.ndarray],
    method: str = "pca",
    seed: int = 42,
) -> np.ndarray:
    """Project vectors to 2D. PCA via SVD (pure numpy) or random projection."""
    if not vectors:
        return np.zeros((0, 2))
    mat = np.stack([to_array(v) for v in vectors])
    dim = mat.shape[1]
    if dim == 2:
        return mat
    if mat.shape[0] == 1:
        # a single point has no PCA axis - embed in the first two dims
        out = np.zeros((1, 2))
        out[0, : min(2, dim)] = mat[0, : min(2, dim)]
        return out
    if method == "random":
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((dim, 2))
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        return mat @ proj
    # PCA
    centered = mat - mat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


class _ProjectionBasis:
    """A reusable PCA/random projection frame.

    The node coordinates and the region centroids must live in the SAME
    frame, otherwise the drawn stars/edges drift away from their clusters
    when the embedding dimension is > 2 (each standalone project_2d call
    computes its own SVD axes). Fit the basis on the node vectors once and
    transform every related point (centroids) through it.
    """

    def __init__(self, method: str, mean, vt, proj) -> None:
        self.method = method
        self.mean = mean
        self.vt = vt
        self.proj = proj

    def transform(self, vectors) -> np.ndarray:
        mat = np.stack([to_array(v) for v in vectors])
        dim = mat.shape[1]
        if dim == 2:
            return mat
        if self.method == "random":
            return mat @ self.proj
        return (mat - self.mean) @ self.vt[:2].T


def fit_project_2d(
    vectors: list[np.ndarray],
    method: str = "pca",
    seed: int = 42,
) -> tuple[np.ndarray, _ProjectionBasis | None]:
    """Project node vectors and return ``(coords, basis)``.

    ``basis`` is None when the input is degenerate (empty / single point /
    already 2D); callers then fall back to the raw first-two-dimensions
    embedding for related points, which matches the node coordinates.
    """
    if not vectors:
        return np.zeros((0, 2)), None
    mat = np.stack([to_array(v) for v in vectors])
    dim = mat.shape[1]
    if dim == 2:
        return mat, None
    if mat.shape[0] == 1:
        out = np.zeros((1, 2))
        out[0, : min(2, dim)] = mat[0, : min(2, dim)]
        return out, None
    if method == "random":
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((dim, 2))
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        return mat @ proj, _ProjectionBasis("random", None, None, proj)
    mean = mat.mean(axis=0)
    centered = mat - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, _ProjectionBasis("pca", mean, vt, None)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain convex hull. Returns hull vertices."""
    if len(points) < 3:
        return points
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    pts = [np.asarray(p, dtype=float) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def visualize(
    engine: object,
    output_path: str = "memory_space.png",
    config: Optional[VisualizationConfig] = None,
    show_graph: bool = True,
) -> str:
    """Render the whole memory space to a PNG file. Returns the file path."""
    if plt is None:  # pragma: no cover
        raise ImportError("matplotlib is required for visualization")
    cfg = config or engine.config.visualization

    memories = [m for m in engine.memories.values() if not m.archived]
    vectors = [m.embedding for m in memories if m.embedding is not None]
    ids = [m.id for m in memories if m.embedding is not None]
    if not vectors:
        raise ValueError("no memories with embeddings to visualize")

    coords, basis = fit_project_2d(vectors, cfg.projection, engine.config.seed or 42)
    region_ids = [engine.space.region_for(mid) or "none" for mid in ids]

    # unique region ids, stable colors
    unique = sorted({r for r in region_ids})
    color_map = plt.get_cmap(cfg.color_map)
    colors = {
        rid: color_map(i / max(1, len(unique) - 1)) for i, rid in enumerate(unique)
    }

    fig, ax = plt.subplots(figsize=cfg.figure_size, dpi=cfg.dpi)

    # region hulls
    for rid in unique:
        pts = coords[[i for i, r in enumerate(region_ids) if r == rid]]
        if len(pts) >= 3:
            hull = _convex_hull(pts)
            ax.add_patch(
                MplPolygon(
                    hull,
                    closed=True,
                    facecolor=colors[rid],
                    alpha=0.12,
                    edgecolor=colors[rid],
                    linewidth=1.2,
                )
            )

    # memory nodes
    for rid in unique:
        mask = [i for i, r in enumerate(region_ids) if r == rid]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            color=colors[rid],
            alpha=0.75,
            label=rid,
            edgecolors="none",
        )

    # region centroids (projected in the SAME frame as the nodes)
    for region in engine.space.regions.values():
        if region.centroid is not None and region.size:
            c = (
                basis.transform([region.centroid])[0]
                if basis is not None
                else project_2d([region.centroid], cfg.projection, engine.config.seed or 42)[0]
            )
            ax.scatter(c[0], c[1], s=90, marker="*", color="black", zorder=5)

    # region neighbor lines
    for edge in engine.space.region_edges.values():
        a = engine.space.regions.get(edge.source)
        b = engine.space.regions.get(edge.target)
        if a is None or b is None or a.centroid is None or b.centroid is None:
            continue
        ca = (
            basis.transform([a.centroid])[0]
            if basis is not None
            else project_2d([a.centroid], cfg.projection, engine.config.seed or 42)[0]
        )
        cb = (
            basis.transform([b.centroid])[0]
            if basis is not None
            else project_2d([b.centroid], cfg.projection, engine.config.seed or 42)[0]
        )
        ax.plot([ca[0], cb[0]], [ca[1], cb[1]], color="grey", alpha=0.35, linewidth=0.8)

    # memory graph edges (optional)
    if show_graph and cfg.show_graph:
        coord = {mid: c for mid, c in zip(ids, coords)}
        for edge in engine.graph.edges:
            if edge.source in coord and edge.target in coord:
                ax.plot(
                    [coord[edge.source][0], coord[edge.target][0]],
                    [coord[edge.source][1], coord[edge.target][1]],
                    color="orange",
                    alpha=0.25,
                    linewidth=0.6,
                )

    ax.set_title(
        f"Spatial Memory Space - {len(ids)} memories, "
        f"{len(engine.space.regions)} regions (projection: {cfg.projection})"
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
