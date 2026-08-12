"""ANN vector index: hnswlib wrapper with an exact-scan fallback.

- When hnswlib is installed and the index scale justifies it, nearest
  neighbor queries are O(log n) instead of O(n x dim).
- Deletion is supported via markDelete + lazy full rebuild on request.
- When hnswlib is unavailable, falls back to a cached numpy full scan with
  identical semantics.
"""

from __future__ import annotations

import numpy as np

try:
    import hnswlib  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    hnswlib = None

_DEFAULT_MAX_ELEMENTS = 2_000_000


class ANNIndex:
    """Keyed approximate-nearest-neighbor index (str keys -> vectors)."""

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        max_elements: int = _DEFAULT_MAX_ELEMENTS,
        ef_construction: int = 200,
        ef: int = 256,
        m: int = 16,
        use_hnsw: bool | None = None,
    ) -> None:
        """`use_hnsw=None` auto-selects hnswlib when it is installed."""
        self.dim = dim
        self.metric = metric
        self.use_hnsw = (
            hnswlib is not None if use_hnsw is None else bool(use_hnsw)
        )
        self._vectors: dict[str, np.ndarray] = {}
        self._labels: dict[str, int] = {}
        self._label_of: dict[int, str] = {}
        # monotonic label counter: labels are NEVER recycled after removal.
        # Reusing len(self._vectors) as the next label collides once elements
        # are deleted (the count shrinks), mapping two keys to one hnswlib
        # label and later triggering "element is already deleted" on removal.
        self._next_label = 0
        self._idx = None
        if self.use_hnsw and hnswlib is not None:
            self._idx = hnswlib.Index(space=metric, dim=dim)
            self._idx.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=m,
            )
            self._idx.set_ef(ef)

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, key: str) -> bool:
        return key in self._vectors

    # ------------------------------------------------------------------ #
    def add(self, key: str, vector: np.ndarray) -> None:
        """Add or update one vector (hnswlib has no in-place update)."""
        if key in self._vectors:
            self.remove(key)
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"dim mismatch: {arr.shape[0]} != {self.dim}")
        label = self._next_label
        self._next_label += 1
        self._vectors[key] = arr
        self._labels[key] = label
        self._label_of[label] = key
        if self._idx is not None:
            self._idx.add_items([arr], [label])

    def remove(self, key: str) -> None:
        if key not in self._vectors:
            return
        label = self._labels.pop(key)
        self._label_of.pop(label, None)
        self._vectors.pop(key, None)
        if self._idx is not None:
            # hnswlib >= 0.8 uses mark_deleted; older builds markDelete
            marker = getattr(self._idx, "mark_deleted", None) or getattr(
                self._idx, "markDelete"
            )
            marker(label)

    # ------------------------------------------------------------------ #
    def query(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return [(key, distance)] for the nearest `top_k` vectors.

        For the cosine metric the returned distance is (1 - cosine).
        """
        if not self._vectors:
            return []
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self._idx is not None:
            k = min(top_k, len(self._vectors))
            labels, dists = self._idx.knn_query([arr], k=k)
            out: list[tuple[str, float]] = []
            for label, dist in zip(labels[0], dists[0]):
                key = self._label_of.get(int(label))
                if key is not None:
                    out.append((key, float(dist)))
            return out
        # exact fallback with metric-consistent distances
        keys = list(self._vectors)
        mat = np.stack([self._vectors[k] for k in keys])
        if self.metric == "cosine":
            # cosine distance = 1 - cosine similarity (matches hnswlib)
            mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
            arrn = arr / np.clip(np.linalg.norm(arr), 1e-12, None)
            dists = 1.0 - (mat @ arrn)
        else:
            diffs = mat - arr.reshape(1, -1)
            dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)[:top_k]
        return [(keys[int(i)], float(dists[int(i)])) for i in order]

    # ------------------------------------------------------------------ #
    def rebuild(self, vectors: dict[str, np.ndarray]) -> None:
        """Rebuild the index from scratch (handles deletions in bulk)."""
        self._vectors = {
            k: np.asarray(v, dtype=np.float32).reshape(-1)
            for k, v in vectors.items()
            if v is not None
        }
        self._labels = {k: i for i, k in enumerate(self._vectors)}
        self._label_of = {i: k for k, i in self._labels.items()}
        self._next_label = len(self._vectors)
        if self._idx is None:
            return
        self._idx = hnswlib.Index(space=self.metric, dim=self.dim)
        self._idx.init_index(
            max_elements=max(_DEFAULT_MAX_ELEMENTS, len(self._vectors) * 2 + 100),
            ef_construction=200,
            M=16,
        )
        self._idx.set_ef(64)
        if self._vectors:
            labels = np.arange(len(self._vectors), dtype=np.intp)
            mat = np.stack(list(self._vectors.values()))
            self._idx.add_items(mat, labels)
