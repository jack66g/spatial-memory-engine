"""ANN vector index package.

Optional hnswlib-based approximate nearest neighbor index with a pure-numpy
exact fallback, so the engine runs with or without the optional dependency.
"""

from sme.index.ann import ANNIndex

__all__ = ["ANNIndex"]
