"""Optional re-ranking adapter (iteration 2.3).

A bge-reranker style cross-encoder scores the retrieved top-N hits against
the query and re-orders them, improving precision on Chinese knowledge-base
queries. It is a *pure optional module*: ``rerank.enabled`` defaults to
False and the lazy import keeps the engine fully offline when disabled.

Hook: ``engine.search`` applies it after the v1 two-stage retrieval and the
v2 search hooks, before returning the final list. The underlying scorer
follows the ``ranking.register_scorer`` philosophy - any cross-encoder with
a ``predict([(query, doc)]) -> scores`` interface can be swapped in.
"""

from __future__ import annotations

from typing import Any

from sme.config import RerankConfig


class Reranker:
    def __init__(self, config: RerankConfig) -> None:
        self.config = config
        self._model = None
        self._name = ""

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _get_model(self):
        """Lazy-load the cross-encoder model (imported on first use)."""
        if self._model is None or self._name != self.config.model:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "rerank requires sentence-transformers: "
                    "pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(self.config.model)
            self._name = self.config.model
        return self._model

    # ------------------------------------------------------------------ #
    def rerank(self, query: str, hits: list[Any], top_n: int = 0) -> list[Any]:
        """Re-order ``hits`` by the cross-encoder scores (descending)."""
        if not self.enabled or not hits or not query:
            return hits
        top_n = top_n or self.config.top_n
        pool = hits[:top_n]
        pairs = [(query, h.memory.text) for h in pool]
        if not pairs:
            return hits
        try:
            scores = self._get_model().predict(
                pairs, convert_to_numpy=True
            )
        except Exception:  # noqa: BLE001 - rerank is best-effort
            return hits
        scored = sorted(
            zip(pool, scores), key=lambda pair: float(pair[1]), reverse=True
        )
        reranked = [h for h, _ in scored]
        # keep the untouched tail (beyond top_n) in its original order
        return reranked + list(hits[top_n:])
