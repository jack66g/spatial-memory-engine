"""Module 12 - Namespaces: multi-user / multi-scene isolation.

Each namespace is a partitioned view over the same engine: memories are
tagged with their namespace on write and filtered on read, so user A can
never retrieve user B's memories. The default namespace behaves exactly
like v1 (no filtering at all) - and with the module disabled there is no
namespace concept whatsoever.

Implementation note: the engine's core structures are untouched; the
namespace is carried as metadata ``ns`` and enforced by a thin view layer.
"""

from __future__ import annotations

from typing import Any, Optional

from sme.config import NamespaceConfig

NS_KEY = "ns"


class NamespaceView:
    """Isolated read/write view over one namespace of an engine."""

    def __init__(self, engine: Any, namespace: str, namespaces: "Namespaces") -> None:
        self.engine = engine
        self.namespace = namespace
        self._mgr = namespaces

    # ------------------------------------------------------------------ #
    def add(self, text: str, **kwargs: Any):
        return self.engine.add(text, ns=self.namespace, **kwargs)

    def search(self, query: Any, **kwargs: Any):
        return self.engine.search(query, ns=self.namespace, **kwargs)

    def memories(self) -> list:
        return [
            m for m in self.engine.memories.values()
            if m.metadata.get(NS_KEY) == self.namespace
        ]

    def stats(self) -> dict:
        mems = self.memories()
        return {"namespace": self.namespace, "memories": len(mems)}


class Namespaces:
    def __init__(self, config: NamespaceConfig) -> None:
        self.config = config
        self._views: dict[str, NamespaceView] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def view(self, engine: Any, namespace: Optional[str] = None) -> NamespaceView:
        ns = namespace or self.config.default_ns
        if ns not in self._views:
            self._views[ns] = NamespaceView(engine, ns, self)
        return self._views[ns]

    def list_namespaces(self) -> list[str]:
        return list(self._views)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "default_ns": self.config.default_ns,
            "views": {ns: v.stats() for ns, v in self._views.items()},
        }
