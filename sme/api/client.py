"""Module 09 - MemoryClient: a Python SDK mirroring the engine API.

Thin httpx client over the REST API with the same method names as the
engine facade (add / search / update / delete / get / archive / restore /
export / stats / regions / consolidate / compress / graph / import ...).

Semantics mirror the engine:

- ``get`` returns None on 404 (engine.get returns None)
- ``delete`` / ``archive`` / ``restore`` return bool (engine returns bool)
- ``reinforce`` returns None on 404 (engine returns None)
- others raise ``httpx.HTTPStatusError`` on server errors
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx


class MemoryProvider(Protocol):
    """The unified memory interface (v2 模块设计 5.3).

    Implemented by ``sme.engine.SpatialMemoryEngine`` in-process and by
    ``MemoryClient`` over HTTP, so SDK and engine stay interchangeable.
    """

    def add(self, text: str, metadata: dict | None = None,
            tags: list | None = None, importance: float = 0.5,
            source: str = "user", **kwargs) -> Any: ...
    def search(self, query: str, top_k: int = 10, **kwargs) -> Any: ...
    def get(self, memory_id: str) -> Any | None: ...
    def update(self, memory_id: str, **fields: Any) -> Any: ...
    def delete(self, memory_id: str) -> bool: ...
    def archive(self, memory_id: str) -> bool: ...
    def restore(self, memory_id: str) -> bool: ...
    def stats(self) -> dict: ...
    def export(self) -> dict: ...


class MemoryClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000",
                 api_key: str = "", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # trust_env=False: 不读系统代理（局域网/本机服务直连）
        self._client = httpx.Client(timeout=timeout, trust_env=False)

    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> Any | None:
        """GET that maps 404 -> None (engine.get semantics)."""
        resp = self._client.request(
            "GET", f"{self.base_url}{path}", headers=self._headers()
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def _bool(self, method: str, path: str, **kw: Any) -> bool:
        """POST/DELETE that maps 404 -> False (engine bool semantics)."""
        resp = self._client.request(
            method, f"{self.base_url}{path}", headers=self._headers(), **kw
        )
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def _json(self, method: str, path: str, **kw: Any) -> Any:
        resp = self._client.request(
            method, f"{self.base_url}{path}", headers=self._headers(), **kw
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # core CRUD
    # ------------------------------------------------------------------ #
    def health(self) -> dict:
        return self._json("GET", "/health")

    def add(self, text: str, metadata: dict | None = None, tags: list | None = None,
            importance: float = 0.5, source: str = "user",
            link_to: str | None = None, link_kind: str = "reference") -> dict:
        return self._json("POST", "/memories", json={
            "text": text, "metadata": metadata or {}, "tags": tags or [],
            "importance": importance, "source": source,
            "link_to": link_to, "link_kind": link_kind,
        })

    def add_many(self, memories: list[dict]) -> dict:
        return self._json("POST", "/memories/batch", json={"memories": memories})

    def get(self, memory_id: str) -> dict | None:
        return self._get(f"/memories/{memory_id}")

    def update(self, memory_id: str, **fields: Any) -> dict:
        return self._json("PATCH", f"/memories/{memory_id}", json=fields)

    def delete(self, memory_id: str) -> bool:
        return self._bool("DELETE", f"/memories/{memory_id}")

    def archive(self, memory_id: str) -> bool:
        return self._bool("POST", f"/memories/{memory_id}/archive")

    def restore(self, memory_id: str) -> bool:
        return self._bool("POST", f"/memories/{memory_id}/restore")

    def reinforce(self, memory_id: str) -> dict | None:
        resp = self._client.request(
            "POST", f"{self.base_url}/memories/{memory_id}/hit",
            headers=self._headers(),
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # search / regions
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 10, graph_expand: int = 0,
               metadata_filters: dict | None = None, tags: list | None = None,
               include_archived: bool = False, top_regions: int | None = None,
               region_retrieval: str | None = None, ns: str | None = None) -> dict:
        filters = dict(metadata_filters or {})
        if ns is not None:
            # module 12 isolation over REST: same metadata tag as the engine
            filters["ns"] = ns
        return self._json("POST", "/memories/search", json={
            "text": query, "top_k": top_k, "graph_expand": graph_expand,
            "metadata_filters": filters, "tags": tags,
            "include_archived": include_archived, "top_regions": top_regions,
            "region_retrieval": region_retrieval,
        })

    def regions(self) -> dict:
        return self._json("GET", "/regions")

    def search_regions(self, text: str, top_k: int = 5) -> dict:
        return self._json("POST", "/regions/search", json={"text": text, "top_k": top_k})

    def evolve_regions(self) -> dict:
        return self._json("POST", "/regions/evolve")

    # ------------------------------------------------------------------ #
    # maintenance
    # ------------------------------------------------------------------ #
    def consolidate(self) -> dict:
        return self._json("POST", "/consolidate")

    def compress(self) -> dict:
        return self._json("POST", "/compress")

    def stats(self) -> dict:
        return self._json("GET", "/stats")

    # ------------------------------------------------------------------ #
    # graph / v2 modules
    # ------------------------------------------------------------------ #
    def graph(self) -> dict:
        return self._json("GET", "/graph")

    def graph_link(self, a: str, b: str, kind: str = "reference",
                   weight: float = 1.0, note: str = "") -> bool:
        return self._bool("POST", "/graph/link", json={
            "a": a, "b": b, "kind": kind, "weight": weight, "note": note,
        })

    def facts(self) -> dict:
        return self._json("GET", "/facts")

    def facts_multi_hop(self, text: str, top_k: int = 10) -> dict:
        return self._json("POST", "/facts/multi_hop", json={"text": text, "top_k": top_k})

    def qapairs(self) -> dict:
        return self._json("GET", "/qapairs")

    def profile(self) -> dict:
        return self._json("GET", "/profile")

    def metrics(self) -> dict:
        return self._json("GET", "/metrics")

    # ------------------------------------------------------------------ #
    # export / import / visualize
    # ------------------------------------------------------------------ #
    def export(self) -> dict:
        return self._json("GET", "/export")

    def import_memories(self, memories: list[dict]) -> dict:
        return self._json("POST", "/import", json={"memories": memories})

    def visualize(self, save_to: str) -> bytes:
        """Download the rendered memory-space PNG."""
        resp = self._client.request(
            "GET", f"{self.base_url}/visualize", headers=self._headers()
        )
        resp.raise_for_status()
        with open(save_to, "wb") as fh:
            fh.write(resp.content)
        return resp.content

    def close(self) -> None:
        self._client.close()
