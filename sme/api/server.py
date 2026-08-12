"""REST API for the Spatial Memory Engine.

FastAPI server exposing the full memory API:

    POST   /memories                    AddMemory
    POST   /memories/batch              Add multiple memories
    GET    /memories/{id}               Get one memory
    PATCH  /memories/{id}               UpdateMemory
    DELETE /memories/{id}               DeleteMemory
    POST   /memories/search             SearchMemory (two-stage + hybrid)
    POST   /memories/{id}/hit           reinforce a hit
    POST   /memories/{id}/archive       archive
    POST   /memories/{id}/restore       restore
    GET    /regions                     list regions
    POST   /regions/search              SearchRegion
    GET    /stats                       MemoryStats + RegionStats
    POST   /consolidate                 run consolidation
    POST   /compress                    run compression
    GET    /graph                       memory graph edges
    GET    /visualize                   render PNG
    GET    /export                      Export (all memories as JSON)
    POST   /import                      Import memories
    GET    /health                      health check

Interactive docs at /docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine
from sme.retrieval import SearchQuery

try:
    from sme import __version__ as SME_VERSION
except ImportError:  # pragma: no cover
    SME_VERSION = "1.2.0"

APP_TITLE = "Spatial Memory Engine API"
APP_VERSION = SME_VERSION


# --------------------------------------------------------------------------- #
# request / response models
# --------------------------------------------------------------------------- #
class AddMemoryRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    importance: float = 0.5
    source: str = "user"
    link_to: Optional[str] = None
    link_kind: str = "reference"


class UpdateMemoryRequest(BaseModel):
    text: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = None
    importance: Optional[float] = None
    weight: Optional[float] = None
    summary: Optional[str] = None


class SearchRequest(BaseModel):
    text: str
    top_k: int = 10
    top_regions: Optional[int] = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    tags: Optional[list[str]] = None
    region_retrieval: Optional[str] = None
    include_archived: bool = False
    graph_expand: int = 0


class RegionSearchRequest(BaseModel):
    text: str
    top_k: int = 5


class ImportRequest(BaseModel):
    memories: list[dict[str, Any]]


class LinkRequest(BaseModel):
    a: str
    b: str
    kind: str = "reference"
    weight: float = 1.0
    note: str = ""


# --------------------------------------------------------------------------- #
# app factory
# --------------------------------------------------------------------------- #
def create_app(engine: SpatialMemoryEngine) -> FastAPI:
    app = FastAPI(title=APP_TITLE, version=APP_VERSION)
    auth_token = getattr(engine.config.api, "auth_token", "") or ""

    if auth_token:
        # Bearer 鉴权：auth_token 非空即启用；健康检查与交互文档放行
        open_paths = {"/health", "/docs", "/redoc", "/openapi.json"}

        @app.middleware("http")
        async def _auth_middleware(request, call_next):
            if request.url.path not in open_paths:
                if request.headers.get("Authorization") != f"Bearer {auth_token}":
                    return JSONResponse(
                        status_code=401, content={"detail": "unauthorized"}
                    )
            return await call_next(request)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "memories": len(engine.memories)}

    # ------------------------- memories ------------------------------- #
    @app.post("/memories", response_model=dict)
    def add_memory(req: AddMemoryRequest) -> dict:
        memory = engine.add(
            text=req.text,
            metadata=req.metadata,
            tags=req.tags,
            importance=req.importance,
            source=req.source,
            link_to=req.link_to,
            link_kind=req.link_kind,
        )
        return memory.to_dict()

    @app.post("/memories/batch", response_model=dict)
    def add_batch(req: ImportRequest) -> dict:
        # route through engine.add so the v2 write pipeline applies exactly
        # like engine.add_many (extraction/factversion/qapair are honored)
        added = 0
        for item in req.memories:
            text = (item or {}).get("text")
            if not text or not str(text).strip():
                continue
            mem = engine.add(
                text=str(text),
                metadata=item.get("metadata", {}),
                tags=item.get("tags", []),
                importance=item.get("importance", 0.5),
                source=item.get("source", "user"),
            )
            # the v2 pipeline may drop the item (extraction noise etc.):
            # only count memories that actually entered the store
            if mem.id in engine.memories:
                added += 1
        return {
            "added": added,
            "dropped": len(req.memories) - added,
            "total": len(req.memories),
        }

    @app.get("/memories/{memory_id}", response_model=dict)
    def get_memory(memory_id: str) -> dict:
        memory = engine.get(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return memory.to_dict()

    @app.patch("/memories/{memory_id}", response_model=dict)
    def update_memory(memory_id: str, req: UpdateMemoryRequest) -> dict:
        try:
            memory = engine.update(
                memory_id,
                text=req.text,
                metadata=req.metadata,
                tags=req.tags,
                importance=req.importance,
                weight=req.weight,
                summary=req.summary,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return memory.to_dict()

    @app.delete("/memories/{memory_id}")
    def delete_memory(memory_id: str) -> dict:
        if not engine.delete(memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"deleted": memory_id}

    @app.post("/memories/{memory_id}/hit", response_model=dict)
    def hit_memory(memory_id: str) -> dict:
        result = engine.reinforce(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return result

    @app.post("/memories/{memory_id}/archive")
    def archive_memory(memory_id: str) -> dict:
        if not engine.archive(memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"archived": memory_id}

    @app.post("/memories/{memory_id}/restore")
    def restore_memory(memory_id: str) -> dict:
        if not engine.restore(memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"restored": memory_id}

    # ------------------------- search --------------------------------- #
    @app.post("/memories/search", response_model=dict)
    def search_memories(req: SearchRequest) -> dict:
        query = SearchQuery(
            text=req.text,
            top_k=req.top_k,
            top_regions=req.top_regions,
            metadata_filters=req.metadata_filters,
            tags=req.tags,
            include_archived=req.include_archived,
            region_retrieval=req.region_retrieval,
            graph_expand=req.graph_expand,
        )
        hits = engine.search(query)
        return {
            "query": req.text,
            "count": len(hits),
            "results": [h.to_dict() for h in hits],
        }

    # ------------------------- regions -------------------------------- #
    @app.get("/regions", response_model=dict)
    def list_regions() -> dict:
        regions = []
        for region in engine.space.regions.values():
            regions.append(region.to_dict())
        return {"count": len(regions), "regions": regions}

    @app.post("/regions/search", response_model=dict)
    def search_regions(req: RegionSearchRequest) -> dict:
        hits = engine.search_regions(req.text, req.top_k)
        return {"count": len(hits), "results": [h.to_dict() for h in hits]}

    @app.post("/regions/evolve")
    def evolve_regions() -> dict:
        events = engine.region_manager.evolution_pass(engine.space)
        return {"events": [e.__dict__ for e in events]}

    # ------------------------- stats ----------------------------------- #
    @app.get("/stats", response_model=dict)
    def stats() -> dict:
        return engine.engine_stats()

    # ------------------------- consolidation --------------------------- #
    @app.post("/consolidate", response_model=dict)
    def consolidate() -> dict:
        created = engine.consolidate()
        return {"created": [m.to_dict() for m in created]}

    @app.post("/compress", response_model=dict)
    def compress() -> dict:
        created = engine.compress()
        return {"created": [m.to_dict() for m in created]}

    # ------------------------- graph ----------------------------------- #
    @app.get("/graph", response_model=dict)
    def graph() -> dict:
        return {
            "edge_count": len(engine.graph),
            "edges": [e.to_dict() for e in engine.graph.edges],
        }

    @app.post("/graph/link")
    def graph_link(req: LinkRequest) -> dict:
        ok = engine.link(req.a, req.b, req.kind, req.weight, req.note)
        if not ok:
            raise HTTPException(status_code=404, detail="one or both memories missing")
        return {"linked": True}

    # ------------------------- export / import ------------------------- #
    @app.get("/export")
    def export() -> dict:
        return {
            "memories": [m.to_dict() for m in engine.memories.values()],
            "graph": engine.graph.to_dict(),
            "stats": engine.engine_stats(),
        }

    @app.post("/import", response_model=dict)
    def import_memories(req: ImportRequest) -> dict:
        count = engine.import_memories(req.memories)
        return {"imported": count}

    @app.post("/import/file", response_model=dict)
    def import_file(file: UploadFile = File(...)) -> dict:
        try:
            content = file.file.read()
            data = json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid file: {exc}") from exc
        items = data.get("memories", data if isinstance(data, list) else [])
        count = engine.import_memories(items)
        return {"imported": count}

    # ------------------------- visualize ------------------------------- #
    @app.get("/visualize")
    def visualize() -> FileResponse:
        path = os.path.join(tempfile.gettempdir(), "sme_space.png")
        engine.visualize(path)
        return FileResponse(path, media_type="image/png")

    # ------------------------- v2 modules ------------------------------ #
    @app.get("/facts")
    def facts() -> dict:
        """Module 03 - knowledge-graph entities & relations (enabled only)."""
        fg = getattr(engine, "factgraph", None)
        if fg is None or not fg.enabled:
            return {"enabled": False, "entities": [], "relations": []}
        return {
            "enabled": True,
            "stats": fg.stats(),
            "entities": [e.to_dict() for e in fg.entities.values()],
            "relations": [r.to_dict() for r in fg.relations],
        }

    @app.post("/facts/multi_hop")
    def facts_multi_hop(req: SearchRequest) -> dict:
        """Module 03 - multi-hop graph query over the fact graph."""
        fg = getattr(engine, "factgraph", None)
        if fg is None or not fg.enabled:
            return {"enabled": False, "results": []}
        found = fg.find_entities(req.text)
        hops = fg.multi_hop([e.id for e in found])
        results = []
        for eid, depth in hops.items():
            ent = fg.entities.get(eid)
            if ent is not None:
                results.append({"entity": ent.name, "kind": ent.kind, "depth": depth})
        return {"enabled": True, "entities": len(found), "results": results}

    @app.get("/profile")
    def profile() -> dict:
        """Module 04 - user profile facts & snapshots (enabled only)."""
        prof = getattr(engine, "profile", None)
        if prof is None or not prof.enabled:
            return {"enabled": False, "profile_facts": [], "snapshots": {}}
        return {
            "enabled": True,
            "stats": prof.stats(),
            "profile_facts": [m.to_dict() for m in prof.profile_memories(engine.memories)],
            "snapshots": prof.snapshots,
        }

    @app.get("/qapairs")
    def qapairs() -> dict:
        """Module 02 - stored question/answer pairs (enabled only)."""
        qa = getattr(engine, "qapair", None)
        if qa is None or not qa.enabled:
            return {"enabled": False, "count": 0, "pairs": []}
        return {
            "enabled": True,
            "count": qa.count(),
            "pairs": [p.to_dict() for p in qa.pairs],
        }

    @app.get("/metrics")
    def metrics() -> dict:
        """Module 10 - observability summary (enabled only)."""
        tele = getattr(engine, "telemetry", None)
        if tele is None or not tele.enabled:
            return {"enabled": False, "summary": {}}
        return {"enabled": True, **tele.report(engine)}

    @app.get("/metrics/report")
    def metrics_report() -> FileResponse:
        """Module 10 - download the full telemetry report as JSON."""
        tele = getattr(engine, "telemetry", None)
        if tele is None or not tele.enabled:
            raise HTTPException(status_code=404, detail="telemetry disabled")
        import tempfile

        path = os.path.join(tempfile.gettempdir(), "sme_report.json")
        tele.export_json(path, engine)
        return FileResponse(path, media_type="application/json",
                            filename="sme_report.json")

    return app


def build_engine_from_env() -> SpatialMemoryEngine:
    """Create an engine from environment variables (or defaults).

    SME_LLM_BASE_URL / SME_LLM_API_KEY / SME_LLM_MODEL
    SME_EMBEDDING_PROVIDER / SME_EMBEDDING_MODEL / SME_EMBEDDING_DIM
    SME_CONFIG_PATH - JSON config file
    """
    config_path = os.environ.get("SME_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        return SpatialMemoryEngine(config_path=config_path)

    config = SMEConfig()
    base_url = os.environ.get("SME_LLM_BASE_URL")
    if base_url:
        config.llm.base_url = base_url
        config.llm.api_key = os.environ.get("SME_LLM_API_KEY", "")
        config.llm.model = os.environ.get("SME_LLM_MODEL", "gpt-4o-mini")
    config.api.auth_token = os.environ.get("SME_API_AUTH_TOKEN", "")
    provider = os.environ.get("SME_EMBEDDING_PROVIDER")
    if provider:
        config.embedding.provider = provider
        config.embedding.model = os.environ.get("SME_EMBEDDING_MODEL", "text-embedding-3-small")
        config.embedding.dim = int(os.environ.get("SME_EMBEDDING_DIM", "64"))
        config.embedding.base_url = os.environ.get("SME_EMBEDDING_BASE_URL", "")
        config.embedding.api_key = os.environ.get("SME_EMBEDDING_API_KEY", "")
    # build the engine from the fully-resolved config so the embedding
    # provider / dimension are created consistently from the env values
    return SpatialMemoryEngine(config)


def main() -> None:
    """Start the REST server: ``python -m sme.api``.

    Config via ``SME_CONFIG_PATH`` / env vars, or ``--config <path>``.
    Authentication: set ``api.auth_token`` (config) or ``SME_API_AUTH_TOKEN``.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="SME REST API 服务")
    parser.add_argument("--config", default="", help="配置文件路径（等价 SME_CONFIG_PATH）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.config:
        os.environ.setdefault("SME_CONFIG_PATH", args.config)
    engine = build_engine_from_env()
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
