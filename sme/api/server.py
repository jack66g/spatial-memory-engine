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

Config center (Web 配置中心, zero-build frontend in ``sme/api/static``):

    GET    /                            config UI (index.html)
    GET    /static/*                    UI assets (app.css / app.js)
    GET    /config                      config descriptor (groups/items/presets)
    PUT    /config                      parse+validate -> hot rebuild (+ save)
    POST   /config/preset               apply a preset (same pipeline as PUT)
    POST   /config/check                config validation / connectivity check
    POST   /config/reset                delete the config file, rebuild defaults

Interactive docs at /docs (Swagger UI) and /redoc.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import threading
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from sme.config import SMEConfig
from sme.config_check import run_config_check
from sme.config_items import (
    ITEMS,
    ITEM_BY_PATH,
    PRESETS,
    PRESET_BY_KEY,
    ConfigItem,
    defaults_config,
    get_value,
    parse_value,
    save_config,
    set_value,
    validate_value,
)
from sme.engine import SpatialMemoryEngine
from sme.retrieval import SearchQuery

try:
    from sme import __version__ as SME_VERSION
except ImportError:  # pragma: no cover
    SME_VERSION = "1.2.0"

APP_TITLE = "Spatial Memory Engine API"
APP_VERSION = SME_VERSION

# 配置中心的默认读写文件（相对启动目录 cwd；可用 --config / SME_CONFIG_PATH 覆盖）
DEFAULT_CONFIG_FILE = "data/sme.config.json"


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


class ConfigPutRequest(BaseModel):
    """PUT /config：values 的值可为字符串（走 parse_value）或原生 JSON 类型（直接校验）。"""

    values: dict[str, Any] = Field(default_factory=dict)
    save: bool = False


class PresetRequest(BaseModel):
    key: str
    save: bool = True


class ConfigCheckRequest(BaseModel):
    ping: bool = False


# --------------------------------------------------------------------------- #
# config center helpers（分组映射 / 取值 / 热重建）
# --------------------------------------------------------------------------- #
# 分组映射：ITEMS 的 path 前缀 -> 展示分组。顺序即页面展示顺序。
GROUP_DEFS: list[dict[str, Any]] = [
    {
        "key": "llm", "name": "模型接入（LLM）",
        "description": "大模型接口地址、密钥与生成参数",
        "prefixes": ("llm",),
    },
    {
        "key": "embedding", "name": "向量（Embedding）",
        "description": "向量化引擎、模型与维度",
        "prefixes": ("embedding",),
    },
    {
        "key": "retrieval", "name": "检索与排序",
        "description": "两阶段检索通道与最终排序权重",
        "prefixes": ("retrieval", "ranking"),
    },
    {
        "key": "region", "name": "空间与Region",
        "description": "空间结构、Region 演化与 ANN 加速",
        "prefixes": ("region",),
    },
    {
        "key": "dynamics", "name": "记忆动力学",
        "description": "衰减 / 强化 / 融合 / 压缩等动态机制",
        "prefixes": ("policy", "decay", "reinforcement", "consolidation", "compression"),
    },
    {
        "key": "storage", "name": "存储",
        "description": "快照落盘、自动保存与增量日志",
        "prefixes": ("storage", "persistence"),
    },
    {
        "key": "api", "name": "服务",
        "description": "REST 服务监听地址、端口与鉴权",
        "prefixes": ("api",),
    },
    {
        "key": "visualization", "name": "可视化",
        "description": "空间可视化渲染参数",
        "prefixes": ("visualization",),
    },
    {
        "key": "memory", "name": "会话层约定",
        "description": "引擎不直接读取，由接入的聊天程序落实",
        "prefixes": ("memory",),
    },
    {
        "key": "extensions", "name": "扩展模块",
        "description": "默认关闭，开启后按各自语义生效",
        "prefixes": ("extraction", "qapair", "factgraph", "profile", "factversion",
                     "noise", "observability", "context", "namespaces", "rerank"),
    },
]

_GROUP_BY_PREFIX: dict[str, dict[str, Any]] = {
    prefix: g for g in GROUP_DEFS for prefix in g["prefixes"]
}


class ConfigApplyError(Exception):
    """热重建引擎失败；message 为可直接展示给用户的中文提示。"""


def _embedding_identity(config: SMEConfig) -> tuple[str, str, int]:
    emb = config.embedding
    return (emb.provider, emb.model, emb.dim)


# 热重建时需要迁移活态状态的 v2 扩展模块：引擎属性名 -> sidecar 文件名
_MODULE_SIDECARS: dict[str, str] = {
    "qapair": "qapairs",
    "factgraph": "factgraph",
    "profile": "profile",
}


def _migrate_module_state(
    old_engine: SpatialMemoryEngine, new_engine: SpatialMemoryEngine
) -> None:
    """把旧引擎扩展模块（qapair 问答对 / factgraph 实体关系 / profile 画像事实）
    的活态状态迁移到新引擎：旧对象 ``to_dict()`` -> 新对象 ``load_dict()``。

    对象路径取不到（缺方法/异常）时回退 sidecar 文件路径（旧引擎
    ``_save_sidecars`` 的落盘位置）。任何失败都只吞掉——模块状态迁移
    绝不允许让热重建整体失败。
    """
    for name, sidecar in _MODULE_SIDECARS.items():
        old_mod = getattr(old_engine, name, None)
        new_mod = getattr(new_engine, name, None)
        if old_mod is None or new_mod is None:
            continue
        migrated = False
        try:
            data = old_mod.to_dict()
            if data:
                new_mod.load_dict(data)
                migrated = True
        except Exception:  # noqa: BLE001 - 模块迁移失败不阻断重建
            migrated = False
        if migrated:
            continue
        try:  # 回退：从旧引擎的 sidecar 文件恢复
            path = old_engine._sidecar_path(sidecar)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    new_mod.load_dict(json.load(fh))
        except Exception:  # noqa: BLE001
            pass


def _engine_counters(engine: SpatialMemoryEngine) -> dict[str, int]:
    """抓取引擎的演化/融合/压缩计数器（与快照 counters 同口径）。"""
    return {
        "splits": engine.region_manager.split_count,
        "merges": engine.region_manager.merge_count,
        "consolidations": engine.consolidation.consolidation_count,
        "compressions": engine.compression.compression_count,
    }


def rebuild_engine_with(
    old_engine: SpatialMemoryEngine, new_config: SMEConfig
) -> SpatialMemoryEngine:
    """用新配置构建引擎，并把旧引擎的记忆 / 图边 / 扩展模块状态迁移过去。

    迁移路径固定为 export_json -> import_json（保 id、含向量与图边），绝不使用
    save/load（快照内嵌旧配置会覆盖新配置）。任何一步失败抛出
    :class:`ConfigApplyError`，旧引擎保持原状、调用方据实返回 ok:false。
    """
    # embedding 身份（provider/model/dim）变更时，旧记忆向量与新配置必然不匹配：
    # 空间维度错位要么立刻报错、要么把毒化状态带进检索。这里直接拒绝并给出中文提示。
    if old_engine.memories and (
        _embedding_identity(old_engine.config) != _embedding_identity(new_config)
    ):
        old, new = old_engine.config.embedding, new_config.embedding
        raise ConfigApplyError(
            f"embedding 配置变更（{old.provider}/{old.model}/{old.dim} → "
            f"{new.provider}/{new.model}/{new.dim}）会使现有记忆的向量维度与新配置"
            "不一致，需要新库或清空现有记忆后重启服务；请先导出记忆并清空，"
            "再修改 embedding 配置"
        )
    counters = _engine_counters(old_engine)
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        old_engine.export_json(tmp_path)
        new_engine = SpatialMemoryEngine(new_config)
        new_engine.import_json(tmp_path)
        # 迁移扩展模块的活态状态（问答对 / 实体关系 / 画像事实）
        _migrate_module_state(old_engine, new_engine)
        # 回填统计计数器（engine_stats 展示口径与旧引擎保持一致）
        try:
            new_engine.region_manager.split_count = counters.get("splits", 0)
            new_engine.region_manager.merge_count = counters.get("merges", 0)
            new_engine.consolidation.consolidation_count = counters.get(
                "consolidations", 0
            )
            new_engine.compression.compression_count = counters.get("compressions", 0)
        except Exception:  # noqa: BLE001 - 计数器回填失败不影响重建
            pass
    except ConfigApplyError:
        raise
    except Exception as exc:  # noqa: BLE001 - 重建失败绝不允许 500 崩溃
        raise ConfigApplyError(f"应用配置失败，引擎保持原配置：{exc}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:  # pragma: no cover
            pass
    return new_engine


def _display_dict(config: SMEConfig) -> dict[str, Any]:
    """to_dict() 的展示视图：None 叶子按 config_check 的口径视为未设置。"""
    out: dict[str, Any] = {}
    for section, body in config.to_dict().items():
        if isinstance(body, dict):
            out[section] = {k: v for k, v in body.items() if v is not None}
        else:
            out[section] = body
    return out


def _item_payload(item: ConfigItem, cfg_dict: dict, defaults: dict) -> dict:
    value = get_value(cfg_dict, item)
    default = get_value(defaults, item)
    payload = {
        "path": item.path,
        "name": item.name,
        "value": value,
        "default": default,
        "help": item.desc,
        "type": item.kind,
        "choices": list(item.choices) if item.kind == "enum" else [],
        "minimum": item.minimum,
        "maximum": item.maximum,
        "modified": value != default,
    }
    # 密钥类配置只下发掩码：空值照常下发空串，非空下发 "******"，
    # 另给一个 set 布尔供前端展示"是否已设置"（modified 仍按真实值计算）
    if item.path.endswith("api_key"):
        payload["set"] = bool(value)
        if value:
            payload["value"] = "******"
    return payload


def _config_groups(cfg_dict: dict) -> list[dict[str, Any]]:
    defaults = defaults_config()
    buckets: dict[str, list[dict]] = {g["key"]: [] for g in GROUP_DEFS}
    for item in ITEMS:
        bucket = _GROUP_BY_PREFIX.get(item.path.split(".", 1)[0])
        key = bucket["key"] if bucket else "extensions"
        buckets[key].append(_item_payload(item, cfg_dict, defaults))
    # 空分组不下发（注册表暂无 visualization.* 可调项）
    return [
        {
            "key": g["key"],
            "name": g["name"],
            "description": g["description"],
            "items": buckets[g["key"]],
        }
        for g in GROUP_DEFS
        if buckets[g["key"]]
    ]


# --------------------------------------------------------------------------- #
# app factory
# --------------------------------------------------------------------------- #
def create_app(
    engine: SpatialMemoryEngine,
    config_file: str = "",
    source: str = "defaults",
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        engine: 启动时构建好的引擎（运行期可被配置中心热替换）。
        config_file: 配置中心读写的配置文件路径；留空用 DEFAULT_CONFIG_FILE。
        source: 配置来源标签："defaults" | "file" | "env"。
    """
    app = FastAPI(title=APP_TITLE, version=APP_VERSION)
    if not config_file:
        config_file = DEFAULT_CONFIG_FILE
    # 当前配置来源（随配置中心的动作实时更新）
    current_source = source

    # 引擎热替换锁：配置变更端点全程持锁（export→重建→import→swap→落盘整体
    # 原子），变更型端点持锁执行引擎操作，只读端点锁内取一次引擎快照。
    # RLock：持锁线程内再调 _swap_engine / 引擎自身 _lock 不会自锁。
    engine_lock = threading.RLock()

    def _snapshot() -> SpatialMemoryEngine:
        """只读端点用：锁内取一次当前引擎引用，之后全程用同一个引擎，
        消灭单请求内读到两个引擎的撕裂。"""
        with engine_lock:
            return engine

    def _swap_engine(new_engine: SpatialMemoryEngine) -> None:
        # 所有端点闭包共享 create_app 的 engine 变量（同一 cell），
        # nonlocal 重赋值后既有端点自动指向新引擎。调用方须已持 engine_lock。
        nonlocal engine
        engine = new_engine

    # Bearer 鉴权：auth_token 非空即启用；健康检查、交互文档与配置页外壳放行。
    # 中间件始终注册，令牌每次请求从当前引擎 config 现读——UI 改 token 立即生效。
    open_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/"}

    @app.middleware("http")
    async def _auth_middleware(request, call_next):
        with engine_lock:
            token = getattr(engine.config.api, "auth_token", "") or ""
        if not token:
            return await call_next(request)
        path = request.url.path
        if path in open_paths or path.startswith("/static"):
            return await call_next(request)
        if request.headers.get("Authorization") != f"Bearer {token}":
            return JSONResponse(
                status_code=401, content={"detail": "unauthorized"}
            )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict:
        eng = _snapshot()
        return {"status": "ok", "memories": len(eng.memories)}

    # ------------------------- config center -------------------------- #
    @app.get("/config", response_model=dict)
    def get_config() -> dict:
        eng = _snapshot()
        # source 只在持锁的配置端点里被赋值（字符串赋值原子），快照读取即可
        return {
            "config_file": config_file,
            "source": current_source,
            "groups": _config_groups(_display_dict(eng.config)),
            "presets": [
                {
                    "key": p["key"],
                    "name": p["name"],
                    "description": p["desc"],
                    "count": len(p["values"]),
                }
                for p in PRESETS
            ],
        }

    def _save_or_error(new_dict: dict) -> Optional[dict]:
        """save=true 时落盘；返回错误响应 dict 或 None。"""
        try:
            save_config(config_file, new_dict)
        except OSError as exc:
            return {
                "ok": False,
                "errors": {"_global": f"配置已生效，但写入 {config_file} 失败：{exc}"},
            }
        return None

    @app.put("/config", response_model=dict)
    def put_config(req: ConfigPutRequest) -> dict:
        nonlocal current_source
        errors: dict[str, str] = {}
        parsed: list[tuple[ConfigItem, Any]] = []
        for path, raw in (req.values or {}).items():
            item = ITEM_BY_PATH.get(path)
            if item is None:
                errors[path] = "未知的配置项"
                continue
            if item.path.endswith("api_key") and (raw == "******" or raw == ""):
                # 掩码回显值 / 空串 = 不修改该项（GET /config 不回显真实密钥，
                # 回传掩码不能覆盖真实值）
                continue
            try:
                if isinstance(raw, str):
                    parsed.append((item, parse_value(item, raw)))
                else:
                    error = validate_value(item, raw)
                    if error:
                        errors[path] = error
                    else:
                        parsed.append((item, raw))
            except ValueError as exc:
                errors[path] = str(exc)
        if errors:
            # 任何一项非法都不做生效动作
            return {"ok": False, "errors": errors}
        if not parsed:
            # 空提交（或全部项都是"不修改"）：不重建引擎
            return {"ok": True, "applied": 0, "config_file": config_file}

        # 配置变更全程持锁：export→重建→import→swap→落盘整体原子
        with engine_lock:
            new_dict = engine.config.to_dict()
            for item, value in parsed:
                set_value(new_dict, item, value)
            try:
                new_engine = rebuild_engine_with(engine, SMEConfig.from_dict(new_dict))
            except ConfigApplyError as exc:
                return {"ok": False, "errors": {"_global": str(exc)}}
            _swap_engine(new_engine)
            if req.save:
                failed = _save_or_error(new_dict)
                if failed:
                    return failed
                current_source = "file"
        return {"ok": True, "applied": len(parsed), "config_file": config_file}

    @app.post("/config/preset", response_model=dict)
    def apply_preset_endpoint(req: PresetRequest) -> dict:
        nonlocal current_source
        preset = PRESET_BY_KEY.get(req.key)
        if preset is None:
            raise HTTPException(
                status_code=400,
                detail=f"未知预设 key：{req.key}（可选：{'、'.join(PRESET_BY_KEY)}）",
            )
        errors: dict[str, str] = {}
        for path, value in preset["values"].items():
            error = validate_value(ITEM_BY_PATH[path], value)
            if error:
                errors[path] = error
        if errors:
            return {"ok": False, "errors": errors}
        # 配置变更全程持锁：export→重建→import→swap→落盘整体原子
        with engine_lock:
            new_dict = engine.config.to_dict()
            for path, value in preset["values"].items():
                set_value(new_dict, ITEM_BY_PATH[path], value)
            try:
                new_engine = rebuild_engine_with(engine, SMEConfig.from_dict(new_dict))
            except ConfigApplyError as exc:
                return {"ok": False, "errors": {"_global": str(exc)}}
            _swap_engine(new_engine)
            if req.save:
                failed = _save_or_error(new_dict)
                if failed:
                    return failed
                current_source = "file"
        return {"ok": True, "applied": list(preset["values"].keys())}

    @app.post("/config/check", response_model=dict)
    def check_config(req: ConfigCheckRequest) -> dict:
        eng = _snapshot()
        return run_config_check(eng.config, ping=req.ping)

    @app.post("/config/reset", response_model=dict)
    def reset_config() -> dict:
        nonlocal current_source
        if os.path.exists(config_file):
            try:
                os.remove(config_file)
            except OSError as exc:
                return {
                    "ok": False,
                    "errors": {"_global": f"删除配置文件 {config_file} 失败：{exc}"},
                }
        # reset = 内置默认 + 启动时 env 注入的配置重放（不能把 env 弄丢）
        config = SMEConfig()
        _apply_env_overrides(config)
        with engine_lock:
            try:
                new_engine = rebuild_engine_with(engine, config)
            except ConfigApplyError as exc:
                return {"ok": False, "errors": {"_global": str(exc)}}
            _swap_engine(new_engine)
            current_source = "defaults"
        return {"ok": True}

    # ------------------------- memories ------------------------------- #
    @app.post("/memories", response_model=dict)
    def add_memory(req: AddMemoryRequest) -> dict:
        # 变更型端点持锁执行引擎操作：与配置热替换互斥，写入绝不丢失
        with engine_lock:
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
        # route through engine.add so the full write pipeline applies exactly
        # like engine.add_many (extraction/factversion/qapair are honored)
        with engine_lock:
            eng = engine
            added = 0
            for item in req.memories:
                text = (item or {}).get("text")
                if not text or not str(text).strip():
                    continue
                mem = eng.add(
                    text=str(text),
                    metadata=item.get("metadata", {}),
                    tags=item.get("tags", []),
                    importance=item.get("importance", 0.5),
                    source=item.get("source", "user"),
                )
                # the extraction pipeline may drop the item (extraction noise etc.):
                # only count memories that actually entered the store
                if mem.id in eng.memories:
                    added += 1
        return {
            "added": added,
            "dropped": len(req.memories) - added,
            "total": len(req.memories),
        }

    @app.get("/memories/{memory_id}", response_model=dict)
    def get_memory(memory_id: str) -> dict:
        eng = _snapshot()
        memory = eng.get(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return memory.to_dict()

    @app.patch("/memories/{memory_id}", response_model=dict)
    def update_memory(memory_id: str, req: UpdateMemoryRequest) -> dict:
        with engine_lock:
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
        with engine_lock:
            if not engine.delete(memory_id):
                raise HTTPException(status_code=404, detail="memory not found")
        return {"deleted": memory_id}

    @app.post("/memories/{memory_id}/hit", response_model=dict)
    def hit_memory(memory_id: str) -> dict:
        with engine_lock:
            result = engine.reinforce(memory_id)
        if result is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return result

    @app.post("/memories/{memory_id}/archive")
    def archive_memory(memory_id: str) -> dict:
        with engine_lock:
            if not engine.archive(memory_id):
                raise HTTPException(status_code=404, detail="memory not found")
        return {"archived": memory_id}

    @app.post("/memories/{memory_id}/restore")
    def restore_memory(memory_id: str) -> dict:
        with engine_lock:
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
        eng = _snapshot()
        hits = eng.search(query)
        return {
            "query": req.text,
            "count": len(hits),
            "results": [h.to_dict() for h in hits],
        }

    # ------------------------- regions -------------------------------- #
    @app.get("/regions", response_model=dict)
    def list_regions() -> dict:
        eng = _snapshot()
        # 迭代引擎内部 dict：持有引擎自身锁，防止并发写入导致迭代崩溃
        with eng._lock:
            regions = [region.to_dict() for region in eng.space.regions.values()]
        return {"count": len(regions), "regions": regions}

    @app.post("/regions/search", response_model=dict)
    def search_regions(req: RegionSearchRequest) -> dict:
        eng = _snapshot()
        hits = eng.search_regions(req.text, req.top_k)
        return {"count": len(hits), "results": [h.to_dict() for h in hits]}

    @app.post("/regions/evolve")
    def evolve_regions() -> dict:
        with engine_lock:
            eng = engine
            # evolution_pass 会 split/merge Region 并改写成员归属，必须与
            # engine 写路径/检索路径互斥（engine._lock 是 RLock，内部
            # _maybe_evolve 重入安全）
            with eng._lock:
                events = eng.region_manager.evolution_pass(eng.space)
        return {"events": [e.__dict__ for e in events]}

    # ------------------------- stats ----------------------------------- #
    @app.get("/stats", response_model=dict)
    def stats() -> dict:
        eng = _snapshot()
        with eng._lock:
            return eng.engine_stats()

    # ------------------------- consolidation --------------------------- #
    @app.post("/consolidate", response_model=dict)
    def consolidate() -> dict:
        with engine_lock:
            created = engine.consolidate()
        return {"created": [m.to_dict() for m in created]}

    @app.post("/compress", response_model=dict)
    def compress() -> dict:
        with engine_lock:
            created = engine.compress()
        return {"created": [m.to_dict() for m in created]}

    # ------------------------- graph ----------------------------------- #
    @app.get("/graph", response_model=dict)
    def graph() -> dict:
        eng = _snapshot()
        with eng._lock:
            return {
                "edge_count": len(eng.graph),
                "edges": [e.to_dict() for e in eng.graph.edges],
            }

    @app.post("/graph/link")
    def graph_link(req: LinkRequest) -> dict:
        with engine_lock:
            ok = engine.link(req.a, req.b, req.kind, req.weight, req.note)
        if not ok:
            raise HTTPException(status_code=404, detail="one or both memories missing")
        return {"linked": True}

    # ------------------------- export / import ------------------------- #
    @app.get("/export")
    def export() -> dict:
        eng = _snapshot()
        with eng._lock:
            return {
                "memories": [m.to_dict() for m in eng.memories.values()],
                "graph": eng.graph.to_dict(),
                "stats": eng.engine_stats(),
            }

    @app.post("/import", response_model=dict)
    def import_memories(req: ImportRequest) -> dict:
        with engine_lock:
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
        with engine_lock:
            count = engine.import_memories(items)
        return {"imported": count}

    # ------------------------- visualize ------------------------------- #
    @app.get("/visualize")
    def visualize() -> FileResponse:
        # per-request 唯一临时文件：并发请求不再互相覆盖固定路径
        fd, path = tempfile.mkstemp(prefix="sme_space_", suffix=".png")
        os.close(fd)
        eng = _snapshot()
        try:
            eng.visualize(path)
        except (RuntimeError, ValueError) as exc:
            # matplotlib 未安装（中文 RuntimeError）或空间为空（无可渲染记忆）：
            # 一律 501 + 原始消息，不允许 500 崩溃
            try:
                os.remove(path)
            except OSError:  # pragma: no cover
                pass
            return JSONResponse(status_code=501, content={"error": str(exc)})
        # 响应体发送完成后再删除临时文件（FileResponse 是后台流式发送）
        return FileResponse(
            path, media_type="image/png",
            background=BackgroundTask(os.remove, path),
        )

    # ------------------------- optional modules ------------------------ #
    @app.get("/facts")
    def facts() -> dict:
        """Module 03 - knowledge-graph entities & relations (enabled only)."""
        eng = _snapshot()
        with eng._lock:
            fg = getattr(eng, "factgraph", None)
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
        eng = _snapshot()
        with eng._lock:
            fg = getattr(eng, "factgraph", None)
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
        eng = _snapshot()
        with eng._lock:
            prof = getattr(eng, "profile", None)
            if prof is None or not prof.enabled:
                return {"enabled": False, "profile_facts": [], "snapshots": {}}
            return {
                "enabled": True,
                "stats": prof.stats(),
                "profile_facts": [
                    m.to_dict() for m in prof.profile_memories(eng.memories)
                ],
                "snapshots": prof.snapshots,
            }

    @app.get("/qapairs")
    def qapairs() -> dict:
        """Module 02 - stored question/answer pairs (enabled only)."""
        eng = _snapshot()
        with eng._lock:
            qa = getattr(eng, "qapair", None)
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
        eng = _snapshot()
        with eng._lock:
            tele = getattr(eng, "telemetry", None)
            if tele is None or not tele.enabled:
                return {"enabled": False, "summary": {}}
            return {"enabled": True, **tele.report(eng)}

    @app.get("/metrics/report")
    def metrics_report() -> FileResponse:
        """Module 10 - download the full telemetry report as JSON."""
        eng = _snapshot()
        with eng._lock:
            tele = getattr(eng, "telemetry", None)
            if tele is None or not tele.enabled:
                raise HTTPException(status_code=404, detail="telemetry disabled")
            # per-request 唯一临时文件：固定共享路径在并发请求下互相覆写、
            # 流式发送期间被改写会产出撕裂 JSON（旧实现）
            fd, path = tempfile.mkstemp(prefix="sme_report_", suffix=".json")
            os.close(fd)
            tele.export_json(path, eng)
        return FileResponse(path, media_type="application/json",
                            filename="sme_report.json",
                            background=BackgroundTask(os.remove, path))

    # ------------------------- config UI ------------------------------- #
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(os.path.join(static_dir, "index.html"), media_type="text/html")

    return app


def _apply_env_overrides(config: SMEConfig) -> bool:
    """把 ``SME_LLM_*`` / ``SME_EMBEDDING_*`` / ``SME_API_AUTH_TOKEN`` 环境变量
    覆盖到给定配置上（纯函数，resolve_engine 与 /config/reset 共用）。

    Returns:
        是否存在环境变量覆盖（用于配置来源标记 "env"）。
    """
    from_env = False
    base_url = os.environ.get("SME_LLM_BASE_URL")
    if base_url:
        config.llm.base_url = base_url
        config.llm.api_key = os.environ.get("SME_LLM_API_KEY", "")
        config.llm.model = os.environ.get("SME_LLM_MODEL", "gpt-4o-mini")
        from_env = True
    # 只有环境变量真实存在时才覆盖：无条件覆盖会在 env 未设置时把
    # 配置文件（或文件重放）设置的 token 清空——鉴权意外失效
    _tok = os.environ.get("SME_API_AUTH_TOKEN")
    if _tok:
        config.api.auth_token = _tok
        from_env = True
    provider = os.environ.get("SME_EMBEDDING_PROVIDER")
    if provider:
        config.embedding.provider = provider
        config.embedding.model = os.environ.get(
            "SME_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        config.embedding.dim = int(os.environ.get("SME_EMBEDDING_DIM", "64"))
        config.embedding.base_url = os.environ.get("SME_EMBEDDING_BASE_URL", "")
        config.embedding.api_key = os.environ.get("SME_EMBEDDING_API_KEY", "")
        from_env = True
    return from_env


def _restore_snapshot(engine: SpatialMemoryEngine) -> bool:
    """服务启动时恢复既有快照（含 WAL 重放），否则每次重启都是空库。

    快照路径取 ``engine.config.storage.path``，兼容 gzip 后缀；
    恢复失败（文件损坏等）不阻断启动，降级为空库并告警。
    重放前 WAL 有积压时，恢复后立即 save() checkpoint——replay() 会清空
    WAL，若不落盘，积压增量只存在于内存，服务在下次自动保存（默认每 30
    次写入）之前被杀会丢失这些增量。
    """
    base = str(getattr(engine.config.storage, "path", "") or "")
    if not base:
        return False
    import logging

    for candidate in (base, base + ".gz"):
        if os.path.exists(candidate):
            try:
                wal_pending = engine.wal.has_pending()
            except Exception:  # noqa: BLE001 - WAL 探测失败不阻断恢复
                wal_pending = False
            try:
                restored = bool(engine.load(candidate))
            except Exception as exc:  # noqa: BLE001 - 启动期降级
                logging.getLogger("sme.api").warning(
                    "快照恢复失败，以空库启动（%s）: %s", candidate, exc
                )
                # 坏快照改名保留：否则空库启动后的自动保存会把它覆盖掉，
                # 把"部分损坏"放大成"全量丢失"（无任何现场可查）
                try:
                    backup = candidate + ".corrupt-" + time.strftime("%Y%m%d%H%M%S")
                    os.replace(candidate, backup)
                    logging.getLogger("sme.api").warning(
                        "已将坏快照备份为 %s", backup
                    )
                except OSError:
                    pass
                return False
            if restored and wal_pending:
                try:
                    engine.save()
                except Exception as exc:  # noqa: BLE001 - checkpoint 失败不阻断服务
                    logging.getLogger("sme.api").warning(
                        "WAL 重放后 checkpoint 失败（增量仍在内存，下次保存会落盘）: %s", exc
                    )
            return restored
    return False


def _apply_file_overrides(engine: SpatialMemoryEngine, path: str) -> None:
    """快照恢复后重放配置文件的"显式键"（与内置默认不同的键）。

    快照内嵌的配置整体覆盖会吞掉用户在两次启动之间手改的
    ``data/sme.config.json``——尤其 ``api.auth_token``（重启后 token 清空
    → API 无鉴权裸奔）。这里把文件里相对内置默认的显式修改重放上去；
    embedding 身份（provider/model/dim）沿用空库闸门：非空库的向量维度
    是数据事实，文件值不覆盖（与 :func:`_rebind_after_env` 同一原则）。
    """
    if not path or not os.path.exists(path):
        return
    from sme.config_items import ITEMS, defaults_config, get_value, load_config

    file_cfg = load_config(path)
    if not isinstance(file_cfg, dict) or not file_cfg:
        return
    defaults = defaults_config()
    empty_library = not engine.memories
    for item in ITEMS:
        try:
            fval = get_value(file_cfg, item)
            dval = get_value(defaults, item)
        except Exception:  # noqa: BLE001 - 单项失败不阻断其余重放
            continue
        if fval == dval:
            continue
        if not empty_library and item.path.startswith("embedding.") and item.path.split(
            "."
        )[-1] in ("provider", "model", "dim"):
            continue  # 非空库不改 embedding 身份
        parts = item.path.split(".")
        try:
            obj = engine.config
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], fval)
        except AttributeError:
            continue


def _rebind_after_env(engine: SpatialMemoryEngine) -> None:
    """快照恢复后重放环境变量覆盖（env 是本次启动的显式意图）。

    LLM/鉴权总是重放；embedding 身份（provider/model/dim）仅在库为空时重放——
    已有记忆的向量维度属于数据事实，env 覆盖会导致维度错配（与 PUT /config 的
    embedding 身份闸门同一原则）。
    """
    cfg = engine.config
    from_env = _apply_env_overrides(cfg)
    if not from_env:
        return
    engine.llm = engine.llm.__class__(cfg.llm)
    engine.consolidation.llm = engine.llm
    engine.compression._llm = engine.llm
    engine.extraction.llm = engine.llm
    engine.factgraph_extractor.llm = engine.llm
    if not engine.memories and (
        engine.embeddings.name != cfg.embedding.provider
        or engine.embeddings.model_name != cfg.embedding.model
        or engine.embeddings.dim != cfg.embedding.dim
    ):
        from sme.embedding import build_embedding_provider

        engine.embeddings = build_embedding_provider(cfg.embedding)


def resolve_engine(
    explicit_config_path: str = "",
) -> tuple[SpatialMemoryEngine, str, str]:
    """按优先级解析引擎配置并构建引擎，并恢复既有快照。

    优先级：``--config`` 参数 > ``SME_CONFIG_PATH`` 环境变量 >
    ``data/sme.config.json``（存在才用，相对 cwd）> 代码内置默认；
    无配置文件时 ``SME_LLM_*`` / ``SME_EMBEDDING_*`` 环境变量仍会覆盖默认值
    （在快照恢复之后重放，env 意图优先于快照内配置；embedding 身份例外见
    :func:`_rebind_after_env`）。

    Returns:
        (engine, config_file, source)；source ∈ {"file", "env", "defaults"}，
        config_file 是配置中心实际读写的文件路径。
    """
    env_path = os.environ.get("SME_CONFIG_PATH", "")
    default_file = DEFAULT_CONFIG_FILE if os.path.exists(DEFAULT_CONFIG_FILE) else ""
    path = explicit_config_path or env_path or default_file
    if path and os.path.exists(path):
        engine = SpatialMemoryEngine(config_path=path)
        restored = _restore_snapshot(engine)
        if restored:
            # 恢复顺序：配置文件显式键（用户手改的意图，如加 auth_token）
            # → env 覆盖（env 最高）。快照内嵌配置只作底，不再压过手改文件。
            _apply_file_overrides(engine, path)
            _rebind_after_env(engine)
        return engine, path, "file"

    config = SMEConfig()
    from_env = _apply_env_overrides(config)
    config_file = explicit_config_path or DEFAULT_CONFIG_FILE
    # build the engine from the fully-resolved config so the embedding
    # provider / dimension are created consistently from the env values
    engine = SpatialMemoryEngine(config)
    if _restore_snapshot(engine):
        # 快照配置生效后重放配置文件显式键与 env 覆盖
        # （embedding 身份在非空库上不覆盖）
        _apply_file_overrides(engine, config_file)
        _rebind_after_env(engine)
    return engine, config_file, ("env" if from_env else "defaults")


def build_engine_from_env() -> SpatialMemoryEngine:
    """Create an engine from environment variables (or defaults).

    SME_LLM_BASE_URL / SME_LLM_API_KEY / SME_LLM_MODEL
    SME_EMBEDDING_PROVIDER / SME_EMBEDDING_MODEL / SME_EMBEDDING_DIM
    SME_CONFIG_PATH - JSON config file

    兼容旧入口；需要同时拿到 config_file / source 时用 :func:`resolve_engine`。
    """
    engine, _config_file, _source = resolve_engine()
    return engine


def main() -> None:
    """Start the REST server: ``python -m sme.api``.

    Config source priority: ``--config`` > ``SME_CONFIG_PATH`` >
    ``data/sme.config.json`` (if present) > built-in defaults.
    Listen address priority: ``--host/--port`` > config ``api.host/api.port``
    > built-in defaults (127.0.0.1:8000).
    Authentication: set ``api.auth_token`` (config) or ``SME_API_AUTH_TOKEN``.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="SME REST API 服务")
    parser.add_argument("--config", default="", help="配置文件路径（等价 SME_CONFIG_PATH，优先级更高）")
    parser.add_argument("--host", default=None, help="监听地址（缺省读配置 api.host，再缺省 127.0.0.1）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（缺省读配置 api.port，再缺省 8000）")
    args = parser.parse_args()
    engine, config_file, source = resolve_engine(args.config)
    # 监听地址：显式启动参数 > 配置文件 api.host/api.port > 内置默认
    host = args.host if args.host is not None else (str(engine.config.api.host) or "127.0.0.1")
    port = args.port if args.port is not None else int(engine.config.api.port or 8000)
    app = create_app(engine, config_file=config_file, source=source)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
