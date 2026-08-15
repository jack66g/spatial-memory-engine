# -*- coding: utf-8 -*-
"""mem3d bridge —— DeepSeek Harness 专用 SME 记忆边车服务。

职责
----
1. 托管多个**互相独立**的记忆模式（每个模式 = 独立配置 + 独立存储文件，
   切换模式即为空白记忆，除非该模式此前写过数据）。
2. 把对话/工具写入的记忆送入 SpatialMemoryEngine，并驱动 Region 演化、
   强化、衰减等全链路记忆动力学。
3. 对全部记忆向量做 PCA 三维投影，输出 3D 场景数据（节点/质心/Region 边/
   新增高亮），供 DeepSeek Harness 的 3D 浮窗动态渲染。

运行
----
    python bridge.py [--host 127.0.0.1] [--port 8756]

纯 Python 标准库 + sme（不修改 sme 任何源码）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # 项目根目录（sme 包所在处）
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from sme.config import SMEConfig  # noqa: E402
from sme.engine import SpatialMemoryEngine  # noqa: E402

# --------------------------------------------------------------------------- #
# 模式定义：每种模式一份独立配置 + 独立存储，互不共享记忆
# --------------------------------------------------------------------------- #
def _apply_semantic(cfg) -> None:
    """语义模式：BGE 本地模型。强制离线（模型未缓存则回退 hashing，绝不联网下载挂起）。"""
    import os as _os

    cache = _os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5")
    if not _os.path.isdir(cache):
        raise RuntimeError("BGE 模型未缓存，避免联网下载，回退 hashing")
    _os.environ["HF_HUB_OFFLINE"] = "1"
    _os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cfg.embedding.provider = "sentence-transformers"
    cfg.embedding.model = "BAAI/bge-small-zh-v1.5"
    cfg.embedding.dim = 512


MODES: dict[str, dict] = {
    "chat": {
        "name": "对话记忆",
        "desc": "默认模式：强化/衰减/融合全开，hashing 离线向量，Region 自动演化。适合日常聊天。",
        "apply": lambda cfg: None,
    },
    "semantic": {
        "name": "语义记忆",
        "desc": "对话记忆 + 本地 BGE 中文向量（sentence-transformers），真实语义聚类；未安装或未缓存时自动回退 hashing。",
        "apply": _apply_semantic,
    },
    "focus": {
        "name": "只记事实",
        "desc": "模块01事实提取（离线规则）+ 模块02问答对回放 + 模块06噪音抑制：只沉淀干货，问题/寒暄不入库。",
        "apply": lambda cfg: (
            setattr(cfg.extraction, "enabled", True),
            setattr(cfg.extraction, "mode", "rules"),
            setattr(cfg.qapair, "enabled", True),
            setattr(cfg.noise, "enabled", True),
        ),
    },
    "v2": {
        "name": "深度对话",
        "desc": "v2 全模块：事实提取+问答对+时序知识图谱+用户画像+事实纠错+噪音抑制，强化/衰减全开。最丰富但占用更大。",
        "apply": lambda cfg: (
            setattr(cfg.extraction, "enabled", True),
            setattr(cfg.extraction, "mode", "rules"),
            setattr(cfg.qapair, "enabled", True),
            setattr(cfg.factgraph, "enabled", True),
            setattr(cfg.factgraph, "extract_mode", "rules"),
            setattr(cfg.profile, "enabled", True),
            setattr(cfg.factversion, "enabled", True),
            setattr(cfg.noise, "enabled", True),
        ),
    },
    "kb_dynamic": {
        "name": "知识库·动态",
        "desc": "知识不衰减、越查越重要（命中强化），适合文档/事实沉淀且持续查阅。",
        "apply": lambda cfg: (
            setattr(cfg.policy, "decay_enabled", False),
        ),
    },
    "kb_static": {
        "name": "知识库·静态",
        "desc": "纯只读：衰减/强化/演化全关，检索结果可复现，适合固定文档问答。",
        "apply": lambda cfg: (
            setattr(cfg.policy, "decay_enabled", False),
            setattr(cfg.region, "auto_evolve", False),
        ),
    },
    "robot": {
        "name": "具身机器人",
        "desc": "WAL 崩溃安全（模块07）+ 多用户隔离（模块12）+ 强化全开，适合长时间无人值守运行。",
        "apply": lambda cfg: (
            setattr(cfg.persistence, "enabled", True),
            setattr(cfg.namespaces, "enabled", True),
            setattr(cfg.namespaces, "default_ns", "default"),
        ),
    },
    "minimal": {
        "name": "裸向量库",
        "desc": "全关：纯向量存取，不演化、不衰减、不强化——当普通向量数据库用。",
        "apply": lambda cfg: (
            setattr(cfg.policy, "decay_enabled", False),
            setattr(cfg.region, "auto_evolve", False),
        ),
    },
}
DEFAULT_MODE = "chat"


def _build_config(mode_id: str) -> SMEConfig:
    """为某一模式构建独立配置。"""
    cfg = SMEConfig()
    base = os.path.join(HERE, "data", mode_id)
    os.makedirs(base, exist_ok=True)
    # 每个模式独占自己的存储目录 -> 不同模式绝不共享同一份记忆
    cfg.storage.path = os.path.join(base, "engine.json.gz")
    cfg.storage.autosave = True
    cfg.storage.autosave_interval = 5
    # 演示/可视化友好：演化更频繁，中文短句 hashing 聚类门槛放宽
    cfg.region.evolve_interval = 10
    cfg.region.min_join_cosine = 0.30
    cfg.region.auto_evolve = True
    MODES[mode_id]["apply"](cfg)
    return cfg


# --------------------------------------------------------------------------- #
# 模式运行时：一个模式一个引擎实例
# --------------------------------------------------------------------------- #
class ModeRuntime:
    def __init__(self, mode_id: str) -> None:
        self.mode_id = mode_id
        cfg = _build_config(mode_id)
        self.fallback_provider = False
        try:
            self.engine = SpatialMemoryEngine(cfg)
        except Exception:
            # 例如 sentence-transformers 未安装 -> 回退离线 hashing
            cfg = _build_config(mode_id)
            cfg.embedding.provider = "hashing"
            cfg.embedding.dim = 64
            self.engine = SpatialMemoryEngine(cfg)
            self.fallback_provider = True
        self._load()
        self.seq = 0                      # 写入序号：Client 据此判断场景变化
        self.recent_ids: list[str] = []   # 最近写入的节点 id（3D 高亮）
        self.recent_ts: dict[str, float] = {}
        self.last_error = ""

    def _load(self) -> None:
        path = self.engine.config.storage.path
        if path and os.path.exists(path):
            try:
                self.engine.load(path)
            except Exception as exc:  # 损坏快照不致命
                self.last_error = f"load failed: {exc}"

    @property
    def provider_name(self) -> str:
        name = self.engine.embeddings.name
        if self.fallback_provider:
            name += "(回退 hashing)"
        return name

    # -- 写入 ----------------------------------------------------------- #
    def add(self, text: str, role: str = "user", source: str = "dsh") -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty text")
        if role not in ("user", "assistant", "system", "tool"):
            role = "user"
        mem = self.engine.add(
            text,
            source=role,
            metadata={"via": "dsh-harness", "role": role, "bridge": source},
        )
        self.seq += 1
        self.recent_ids.append(mem.id)
        self.recent_ts[mem.id] = time.time()
        self.recent_ids = self.recent_ids[-16:]
        return {"id": mem.id, "seq": self.seq, "region": mem.region_id}

    def recall(self, text: str, top_k: int = 8) -> dict:
        hits = self.engine.search(text, top_k=top_k)
        return {
            "mode": self.mode_id,
            "hits": [
                {
                    "id": h.memory.id,
                    "text": h.memory.text,
                    "score": round(float(h.score), 4),
                    "region": h.region_id,
                    "hit_count": h.memory.hit_count,
                    "source": h.memory.source,
                }
                for h in hits
            ],
        }

    def reinforce(self, mid: str) -> dict:
        res = self.engine.reinforce(mid)
        if res is not None:
            self.seq += 1
        return {"ok": res is not None}

    # -- 3D 场景 -------------------------------------------------------- #
    def scene(self) -> dict:
        """全部记忆的 PCA 三维投影 + Region 结构。"""
        memories = [
            m
            for m in self.engine.memories.values()
            if not m.archived and m.embedding is not None
        ]
        base = {
            "mode": self.mode_id,
            "seq": self.seq,
            "memories": len(memories),
            "regions": len(self.engine.space.regions),
            "provider": self.provider_name,
            "splits": self.engine.region_manager.split_count,
            "merges": self.engine.region_manager.merge_count,
            "fallback": self.fallback_provider,
        }
        if not memories:
            base.update({"nodes": [], "centroids": [], "edges": [], "recent": []})
            return base

        vectors = [m.embedding for m in memories]
        mat = np.stack(vectors)
        coords, basis = _fit_project_3d(mat)
        if coords.shape[1] < 3:  # 不足 3 维补零
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))

        # 节点颜色按 Region 稳定分配（HSL 色相散列）
        region_ids = [self.engine.space.region_for(m.id) or "none" for m in memories]
        unique = sorted({r for r in region_ids})
        palette = {r: _hsl_color(r, unique) for r in unique}

        now_ts = time.time()
        recent_set = set(self.recent_ids[-12:])
        nodes = []
        for m, xy, rid in zip(memories, coords, region_ids):
            ts = m.created_at
            fresh = 1.0 if m.id in recent_set else 0.0
            nodes.append(
                {
                    "id": m.id,
                    "x": round(float(xy[0]), 4),
                    "y": round(float(xy[1]), 4),
                    "z": round(float(xy[2]), 4),
                    "c": palette[rid],
                    "region": rid,
                    "label": m.text[:48],
                    "source": m.source,
                    "hits": m.hit_count,
                    "fresh": fresh,
                    "age": round(max(0.0, now_ts - ts) / 3600.0, 2),  # 小时
                }
            )

        # Region 质心：与节点同一投影基
        centroids = []
        for region in self.engine.space.regions.values():
            if region.centroid is None or region.size == 0:
                continue
            c = _transform(basis, mat, region.centroid)
            if c.shape[0] < 3:
                c = np.pad(c, (0, 3 - c.shape[0]))
            centroids.append(
                {
                    "id": region.id,
                    "x": round(float(c[0]), 4),
                    "y": round(float(c[1]), 4),
                    "z": round(float(c[2]), 4),
                    "size": region.size,
                }
            )

        # Region 邻居边（质心连线）
        edges = []
        for edge in self.engine.space.region_edges.values():
            a = self.engine.space.regions.get(edge.source)
            b = self.engine.space.regions.get(edge.target)
            if a is None or b is None or a.centroid is None or b.centroid is None:
                continue
            edges.append({"a": edge.source, "b": edge.target})

        base.update(
            {
                "nodes": nodes,
                "centroids": centroids,
                "edges": edges,
                "recent": [r for r in self.recent_ids[-12:] if r in recent_set],
            }
        )
        return base

    def stats(self) -> dict:
        ms = self.engine.memory_stats().to_dict()
        rs = self.engine.region_stats().to_dict()
        return {
            "mode": self.mode_id,
            "seq": self.seq,
            "provider": self.provider_name,
            "memory": ms,
            "region": rs,
            "last_error": self.last_error,
        }


# --------------------------------------------------------------------------- #
# 3D 投影（在 bridge 内实现，与 sme.visualization 的 PCA 思路一致但不依赖它）
# --------------------------------------------------------------------------- #
def _fit_project_3d(mat: np.ndarray):
    """fit 节点向量，返回 (3d 坐标, 投影基)。基 = (mean, vt, k)。"""
    if mat.shape[0] == 1:
        out = np.zeros((1, 3))
        out[0, : min(3, mat.shape[1])] = mat[0, : min(3, mat.shape[1])]
        return out, None
    mean = mat.mean(axis=0)
    centered = mat - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(3, vt.shape[0])
    return centered @ vt[:k].T, (mean, vt, k)


def _transform(basis, mat, vector: np.ndarray):
    if basis is None:
        out = np.zeros(3)
        out[: min(3, vector.shape[0])] = vector[: min(3, vector.shape[0])]
        return out
    mean, vt, k = basis
    return (vector - mean) @ vt[:k].T


def _hsl_color(rid: str, unique: list[str]) -> str:
    """按 Region id 稳定散列一个色相。"""
    h = 0
    for ch in rid:
        h = (h * 31 + ord(ch)) % 360
    s = 0.62
    l = 0.55
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    r, g, b = (
        (c, x, 0) if h < 60 else
        (x, c, 0) if h < 120 else
        (0, c, x) if h < 180 else
        (0, x, c) if h < 240 else
        (x, 0, c) if h < 300 else
        (c, 0, x)
    )
    return "#%02x%02x%02x" % (
        int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    )


# --------------------------------------------------------------------------- #
# 服务主体
# --------------------------------------------------------------------------- #
class Bridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 8756) -> None:
        self.host = host
        self.port = port
        self.lock = threading.RLock()
        self.runtimes: dict[str, ModeRuntime] = {}
        self.mode_file = os.path.join(HERE, "data", "modes.json")
        self.auto_file = os.path.join(HERE, "data", "auto.json")
        self.current = self._read_current() or DEFAULT_MODE
        if self.current not in MODES:
            self.current = DEFAULT_MODE
        self.auto_enabled = self._read_auto()
        self.started_at = time.time()

    # -- 自动写入开关（浏览器端 UI 与动态插件共用，重启后保持） ---------- #
    def _read_auto(self) -> bool:
        try:
            with open(self.auto_file, "r", encoding="utf-8") as fh:
                return bool(json.load(fh).get("auto", True))
        except Exception:
            return True

    def _write_auto(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.auto_file), exist_ok=True)
            with open(self.auto_file, "w", encoding="utf-8") as fh:
                json.dump({"auto": self.auto_enabled}, fh, ensure_ascii=False)
        except Exception:
            pass

    # -- 模式状态 ------------------------------------------------------- #
    def _read_current(self) -> str | None:
        try:
            with open(self.mode_file, "r", encoding="utf-8") as fh:
                return json.load(fh).get("current")
        except Exception:
            return None

    def _write_current(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.mode_file), exist_ok=True)
            with open(self.mode_file, "w", encoding="utf-8") as fh:
                json.dump({"current": self.current}, fh, ensure_ascii=False)
        except Exception:
            pass

    def runtime(self, mode_id: str | None = None) -> ModeRuntime:
        mode_id = mode_id or self.current
        with self.lock:
            rt = self.runtimes.get(mode_id)
            if rt is None:
                rt = ModeRuntime(mode_id)
                self.runtimes[mode_id] = rt
            return rt

    def switch_mode(self, mode_id: str) -> dict:
        if mode_id not in MODES:
            raise ValueError(f"unknown mode: {mode_id}")
        with self.lock:
            prev = self.current
            self.current = mode_id
            self._write_current()
            rt = self.runtime(mode_id)
            blank = rt.engine.memory_stats().total == 0
        return {
            "mode": mode_id,
            "previous": prev,
            "blank": blank,
            "notice": (
                f"已切换到「{MODES[mode_id]['name']}」模式。"
                + ("该模式目前是空白记忆（从未写入过）。"
                   if blank else "该模式已有历史记忆，直接续用。")
            ),
            "memories": rt.engine.memory_stats().total,
        }

    def modes_info(self) -> dict:
        info = []
        for mid, spec in MODES.items():
            with self.lock:
                rt = self.runtimes.get(mid)
                total = rt.engine.memory_stats().total if rt is not None else None
            # 未加载的模式检查磁盘快照即可
            info.append(
                {
                    "id": mid,
                    "name": spec["name"],
                    "desc": spec["desc"],
                    "current": mid == self.current,
                    "memories": total,
                    "loaded": rt is not None,
                }
            )
        return {"modes": info, "current": self.current}

    # -- 对外 API 结果 -------------------------------------------------- #
    def handle(self, path: str, body: dict | None) -> dict:
        """把 HTTP 请求路由为 JSON 结果（便于单测）。"""
        if path == "/health":
            return {"ok": True, "mode": self.current, "uptime": round(time.time() - self.started_at, 1)}
        if path == "/auto":
            if body is not None and "on" in body:
                with self.lock:
                    self.auto_enabled = bool(body.get("on"))
                    self._write_auto()
            return {"auto": self.auto_enabled}
        if path == "/modes":
            return self.modes_info()
        if path == "/scene":
            return self.runtime().scene()
        if path == "/stats":
            return self.runtime().stats()
        if path == "/memory":
            if not body or not body.get("text"):
                raise ValueError("missing text")
            source = str(body.get("source", "dsh"))
            # 自动写入开关（仅拦截 source=auto 的会话自动写入；工具显式写入不受限）
            if source == "auto" and not self.auto_enabled:
                return {"skipped": True}
            return self.runtime().add(
                str(body["text"]), role=str(body.get("role", "user")),
                source=source,
            )
        if path == "/recall":
            if not body or not body.get("text"):
                raise ValueError("missing text")
            return self.runtime().recall(str(body["text"]), int(body.get("top_k", 8)))
        if path == "/mode":
            if not body or not body.get("id"):
                raise ValueError("missing mode id")
            return self.switch_mode(str(body["id"]))
        if path == "/reinforce":
            if not body or not body.get("id"):
                raise ValueError("missing memory id")
            return self.runtime().reinforce(str(body["id"]))
        if path == "/list":
            rt = self.runtime()
            return {
                "mode": self.current,
                "memories": [
                    {
                        "id": m.id,
                        "text": m.text[:200],
                        "source": m.source,
                        "region": m.region_id,
                        "hits": m.hit_count,
                    }
                    for m in rt.engine.memories.values()
                    if not m.archived
                ],
            }
        if path == "/delete":
            if not body or not body.get("id"):
                raise ValueError("missing memory id")
            rt = self.runtime()
            ok = rt.engine.delete(str(body["id"]))
            if ok:
                rt.seq += 1
            return {"ok": ok}
        if path == "/clear":
            # 清空当前模式（保留模式与配置，删除全部未归档记忆）
            rt = self.runtime()
            ids = [m.id for m in rt.engine.memories.values() if not m.archived]
            for mid in ids:
                rt.engine.delete(mid)
            rt.engine.save()
            rt.seq += 1
            rt.recent_ids = []
            return {"ok": True, "cleared": len(ids)}
        raise KeyError(f"unknown path: {path}")


# --------------------------------------------------------------------------- #
# HTTP 层（纯标准库）
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    bridge: Bridge = None  # 注入

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "3600")

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        # 浏览器直连（持久化 bundle UI）的 CORS 预检
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            self._send(200, self.bridge.handle(path, None))
        except KeyError:
            self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8")) if raw else {}
            if path == "/shutdown":
                self._send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._send(200, self.bridge.handle(path, body))
        except Exception as exc:
            self._send(500, {"error": str(exc), "trace": traceback.format_exc()[-600:]})


def main() -> None:
    parser = argparse.ArgumentParser(description="mem3d SME bridge for DeepSeek Harness")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8756)
    args = parser.parse_args()

    bridge = Bridge(args.host, args.port)
    handler = type("Handler", (_Handler,), {"bridge": bridge})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"[mem3d bridge] listening on http://{args.host}:{args.port} "
        f"(mode={bridge.current})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
