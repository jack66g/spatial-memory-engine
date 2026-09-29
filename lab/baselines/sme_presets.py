"""SME 三位选手：chat / kb_dynamic / minimal（预设来自 sme.config_items.PRESETS）。

要点：
- 只 import sme，不改 sme。
- 统一协议：embedding = 本地 sentence-transformers Qwen3-Embedding-0.6B（1024 维）。
- kb_dynamic 的 extraction(mode=llm)/factversion 等模块需要 LLM：指到 lab/.env 的
  DeepSeek（reasoning_effort=none）。
- storage 落在调用方给的 workspace 目录（battle 跑完即删，绝不写仓库 data/）；
  autosave 关（冒烟赛不需要中途落盘）。
- PRESETS 里的 ``memory.*`` 是会话层配置（由聊天程序消费，不是引擎字段），
  这里按 SME 的设计意图在适配器里显式模拟：
    * consolidate_every / compress_every：每 N 轮 add 后调 engine.consolidate()/compress()
    * reinforce_on：search 命中后对 top 命中调 engine.reinforce(id)
  OFF_PERIOD(=1e9) 视为"永不触发"。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from sme import SpatialMemoryEngine, SMEConfig
from sme.config_items import PRESET_BY_KEY, OFF_PERIOD

from .base import Contestant
from .common import load_env, EMBED_MODEL, EMBED_DIM

MEMORY_KEYS = {
    "memory.reinforce_on", "memory.consolidate_every",
    "memory.compress_every", "memory.graph_expand",
}


def _set_path(cfg: SMEConfig, path: str, value) -> None:
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


class SmeContestant(Contestant):
    def __init__(self, preset_key: str, workspace: Path):
        self.preset_key = preset_key
        self.workspace = Path(workspace)
        self.name = f"sme_{preset_key}"
        self.reinit_cost_s = 0.0
        self.reset()

    # ------------------------------------------------------------------ #
    def _build_engine(self) -> SpatialMemoryEngine:
        env = load_env()
        cfg = SMEConfig()
        # 统一 embedding 协议
        cfg.embedding.provider = "sentence-transformers"
        cfg.embedding.model = EMBED_MODEL
        cfg.embedding.dim = EMBED_DIM
        # LLM（kb_dynamic 的 llm 抽取模块需要；chat/minimal 不触发）
        cfg.llm.base_url = env["LAB_LLM_BASE_URL"]
        cfg.llm.api_key = env["LAB_LLM_API_KEY"]
        cfg.llm.model = env["LAB_LLM_MODEL"]
        cfg.llm.reasoning_effort = "none"
        cfg.llm.max_tokens = 1024
        cfg.llm.temperature = 0.3
        cfg.llm.timeout = 120.0
        # 存储：只在 lab workspace 落盘（autosave 关，WAL 类 preset 走自己的节奏）
        self.workspace.mkdir(parents=True, exist_ok=True)
        cfg.storage.path = str(self.workspace / f"sme_{self.preset_key}.json.gz")
        cfg.storage.autosave = False

        session = {}
        for path, value in PRESET_BY_KEY[self.preset_key]["values"].items():
            if path in MEMORY_KEYS:
                session[path] = value  # 会话层，见模块 docstring
                continue
            _set_path(cfg, path, value)
        self._session = session
        self._add_count = 0
        return SpatialMemoryEngine(config=cfg)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        t0 = time.time()
        self.engine = self._build_engine()
        self.reinit_cost_s = time.time() - t0

    def add(self, text: str, ns: Optional[str] = None) -> None:
        self.engine.add(text, source="user", ns=ns)
        self._add_count += 1
        ce = self._session.get("memory.consolidate_every", OFF_PERIOD)
        pe = self._session.get("memory.compress_every", OFF_PERIOD)
        if ce < OFF_PERIOD and self._add_count % ce == 0:
            self.engine.consolidate()
        if pe < OFF_PERIOD and self._add_count % pe == 0:
            self.engine.compress()

    def search(self, q: str, top_k: int = 5) -> list[tuple[str, float]]:
        hits = self.engine.search(q, top_k=top_k)
        out = [(h.memory.text, float(h.score)) for h in hits]
        if self._session.get("memory.reinforce_on") and hits:
            for h in hits[:3]:
                self.engine.reinforce(h.memory.id)
        return out

    def count(self) -> int:
        return len(self.engine.memories)

    def close(self) -> None:
        # 让引擎（含 embedding 模型/WAL 句柄）可被回收；Qwen3 约 2.4GB，
        # 串行跑必须真释放（Windows 下 WAL 文件句柄不 gc 就锁着，删不掉工作区）
        import gc

        self.engine = None
        gc.collect()


def make_sme_chat(workspace: Path) -> Contestant:
    return SmeContestant("chat", workspace)


def make_sme_kb_dynamic(workspace: Path) -> Contestant:
    return SmeContestant("kb_dynamic", workspace)


def make_sme_minimal(workspace: Path) -> Contestant:
    return SmeContestant("minimal", workspace)
