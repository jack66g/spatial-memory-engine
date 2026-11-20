# lab.baselines — 记忆擂台赛选手统一接口层（只 import sme，绝不改 sme）
#
# 选手注册表：run_battle.py 按 FACTORIES 顺序串行开赛。

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import Contestant
from .sme_presets import make_sme_chat, make_sme_kb_dynamic, make_sme_minimal

FACTORIES: dict[str, Callable[[Path], Contestant]] = {
    "sme_chat": make_sme_chat,
    "sme_kb_dynamic": make_sme_kb_dynamic,
    "sme_minimal": make_sme_minimal,
}

# 竞品选手惰性注册：依赖包已卸载（见 UNINSTALL.md）时优雅降级为缺席，
# SME 选手与评测框架不受影响，重装后自动恢复。
import importlib

import importlib.util as _ilu

for _name, _mod, _fn, _dep in (
    ("mem0", "lab.baselines.mem0_adapter", "make_mem0", "mem0"),
    ("rag_qwen3", "lab.baselines.rag_adapter", "make_rag", None),
    ("bm25_bigram", "lab.baselines.bm25_adapter", "make_bm25", "rank_bm25"),
    ("graphiti_kuzu", "lab.baselines.graphiti_adapter", "make_graphiti", "graphiti_core"),
):
    # 探测适配器模块 + 其真实依赖包（适配器多为函数内懒 import，仅测模块不够）
    if _dep is not None and _ilu.find_spec(_dep) is None:
        continue
    try:
        FACTORIES[_name] = getattr(importlib.import_module(_mod), _fn)
    except ImportError:
        pass

# 缺席选手及原因（如实记录进 battle 结果与报告）
ABSENT: dict[str, str] = {
    "langmem": "全系 0.0.11-0.0.30 使用 Python 3.11+ 语法（typing.Literal[*x]），本机仅 3.10.6，无法 import",
    "zep": "社区版服务端依赖 docker，本机无 docker（按约定不装）",
}
for _name, _mod in (
    ("mem0", "lab.baselines.mem0_adapter"),
    ("bm25_bigram", "lab.baselines.bm25_adapter"),
    ("graphiti_kuzu", "lab.baselines.graphiti_adapter"),
):
    if _name not in FACTORIES:
        ABSENT[_name] = "依赖包已按 UNINSTALL.md 卸载，重装命令见该文件"
