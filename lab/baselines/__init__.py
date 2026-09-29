# lab.baselines — 记忆擂台赛选手统一接口层（只 import sme，绝不改 sme）
#
# 选手注册表：run_battle.py 按 FACTORIES 顺序串行开赛。

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .base import Contestant
from .bm25_adapter import make_bm25
from .graphiti_adapter import make_graphiti
from .mem0_adapter import make_mem0
from .rag_adapter import make_rag
from .sme_presets import make_sme_chat, make_sme_kb_dynamic, make_sme_minimal

FACTORIES: dict[str, Callable[[Path], Contestant]] = {
    "sme_chat": make_sme_chat,
    "sme_kb_dynamic": make_sme_kb_dynamic,
    "sme_minimal": make_sme_minimal,
    "mem0": make_mem0,
    "rag_qwen3": make_rag,
    "bm25_bigram": make_bm25,
    "graphiti_kuzu": make_graphiti,
}

# 缺席选手及原因（如实记录进 battle 结果与报告）
ABSENT: dict[str, str] = {
    "langmem": "全系 0.0.11-0.0.30 使用 Python 3.11+ 语法（typing.Literal[*x]），本机仅 3.10.6，无法 import",
    "zep": "社区版服务端依赖 docker，本机无 docker（按约定不装）",
}
