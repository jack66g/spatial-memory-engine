"""大实验基线包：统一 MemoryProvider 接口。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # SME 插件根

from baselines.base import BaseAdapter, SearchResult  # noqa: E402,F401
from baselines.sme_adapter import SMEAdapter  # noqa: E402,F401
from baselines.rag_adapter import RagAdapter  # noqa: E402,F401
from baselines.bm25_adapter import BM25Adapter  # noqa: E402,F401
from baselines.mem0_adapter import Mem0Adapter  # noqa: E402,F401
from baselines.langmem_adapter import LangMemAdapter  # noqa: E402,F401

ALL_ADAPTERS = {
    "sme_chat": lambda **kw: SMEAdapter(preset="chat", **kw),
    "sme_kb_dynamic": lambda **kw: SMEAdapter(preset="kb_dynamic", **kw),
    "sme_kb_static": lambda **kw: SMEAdapter(preset="kb_static", **kw),
    "sme_robot": lambda **kw: SMEAdapter(preset="robot", **kw),
    "sme_minimal": lambda **kw: SMEAdapter(preset="minimal", **kw),
    "rag": lambda **kw: RagAdapter(**kw),
    "bm25": lambda **kw: BM25Adapter(**kw),
    "mem0": lambda **kw: Mem0Adapter(**kw),
    "langmem": lambda **kw: LangMemAdapter(**kw),
}

__all__ = ["BaseAdapter", "SearchResult", "ALL_ADAPTERS", "SMEAdapter",
           "RagAdapter", "BM25Adapter", "Mem0Adapter", "LangMemAdapter"]
