"""mem0 选手（mem0ai 2.2.1）。

配置要点：
- llm: provider=openai 指到 lab/.env 的 DeepSeek（mem0 的 infer 抽取每轮 add 一次 LLM）
- embedder: provider=huggingface 本地 sentence-transformers，模型统一 Qwen3-Embedding-0.6B
  （mem0 内部 SentenceTransformer 裸 encode，与统一协议一致）
- vector_store: 默认内嵌 qdrant，path 指到 lab workspace（跑完删，不留残余）
- search 返回 qdrant 余弦分（0-1）
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .base import Contestant
from .common import load_env, EMBED_MODEL, EMBED_DIM


class Mem0Contestant(Contestant):
    name = "mem0"

    def __init__(self, workspace: Path, user_id: str = "lab_user"):
        self.workspace = Path(workspace)
        self.user_id = user_id
        self.reinit_cost_s = 0.0
        self.reset()

    def _build(self):
        from mem0 import Memory

        env = load_env()
        self.workspace.mkdir(parents=True, exist_ok=True)
        return Memory.from_config({
            "llm": {
                "provider": "openai",
                "config": {
                    "model": env["LAB_LLM_MODEL"],
                    "openai_base_url": env["LAB_LLM_BASE_URL"],
                    "api_key": env["LAB_LLM_API_KEY"],
                    "temperature": 0.3,
                    "max_tokens": 2048,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": EMBED_MODEL},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "lab_battle",
                    "embedding_model_dims": EMBED_DIM,
                    "path": str(self.workspace / "qdrant"),
                },
            },
            "history_db_path": str(self.workspace / "mem0_history.db"),
        })

    def reset(self) -> None:
        t0 = time.time()
        self.mem = self._build()
        self.reinit_cost_s = time.time() - t0

    def add(self, text: str, ns: Optional[str] = None) -> None:
        uid = ns or self.user_id
        # infer=True（默认）：mem0 用 LLM 抽取事实并自动 add/update/delete
        self.mem.add(text, user_id=uid)

    def search(self, q: str, top_k: int = 5) -> list[tuple[str, float]]:
        res = self.mem.search(
            q, filters={"user_id": self.user_id}, top_k=top_k, threshold=0.0,
        )
        return [(r["memory"], float(r.get("score", 0.0))) for r in res.get("results", [])]

    def count(self) -> int:
        return len(self.mem.get_all(filters={"user_id": self.user_id}).get("results", []))

    def close(self) -> None:
        import gc

        self.mem = None
        gc.collect()  # Windows 下 sqlite/qdrant 句柄不 gc 就锁着工作区文件


def make_mem0(workspace: Path) -> Contestant:
    return Mem0Contestant(workspace)
