"""Graphiti 选手：kuzu 嵌入式图后端（无 docker/Neo4j 环境）。

- graphiti-core 0.30.2 的 KuzuDriver 可用（官方已标注 deprecated，但能跑；如实记录）
- LLM: OpenAIGenericClient 指到 DeepSeek，structured_output_mode=json_object
  （generic client 文档明示 DeepSeek 系端点要降级到 json_object 模式）
- embedder: 自定义 EmbedderClient 包本地 Qwen3 单例（统一协议）
- add_episode 每轮会触发实体/关系抽取（多次 LLM 调用），是全场最慢的选手
- search 返回 EntityEdge 列表（fact 文本）；graphiti 不给显式相关度分，
  用排名递减占位分（judge 只看文本）
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import time
from pathlib import Path
from typing import Optional

from .base import Contestant
from .common import load_env, EMBED_DIM


async def _qwen_create(input_data) -> list[float]:
    texts = [input_data] if isinstance(input_data, str) else list(input_data)
    return (await _qwen_create_batch(texts))[0]


async def _qwen_create_batch(texts: list) -> list:
    from .common import embed_texts

    return embed_texts([str(t) for t in texts])


async def _noop_rank(query: str, passages: list) -> list:
    # 不重排：保持混合检索原序（默认 OpenAIRerankerClient 需要 OPENAI_API_KEY，
    # 本实验统一走 DeepSeek，且 judge 只看文本，重排不是本赛考察点）
    return [(p, 1.0 - 0.01 * i) for i, p in enumerate(passages)]


def _make_llm_client():
    """DeepSeek 的 OpenAIGenericClient + schema-echo 容错。

    json_object 模式下 graphiti 把目标 schema 注进 prompt，DeepSeek 偶发把
    schema 本身（{"properties": ...}）原样当回答返回，EdgeDuplicate(**resp) 直接炸。
    子类检测到这种"schemas 回声"就追加明确指令重试一次。"""
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    env = load_env()

    class DeepSeekGenericClient(OpenAIGenericClient):
        async def generate_response(self, messages, response_model=None, **kw):
            out = await super().generate_response(messages, response_model=response_model, **kw)
            if response_model is not None and isinstance(out, dict):
                fields = set(getattr(response_model, "model_fields", {}))
                if fields and not (fields & set(out.keys())) and "properties" in out:
                    messages[-1].content += (
                        "\n\nIMPORTANT: Reply with the JSON DATA object itself "
                        "(concrete values for the schema fields), NOT the JSON schema."
                    )
                    out = await super().generate_response(
                        messages, response_model=response_model, **kw)
            return out

    return DeepSeekGenericClient(
        config=LLMConfig(
            api_key=env["LAB_LLM_API_KEY"],
            model=env["LAB_LLM_MODEL"],
            base_url=env["LAB_LLM_BASE_URL"],
            temperature=0.0,
        ),
        structured_output_mode="json_object",
    )


# 引导 graphiti 把第一人称"我"当核心实体（官方 custom_extraction_instructions 通道）。
# 不加它：默认抽取只建名词实体间关系，日常陪伴对话抽不出边，检索恒空——
# 这不是 graphiti 的设计意图（它是 user_id 记忆图谱），属正确使用而非调参。
_EXTRACT_INSTR = (
    'IMPORTANT: Always treat the first-person speaker (the user, "我") as a core entity '
    'named "用户". Encode the user\'s facts, preferences and corrections as edges between '
    '"用户" and the other entities (e.g. (用户)-[养了]->(柯基 豆豆)). Extract in Chinese. '
    'When the user corrects an earlier statement, create the new edge reflecting the correction.'
)


class GraphitiContestant(Contestant):
    name = "graphiti_kuzu"

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.reinit_cost_s = 0.0
        self.reset()

    # ------------------------------------------------------------------ #
    def _build(self):
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.client import CrossEncoderClient
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

        self.workspace.mkdir(parents=True, exist_ok=True)
        # kuzu 0.11.3 Windows 坑：C++ 层打开"绝对中文路径"报 Error 3（UTF-8 路径
        # 被 ANSI API 解释成乱码目录）；相对路径正常。db 路径统一转相对 cwd。
        import os

        db_path = os.path.relpath(self.workspace / "graphiti.kuzu", os.getcwd())
        llm = _make_llm_client()

        embedder = type("LocalQwenEmbedder", (EmbedderClient,), {
            "__init__": lambda self: setattr(
                self, "config", EmbedderConfig(embedding_dim=EMBED_DIM)),
            "create": staticmethod(_qwen_create),
            "create_batch": staticmethod(_qwen_create_batch),
        })()

        cross_encoder = type("NoopCrossEncoder", (CrossEncoderClient,), {
            "rank": staticmethod(_noop_rank),
        })()

        driver = KuzuDriver(db=db_path)
        return Graphiti(
            graph_driver=driver, llm_client=llm,
            embedder=embedder, cross_encoder=cross_encoder,
        )

    def reset(self) -> None:
        t0 = time.time()
        # 持久专用 loop：graphiti 的 AsyncOpenAI/Kuzu AsyncConnection 连接池
        # 绑定首个 loop，asyncio.run() 每次 closing loop 会炸 "Event loop is closed"
        self._loop = asyncio.new_event_loop()
        self.g = self._build()
        self._loop.run_until_complete(self.g.build_indices_and_constraints())
        self._ensure_fts_indices()
        self._episodes = 0
        self.reinit_cost_s = time.time() - t0

    def _ensure_fts_indices(self) -> None:
        """graphiti 0.30.2 的 kuzu 后端缺索引：SCHEMA_QUERIES 只建表，
        build_indices_and_constraints 是 no-op，但 search 又要查 FTS 索引。
        这里手动补跑 graphiti 自己的 kuzu FTS 建索引语句（运行时补，不改库）。"""
        import kuzu

        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_fulltext_indices

        conn = kuzu.Connection(self.g.driver.db)
        try:
            for stmt in get_fulltext_indices(GraphProvider.KUZU):
                try:
                    conn.execute(stmt)
                except RuntimeError as e:  # 已存在时报错，忽略
                    if "already exists" not in str(e):
                        raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    def add(self, text: str, ns: Optional[str] = None) -> None:
        from graphiti_core.nodes import EpisodeType

        self._episodes += 1
        # group_id 必须留 None：graphiti 0.30.2 的 _resolve_request_scope 在显式
        # group_id 时访问 driver._database，KuzuDriver 没有该属性（上游对 kuzu
        # 支持不完整），传 None 走 provider 默认组分支可绕开
        self._loop.run_until_complete(self.g.add_episode(
            name=f"round_{self._episodes}",
            episode_body=text,
            source_description="user chat",
            source=EpisodeType.message,
            reference_time=_dt.datetime.now(_dt.timezone.utc),
            custom_extraction_instructions=_EXTRACT_INSTR,
        ))

    def search(self, q: str, top_k: int = 5) -> list[tuple[str, float]]:
        edges = self._loop.run_until_complete(self.g.search(q, num_results=top_k))
        return [
            (e.fact, round(1.0 - 0.05 * i, 4))  # 占位分：排名递减
            for i, e in enumerate(edges)
        ]

    def count(self) -> int:
        # 数实体关系边（graphiti 真正存储的知识形态）
        async def _n() -> int:
            res = await self.g.driver.client.run(
                "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS n"
            )
            return int(res[0]["n"]) if res else 0
        try:
            return self._loop.run_until_complete(_n())
        except Exception:  # noqa: BLE001 - schema 名不符时降级为 episode 数
            return self._episodes

    def close(self) -> None:
        try:
            self._loop.run_until_complete(self.g.close())
        except Exception:  # noqa: BLE001
            pass
        try:
            self._loop.close()
        except Exception:  # noqa: BLE001
            pass
        self.g = None


def make_graphiti(workspace: Path) -> Contestant:
    return GraphitiContestant(workspace)
