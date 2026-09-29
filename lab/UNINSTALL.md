# 竞品依赖清单（打完可卸载）

安装时间：2026-09-28（冒烟赛）。本机 Python 3.10.6，pip 走默认代理源。

## 主动安装（本次擂台赛对手）

| 包 | 版本 | 用途 | 卸载 |
|---|---|---|---|
| mem0ai | 2.2.1 | mem0 选手（LLM 抽取 + qdrant 内嵌向量库） | `pip uninstall mem0ai` |
| langmem | 0.0.15 | **缺席选手**（Python 3.11+ 语法，装了无法 import，仅留坑位） | `pip uninstall langmem` |
| rank-bm25 | 0.2.2 | BM25 选手 | `pip uninstall rank-bm25` |
| graphiti-core | 0.30.2 | Graphiti 选手 | `pip uninstall graphiti-core` |
| kuzu | 0.11.3 | Graphiti 的嵌入式图库后端 | `pip uninstall kuzu` |

一键卸载：

```bash
pip uninstall -y mem0ai langmem rank-bm25 graphiti-core kuzu
```

## 被动升级/安装（pip 解析依赖带进来的，卸载对手后按需回滚）

- 升级：openai 2.15.0 → 3.19.2、anthropic 0.107.1 → 1.8.0、jiter 0.12.0 → 0.17.0、
  pydantic → 2.13.5、protobuf → 6.33.6、websockets → 16.1.1
- 新装（部分主要）：qdrant-client 1.19.1、neo4j 6.3.1、langchain 1.4.2 全家
  （langchain-core/openai/anthropic/protocol、langgraph 全家）、langsmith、posthog、
  tiktoken、tenacity、trustcall、orjson、ormsgpack、portalocker、uuid-utils、
  requests-toolbelt、backoff、jsonpatch、jsonpointer、h2/hpack/hyperframe、dydantic
- pip 已警告的潜在冲突（这些包在本仓库 lab/ 之外的用途里才可能受影响）：
  gradio 5.30.0 要求 pydantic<2.12；mediapipe / open-clip-torch / tensorflow 2.19
  要求 protobuf<6 或 <5。lab 与 sme 的依赖不受影响。

## 本来就在（未动）

sentence-transformers 5.6.1（Qwen3-Embedding-0.6B 统一协议用）、httpx、numpy、torch 2.6.0。
