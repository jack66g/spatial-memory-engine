# Spatial Memory Engine（SME）

> 新一代 AI 长期记忆系统：空间记忆（Spatial Memory）+ 记忆动力学 + 可解释检索。
> 零配置即可离线运行，也可以接入任意 OpenAI 兼容的 LLM / Embedding 服务（云端或本地）。

---

## 1. 项目简介

SME 是给 AI 应用用的"长期记忆插件"。它不像传统向量数据库那样只做
`Memory → Embedding → TopK` 的简单存取，而是把记忆组织成一个**会自我演化的空间**：

- 全部记忆构成连续 Embedding 空间，写入时自动形成**密度驱动的 Memory Region**（动态增长/拆分/融合），Region 之间构成图连接
- 记忆之间另有六种语义边构成 **Memory Graph**（引用/因果/对话/摘要/父子/邻居）
- 叠加**记忆动力学**：Ebbinghaus 强化、时间衰减（永不删除）、自动融合、长期压缩
- **两阶段混合检索**：Region 主导召回 + 向量 / BM25（中文 bigram）/ metadata 混合 + 可选图增强 + 8 信号可解释排序
- **v2 模块层（12 个，默认全关 = v1 行为）**：事实提取、问答对回放、时序知识图谱、用户画像、事实版本纠错、噪音抑制、WAL 增量持久化、存储后端、REST+SDK、可观测性、分层上下文、多用户隔离

一句话：**把你的 AI 从"聊完就忘"变成"越聊越懂你"**。可直接接入聊天机器人、知识库问答、具身机器人、Agent 等任意需要记忆的场景。

## 2. 核心特性

| 特性 | 说明 |
|---|---|
| 零配置离线 | 内置 hashing embedding，无任何依赖/密钥即可跑通全链路 |
| 可解释检索 | 每条命中带 `breakdown`（semantic/importance/freshness/weight/decay/hit_count/recency/region 八信号） |
| 中文友好 | 中文 1-2 元 BM25 关键词通道（默认开）+ 文档分条导入（法律/医疗等知识库场景） |
| 记忆生命周期 | 强化/衰减/融合/压缩可配可关，记忆永不因衰减被删除，归档可恢复 |
| 全部可关 | 89 项配置 + 5 个预设（聊天助手/知识库动态/知识库静态/具身机器人/全关） |
| 多种接入 | Python SDK、REST API（FastAPI）+ 官方 Client、`python -m sme.menu` 交互配置 |
| 性能 | 写入路径 O(N²)→近线性（10k 写入 ~4s）、100k 加载 ~3s、hashing 检索 p50 ~4ms（2k 条）/ ~22ms（10k 条，实测） |

## 3. 快速开始（10 秒）

```python
from sme.engine import SpatialMemoryEngine

engine = SpatialMemoryEngine()          # 零配置离线（内置 hashing embedding，无需任何 API Key）

engine.add("用户喜欢苹果")
engine.add("用户每天喝一杯苹果汁")
engine.add("用户周末打篮球")

hits = engine.search("用户喜欢什么水果？", top_k=5)
for h in hits:
    print(h.score, h.memory.text)       # 每条命中自带 8 信号打分分解 breakdown

engine.reinforce(hits[0].memory.id)     # 命中强化（Ebbinghaus）
engine.visualize("space.png")           # 2D 空间可视化
engine.save("memory_state.json.gz")     # 持久化
```

```bash
pip install -r requirements.txt
python 大实验/examples/quickstart.py     # 完整示例：离线记忆 / 知识库导入 / 持久化
```

## 4. 安装

- Python ≥ 3.10，Windows / Linux 均可用
- 核心依赖：`pip install -r requirements.txt`（numpy / httpx / fastapi / uvicorn / matplotlib / networkx / hnswlib / python-multipart）
- 或安装本包：`pip install -e .`（hnswlib 在 `[ann]` extra 中，按需 `pip install -e ".[ann]"`；Region 多时自动启用 ANN 加速，缺省回退精确扫描）
- 可选本地 embedding：`pip install -e ".[local-embeddings]"`（sentence-transformers，如 `BAAI/bge-small-zh-v1.5`）

## 5. 配置

### 5.1 三种配置方式（同一套 89 项可调项）

| 方式 | 说明 |
|---|---|
| **公共配置 `sme/config.json`** | 随包分发、默认零配置离线、无密钥；通过 `SpatialMemoryEngine(config_path="sme/config.json")` 显式传入（引擎不会自动读取该文件；环境变量 `SME_CONFIG_PATH` 仅 REST 服务端生效，见 5.5）；文件顶部 `_help` 是每项中文说明 |
| **配置菜单 `python -m sme.menu`** | 交互式浏览/修改全部可调项（带说明与校验），**改动立即写回 config.json**；预设一键套用 |
| **代码方式 `SMEConfig`** | 任意一项都可代码设置（见 5.4） |

> 无参 `SpatialMemoryEngine()` 使用代码内置 dataclass 默认（如 `llm.base_url="https://api.openai.com/v1"`、
> `llm.model="gpt-4o-mini"`、`llm.temperature=0.3`，与 config.json 的空默认不同）；无密钥时 LLM 保持未配置（纯离线）。

### 5.2 配置菜单（推荐）

```bash
python -m sme.menu                       # 交互菜单（选组→选项→改值→自动写回）
python -m sme.menu --list                # 列出全部 89 项（含说明）
python -m sme.menu --show                # 查看当前配置内容
python -m sme.menu --check [--ping]      # 验证配置 + LLM/embedding 连通性（含维度校验）
python -m sme.menu --set llm.model=deepseek-v4-flash   # 非交互设置（可多次）
python -m sme.menu --preset kb_static    # 非交互套用预设
python -m sme.menu --config path.json    # 指定配置文件
```

### 5.3 代码方式设置

```python
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine

config = SMEConfig()
config.policy.decay_enabled = False          # 关闭衰减
config.region.auto_evolve = False            # 关闭演化
config.storage.path = "my_state.json.gz"
engine = SpatialMemoryEngine(config)         # 或 SpatialMemoryEngine(config_path="sme/config.json")
```

### 5.4 环境变量（仅 REST 服务端生效）

所有 `SME_*` 环境变量**仅在 `python -m sme.api` 启动 REST 服务时生效**（`sme/api/server.py` 的 `build_engine_from_env` 读取）；
直接使用 `SpatialMemoryEngine()`（SDK 直连）**不读取任何环境变量**，密钥需代码传入（示例见 [docs/接入使用.md](docs/接入使用.md) §3.5）：

| 变量 | 作用 |
|---|---|
| `SME_CONFIG_PATH` | REST 服务读取的配置文件路径（等价 `--config`） |
| `SME_LLM_BASE_URL` / `SME_LLM_MODEL` / `SME_LLM_API_KEY` | REST 服务端 LLM 配置（设了 BASE_URL 才读 KEY） |
| `SME_EMBEDDING_PROVIDER` / `MODEL` / `DIM` / `BASE_URL` / `API_KEY` | REST 服务端 embedding 配置（设了 PROVIDER 才读 KEY） |
| `SME_API_AUTH_TOKEN` | REST 服务 Bearer 鉴权（等价 `api.auth_token`） |

> 89 项配置完整总览（分组、默认值、逐项说明）与 `memory.*` 会话层参数说明见 [docs/接入使用.md](docs/接入使用.md) §3.3。

## 6. 接入 AI：LLM 与 Embedding

> LLM 用于生成融合/压缩摘要、事实提取、问答对回放等；基础写/查记忆**不需要 LLM**（离线 hashing 即可跑通）。

### 6.1 最小配置（改 sme/config.json 或菜单）

```json
{
  "llm": {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "",
    "model": "deepseek-v4-flash",
    "reasoning_effort": "none"
  },
  "embedding": {
    "provider": "sentence-transformers",
    "model": "BAAI/bge-small-zh-v1.5",
    "dim": 512
  }
}
```

配置后验证：`python -m sme.menu --check --ping`（发真实请求测连通）。

### 6.2 云端 LLM 服务商对照表（OpenAI 兼容 /chat/completions）

| 服务 | base_url | 备注 |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | 官方 |
| DeepSeek | `https://api.deepseek.com/v1` | 便宜；v4-flash 需 `reasoning_effort=none`；无 embedding 接口 |
| Qwen（通义） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 兼容模式 |
| GLM（智谱） | `https://open.bigmodel.cn/api/paas/v4` | |
| OpenRouter | `https://openrouter.ai/api/v1` | 聚合 |
| SiliconFlow | `https://api.siliconflow.cn/v1` | 聚合，含免费 BGE embedding |
| vLLM | `http://localhost:8000/v1` | 自部署 |
| LM Studio | `http://localhost:1234/v1` | 本地 |
| Ollama | `http://localhost:11434/v1` | 本地 |

> 推理型模型（deepseek-v4-flash 等）务必配 `reasoning_effort: "none"`，否则小 max_tokens 时返回空回复。

### 6.3 本地 AI 怎么用

**本地 LLM（Ollama 示例）**：先 `ollama pull qwen2.5:7b`，再在配置里填：

```json
{
  "llm": {
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model": "qwen2.5:7b",
    "reasoning_effort": "none"
  }
}
```

LM Studio / vLLM 同理，只需把 `base_url` 换成对应端口（LM Studio `http://localhost:1234/v1`，vLLM `http://localhost:8000/v1`）。

**本地 Embedding**：`embedding.provider = "sentence-transformers"`，`model` 填本地模型名（如 `BAAI/bge-small-zh-v1.5`，dim 512）；首次运行自动下载到本地缓存，之后离线可用。

### 6.4 Embedding 三选一

| provider | 适用 | 说明 |
|---|---|---|
| `hashing` | 离线演示/零依赖 | 确定性伪向量，中文效果弱于真实模型 |
| `openai` | 任意兼容 API | 需 `base_url` + `api_key` + `model`（如 BAAI/bge-m3, dim 1024） |
| `sentence-transformers` | 本地 | 需 `pip install sentence-transformers`；`model` 如 `BAAI/bge-small-zh-v1.5`（dim 512） |

## 7. 实战：给你的 AI 接入记忆（记录 + 检索闭环）

配置只是第一步。这一节演示"配好之后"在你的 AI 里怎么用——一个完整的
带记忆 AI 助手循环，覆盖记录、检索、注入、回答、强化、持久化全流程。

### 7.1 核心闭环（四步）

```python
# 1. 记录：用户说了什么，就记什么
engine.add("用户喜欢苹果", source="user")

# 2. 检索：拿到当前问题，先查记忆
hits = engine.search("用户喜欢什么水果？", top_k=6)

# 3. 注入：把命中记忆拼进 prompt，再让 LLM 回答（模板见 7.2）
messages = build_prompt(question, hits, history)

# 4. 反馈：回答用上了哪条记忆，就强化哪条（越聊越懂）
engine.reinforce(hits[0].memory.id)
```

### 7.2 完整示例：带记忆的 AI 助手

LLM 调用使用项目自带的 `sme.llm.LLMClient`（自动读配置的 `llm.*`，
也可以用 `engine.llm` 直接拿）；如果你已有自己的 LLM 调用方式，
只需替换第 3 步（回答）里的 `llm.chat(...)` 调用，SME 部分不受影响。

```python
from sme.engine import SpatialMemoryEngine

def build_prompt(question: str, hits, history) -> list[dict]:
    memory_block = "\n".join(
        f"- (相关度 {h.score:.2f}) {h.memory.text}" for h in hits
    ) or "（暂无相关记忆）"
    history_block = "\n".join(
        f"{role}: {text}" for role, text in history[-8:]
    ) or "（无）"
    return [
        {"role": "system", "content": "你是用户的 AI 助手，拥有长期记忆。"
         "【记忆】是用户过去说过的内容，回答时可参考；与当前问题无关就忽略。"},
        {"role": "user", "content":
            f"【相关记忆】\n{memory_block}\n\n"
            f"【最近对话】\n{history_block}\n\n"
            f"【当前问题】\n{question}"},
    ]

engine = SpatialMemoryEngine()   # 零配置离线（内置 hashing）；要接真实 LLM/embedding 请先按第 5-6 章配置
llm = engine.llm                 # LLMClient（未配置 llm.* 时 llm.configured=False）
history: list[tuple[str, str]] = []   # 你自己维护的最近对话（role, text）
writes = 0                            # 写入计数（用于周期融合/压缩）

while True:
    question = input("你: ").strip()
    if question in ("exit", "quit"):
        break

    # 1) 记录：用户发言入库（source 标记来源；开启 extraction 后自动只存事实）
    engine.add(question, source="user")
    writes += 1
    history.append(("用户", question))

    # 2) 检索：取 top_k 条相关记忆，可选沿记忆图扩展
    hits = engine.search(question, top_k=6, graph_expand=1)

    # 3) 回答：注入记忆 + 最近对话，调 LLM
    messages = build_prompt(question, hits, history)
    if llm.configured:
        reply = llm.chat(messages, max_tokens=512)
    else:
        reply = "（未配置 LLM，仅演示记忆闭环）" + "\n".join(
            f"  [{h.score:.2f}] {h.memory.text}" for h in hits
        )
    print(f"AI: {reply}")
    history.append(("助手", reply))

    # 4) 强化：这次回答用到的记忆（对应 memory.reinforce_on）
    if hits:
        engine.reinforce(hits[0].memory.id)

    # 周期维护：按会话层语义触发（对应 memory.consolidate_every / compress_every）
    if writes % 8 == 0:
        engine.consolidate()
    if writes % 16 == 0:
        engine.compress()

    engine.save("data/engine.json")   # 落盘（autosave 默认开，手动更保险）
```

> 更完整的提示词组装（用户画像常驻、token 预算、对话窗口）可参考
> `sme/context.py` 的 `ContextManager.build`（模块 11 分层上下文）。

### 7.3 配置项 → 调用映射（会话层语义落实）

引擎本身不消费 `memory.*`，接入聊天程序时按此表手动落实
（这也是 `大实验/scripts/baselines/sme_adapter.py` 的复刻方式）：

| 配置项（sme/config.json） | 对应调用 |
|---|---|
| `memory.top_k`（6） | `engine.search(q, top_k=6)` —— 每次注入 prompt 的记忆条数 |
| `memory.reinforce_on`（开） | 命中后调 `engine.reinforce(hit.memory.id)` |
| `memory.graph_expand`（1） | `engine.search(q, graph_expand=1)` —— 沿记忆图扩展关联记忆 |
| `memory.consolidate_every`（8） | 每写入 8 条调一次 `engine.consolidate()`（off=永不） |
| `memory.compress_every`（16） | 每写入 16 条调一次 `engine.compress()`（off=永不） |
| `memory.window_rounds`（20） | 注入最近 20 轮对话历史（超 token 预算丢最旧） |
| `memory.persist_path` | `engine.save(路径)` 的快照位置（留空则不持久化） |

> 区分：`retrieval.top_k`（默认 10）是引擎一次能返回的**候选上限**；
> `memory.top_k` 是你要**真正拼进 prompt** 的条数。两者可以不同——
> 例如 `search(top_k=10)` 多取候选、注入时只留前 6 条。

### 7.4 不同场景的接法

| 场景 | 接法 |
|---|---|
| 你的 Agent / 工具脚本 | 直接用 Python SDK（进程内共享一个 engine，API 见 [docs/接入使用.md](docs/接入使用.md) §5） |
| 聊天机器人（QQ/微信/Telegram 等） | 跑 `python -m sme.api`，机器人进程走 REST（见第 8 章） |
| 多用户服务 | 写/查都传 `ns="用户id"`；并开 `namespaces.enabled` |
| 知识库问答 | `import_documents` 入库，检索用 `metadata_filters` 圈定文档范围 |
| 只记"用户偏好/事实" | 开 `extraction.enabled`（自动只存干净事实，AI 回答默认不入库）+ 预设 01 |

## 8. 接入聊天软件 / 其他进程（REST API）

REST 服务适合把 SME 独立跑成一个"记忆服务"，供 QQ / 微信 / Telegram / Slack 机器人、
Web 后端、其他语言（JS/Go/Java 等）的项目通过 HTTP 调用。

### 8.1 启动

```bash
python -m sme.api [--config path.json] [--host 127.0.0.1] [--port 8000]
# 或：set SME_CONFIG_PATH=my_config.json 后 python -m sme.api
# 文档（Swagger UI）: http://127.0.0.1:8000/docs
```

`--host 0.0.0.0` 可让局域网其他机器访问。`api.auth_token`（或 `SME_API_AUTH_TOKEN`）非空即启用 Bearer 鉴权。

### 8.2 最小示例（curl + MemoryClient）

```bash
curl -X POST http://127.0.0.1:8000/memories -H "Content-Type: application/json" \
  -d '{"text": "用户喜欢苹果", "tags": ["fruit"]}'
curl -X POST http://127.0.0.1:8000/memories/search -H "Content-Type: application/json" \
  -d '{"text": "用户喜欢什么水果？", "top_k": 5, "graph_expand": 1}'
```

```python
from sme.api.client import MemoryClient

sdk = MemoryClient("http://127.0.0.1:8000", api_key="token")   # api_key 可选（鉴权时必填）
sdk.add("用户喜欢打篮球", tags=["sport"])
hits = sdk.search("喜欢什么运动", top_k=3)
sdk.close()
```

### 8.3 聊天软件接入推荐流程

1. **入库**：用户发言 → `POST /memories`（带 `ns` 区分用户）
2. **召回**：当前消息 → `POST /memories/search`，得到相关历史事实/偏好
3. **作答**：把命中记忆拼进提示词，交给你的 LLM 生成回复
4. **反馈**：回复发出后强化命中记忆 → `POST /memories/{id}/hit`（越聊越懂）

> 完整端点表、鉴权细节、MemoryClient 全方法见 [docs/接入使用.md](docs/接入使用.md) §6。

## 9. 预设场景（一键套用）

```bash
python -m sme.menu --preset chat        # 01 聊天助手（默认）：强化/衰减/融合/压缩全开
python -m sme.menu --preset kb_dynamic  # 02 知识库·动态：知识不衰减，越查越重要
python -m sme.menu --preset kb_static   # 03 知识库·静态：纯只读，结果可复现
python -m sme.menu --preset robot       # 04 具身机器人：WAL 崩溃安全 + 多用户隔离
python -m sme.menu --preset minimal     # 05 全关：当普通向量库用
```

> v2 模块（事实提取/问答对/纠错/图谱/画像/噪音抑制）默认全关 = v1 行为；
> 面向**中文对话**场景按需开启（`extraction.enabled` 等）。全开并非最优配置。

## 10. 常见问题（FAQ）

| 问题 | 回答 |
|---|---|
| 记忆会丢吗？ | 不会：衰减只降命中概率永不删除；归档可恢复；快照原子写防损坏；WAL 崩溃自动恢复 |
| LLM 返回空字符串？ | 推理型模型（deepseek-v4-flash 等）配 `llm.reasoning_effort = "none"` |
| 本地模型怎么用？ | LLM 指向本地服务（Ollama/LM Studio/vLLM）的 OpenAI 兼容端点；Embedding 用 `sentence-transformers` + 本地模型 |
| 中文效果差？ | 换更强 embedding（BGE-m3）；BM25 中文 1-2 元切分默认开启；文档资料用 `engine.import_documents` 分条入库 |
| 多用户隔离？ | 写入/检索传 `ns` 参数（`engine.add(..., ns="user_a")` / `engine.search(..., ns="user_a")`） |
| 性能如何？ | 10k 条写入 ~4s、100k 条加载 ~3s、hashing 检索 p50 ~4ms（2k 条）/ ~22ms（10k 条，实测）；大库（≥256 Region）自动启用 ANN 加速 |

## 11. 评测

```bash
cd 大实验
python benchmarks/generate_assets.py            # 生成评测资产（seed 固定，可复现）
cd ..
python -m sme.benchmark --eval 大实验/benchmarks/qa183.json       # 183 题英语考问
python -m sme.benchmark --eval 大实验/benchmarks/zh_law.json      # 中文法律考问
python -m sme.benchmark --eval 大实验/benchmarks/zh_medical.json  # 中文医疗考问
python -m sme.benchmark --n-memories 2000                 # 写入/检索压测
```

## 12. 大实验（与主流记忆方案的擂台赛）

`大实验/` 是独立实验工程（**不修改 sme/ 主体代码**）：与 mem0 / langmem / 裸 RAG / BM25
在"同一对话流 + 同一 embedding + 同一 LLM + 同一考问集"的公平协议下对比。

- [大实验/大实验全档_合并.md](大实验/大实验全档_合并.md) — 工程全档（四篇文档合并版）：
  公平协议与跑法、实验原理与方法、标准评测资产、两轮擂台赛全量数据与逐题明细、
  攻击题、消融矩阵、成本、优势分析与复现命令

主要结论：对话记忆端到端 acc 0.421-0.441（3 seed 均值）仅次于 mem0 0.477（单 seed），
是裸 RAG 0.123-0.128 的 3 倍以上；双重纠错/噪音霸榜等攻击题下版本管理与噪音抑制实锤有效。

## 13. 目录结构

```
├── sme/                    # 插件本体（引擎/空间/检索/动力学/v2 模块/REST）
├── docs/                   # 接入使用 / 原理解析 / 迭代计划 / 修改记录
├── 大实验/                 # 独立实验工程：全档文档 + 脚本/数据/结果 + 评测资产 + 示例 + 回归测试
│   ├── 大实验全档_合并.md   # 工程全档（合并自 README/EXPERIMENTS/REPORT/benchmarks 四篇）
│   ├── benchmarks/         # 标准评测资产（qa183 / paraphrase30 / 法律 / 医疗）
│   ├── examples/           # 可运行示例（quickstart.py）
│   ├── tests/              # 回归测试（pytest，79 项）
│   └── scripts/ results/   # 实验脚本与数据
├── requirements.txt / pyproject.toml
└── README.md
```

## 14. 密钥安全

- 默认零密钥：`sme/config.json` 随包分发的配置中所有密钥字段为空，可安全入库
- **REST 服务模式**：密钥走环境变量 `SME_LLM_API_KEY` / `SME_EMBEDDING_API_KEY` / `SME_API_AUTH_TOKEN`，不落盘（见 5.4）
- **SDK 直连模式**：引擎不读环境变量，运行时从你自己的环境变量读入 `SMEConfig`（示例见 [docs/接入使用.md](docs/接入使用.md) §3.5），
  或把含密钥的配置文件放在不入库的路径（如 `~/.sme/config.json`）用 `config_path` 加载
- 引擎运行时状态文件（`data/`）已被 .gitignore 排除

## 15. 文档导航

| 文档 | 内容 |
|---|---|
| [docs/接入使用.md](docs/接入使用.md) | 完整接入参考：89 项配置、环境变量、Python SDK 全 API、REST 端点表、预设、FAQ |
| [docs/原理解析.md](docs/原理解析.md) | 程序原理 + 每个文件/函数的作用（函数级解析） |
| [docs/迭代计划.md](docs/迭代计划.md) | 对标第一梯队（Mem0/Zep/Graphiti/Letta）的迭代记录 |
| [docs/修改记录.md](docs/修改记录.md) | 代码审查修复/完善/清理 + 修复前后对比数据 |
| [大实验/examples/quickstart.py](大实验/examples/quickstart.py) | 可运行示例：离线记忆 / 知识库导入 / 持久化 |
| [大实验/大实验全档_合并.md](大实验/大实验全档_合并.md) | 与 mem0/RAG/BM25 擂台赛全档（协议、数据、复现命令） |

> 注：原 `快速接入指南.md`、`项目介绍.md` 已合并进本文档。

## License

待定（发布前补充）。
