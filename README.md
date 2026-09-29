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
- **扩展模块层（12 个，默认全关 = 基础行为）**：事实提取、问答对回放、时序知识图谱、用户画像、事实版本纠错、噪音抑制、WAL 增量持久化、存储后端、REST+SDK、可观测性、分层上下文、多用户隔离
- **检索 1.3**：min-max 混合融合（3 seed A/B 实测 acc +2.5~4.2pp、带偏减半）+ Personalized PageRank 图扩展（HippoRAG 2 路线，桥接关联自动浮出）；可选空闲期自动整理（sleep-time compute）与 ONNX int8 嵌入（`[onnx]` extra）

一句话：**把你的 AI 从"聊完就忘"变成"越聊越懂你"**。可直接接入聊天机器人、知识库问答、具身机器人、Agent 等任意需要记忆的场景。

## 2. 核心特性

| 特性 | 说明 |
|---|---|
| 零配置离线 | 内置 hashing embedding，无任何依赖/密钥即可跑通全链路 |
| 可解释检索 | 每条命中带 `breakdown`（semantic/importance/freshness/weight/decay/hit_count/recency/region 八信号） |
| 中文友好 | 中文 1-2 元 BM25 关键词通道（默认开）+ 文档分条导入（法律/医疗等知识库场景） |
| 记忆生命周期 | 强化/衰减/融合/压缩可配可关，记忆永不因衰减被删除，归档可恢复 |
| 全部可关 | 97 项配置 + 5 个预设（聊天助手/知识库动态/知识库静态/具身机器人/全关） |
| 多种接入 | Python SDK、REST API（FastAPI）+ 官方 Client、**MCP（智能体原生，一行挂载）**、Web 配置中心（`python -m sme.api` 启动后浏览器打开） |
| 性能 | WAL 组提交 2557 写/s（1.79x）；ONNX int8 批量嵌入 7.4x；记忆图热路径 O(E)→O(deg) 实测 127-159x；hashing 检索 p50 ~4ms（2k 条） |

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
engine.visualize("space.png")           # 2D 空间可视化（需可选 [viz] extra，见第 4 章）
engine.save("memory_state.json.gz")     # 持久化

engine2 = SpatialMemoryEngine()
engine2.load("memory_state.json.gz")    # 重新加载（重启不丢记忆）
```

```bash
pip install -r requirements.txt
```

## 4. 安装

- Python ≥ 3.10，Windows / Linux 均可用
- 核心依赖：`pip install -r requirements.txt`（numpy / httpx / fastapi / uvicorn / networkx / python-multipart / hnswlib）
- 或安装本包：`pip install -e .`（hnswlib 在 `[ann]` extra 中，按需 `pip install -e ".[ann]"`；Region 多时自动启用 ANN 加速，缺省回退精确扫描）
- 可选本地 embedding：`pip install -e ".[local-embeddings]"`（sentence-transformers，如 `BAAI/bge-small-zh-v1.5`）
- 可选可视化：`pip install -e ".[viz]"`（matplotlib）。核心依赖已不含 matplotlib；未安装时 `engine.visualize()` 与 REST `/visualize`（返回 501）会给出中文安装提示

## 5. 配置

### 5.1 三种配置方式（同一套 89 项可调项）

| 方式 | 说明 |
|---|---|
| **Web 配置中心（推荐）** | `python -m sme.api` 启动服务后浏览器打开 `http://127.0.0.1:8000/`；全部 89 项分组展示、逐项校验、预设一键套用、测试连接；**保存到 `data/sme.config.json`**，引擎热重建立即生效（详见 5.2） |
| **显式 JSON 文件** | 自己写一个 JSON 配置文件：SDK 用 `SpatialMemoryEngine(config_path="路径")` 显式传入；REST 服务用环境变量 `SME_CONFIG_PATH` 或 `--config` 指定（见 5.4） |
| **代码方式 `SMEConfig`** | 任意一项都可代码设置（见 5.3） |

> 配置来源只有一套：**代码内置默认**（89 项注册表 `sme/config_items.py`，含每项中文说明）→
> 你通过 Web 配置中心修改，保存到 `data/sme.config.json`（`data/` 已被 .gitignore 排除；
> 需要其他位置时用环境变量 `SME_CONFIG_PATH` 或 `python -m sme.api --config 路径`）。
> 项目**不再随包分发 `config.json`**，引擎从不自动读取任何打包配置；SDK 直连请用
> `SpatialMemoryEngine(config_path="...")` 或代码构造 `SMEConfig`。
>
> 无参 `SpatialMemoryEngine()` 使用代码内置 dataclass 默认（如 `llm.base_url="https://api.openai.com/v1"`、
> `llm.model="gpt-4o-mini"`、`llm.temperature=0.3`）；无密钥时 LLM 保持未配置（纯离线）。

### 5.2 Web 配置中心（推荐）

```bash
python -m sme.api          # 启动 REST 服务（默认 127.0.0.1:8000）
# 浏览器打开 http://127.0.0.1:8000/ 即配置中心；API 文档在 /docs
```

| 功能 | 说明 |
|---|---|
| 分组展示 | 全部 89 项按功能分 9 组：模型接入 / 向量 / 检索与排序 / 空间与Region / 记忆动力学 / 存储 / 服务 / 会话层约定 / 扩展模块 |
| 详情弹窗 | 每项"详情"显示：作用说明全文、当前值/默认值、类型与可选值 |
| 逐项校验 | 输入非法值当场红字提示（类型/枚举/范围） |
| 密钥掩码 | API 密钥类项只显示是否已设置（`******`），不回显明文 |
| 预设一键套用 | 5 个预设：01 聊天助手 / 02 知识库·动态 / 03 知识库·静态 / 04 具身机器人 / 05 全关 |
| 测试连接 | 配置校验 + 可选真实 ping LLM/embedding（含维度检查），等价原 `--check --ping` |
| 保存即生效 | 保存后引擎立即以新配置**热重建**，现有记忆自动迁移（export/import 方式）；配置写入 `data/sme.config.json`，保存完关掉页面配置依然生效；改 embedding provider/model/dim 且库内有记忆时会被拒绝（防向量维度错配） |
| 重置为默认 | 一键恢复代码内置默认值（环境变量注入的配置保留） |

配置中心对应的 REST 端点（也可程序化调用）：`GET /config`（全量配置+分组+预设）、
`PUT /config`（校验+热重建+可选落盘）、`POST /config/preset`、`POST /config/check`、
`POST /config/reset`、`GET /`（配置中心页面）。端点表见 [docs/接入使用.md](docs/接入使用.md) §6.3。

### 5.3 代码方式设置

```python
from sme.config import SMEConfig
from sme.engine import SpatialMemoryEngine

config = SMEConfig()
config.policy.decay_enabled = False          # 关闭衰减
config.region.auto_evolve = False            # 关闭演化
config.storage.path = "my_state.json.gz"
engine = SpatialMemoryEngine(config)         # 或 SpatialMemoryEngine(config_path="my_config.json")
```

### 5.4 环境变量（仅 REST 服务端生效）

所有 `SME_*` 环境变量**仅在 `python -m sme.api` 启动 REST 服务时生效**（`sme/api/server.py` 的 `build_engine_from_env` 读取）；
直接使用 `SpatialMemoryEngine()`（SDK 直连）**不读取任何环境变量**，密钥需代码传入（示例见 [docs/接入使用.md](docs/接入使用.md) §3.5）：

| 变量 | 作用 |
|---|---|
| `SME_CONFIG_PATH` | REST 服务读取的配置文件路径（等价 `--config`）；未设置时默认尝试 `data/sme.config.json`（Web 配置中心的保存位置） |
| `SME_LLM_BASE_URL` / `SME_LLM_MODEL` / `SME_LLM_API_KEY` | REST 服务端 LLM 配置（设了 BASE_URL 才读 KEY） |
| `SME_EMBEDDING_PROVIDER` / `MODEL` / `DIM` / `BASE_URL` / `API_KEY` | REST 服务端 embedding 配置（设了 PROVIDER 才读 KEY） |
| `SME_API_AUTH_TOKEN` | REST 服务 Bearer 鉴权（等价 `api.auth_token`） |

> 89 项配置完整总览（分组、默认值、逐项说明）与 `memory.*` 会话层参数说明见 [docs/接入使用.md](docs/接入使用.md) §3.3。

## 6. 接入 AI：LLM 与 Embedding

> LLM 用于生成融合/压缩摘要、事实提取、问答对回放等；基础写/查记忆**不需要 LLM**（离线 hashing 即可跑通）。

### 6.1 最小配置（Web 配置中心或代码方式）

在 Web 配置中心（见 5.2）的"模型接入 / 向量"分组里把这几项改成：

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

也可以把上面的 JSON 写进自己的配置文件用 `config_path` 传入，或代码构造 `SMEConfig`（见 5.3）。

配置后验证：Web 配置中心的"测试连接"按钮（配置校验 + 可选真实 ping LLM/embedding，含维度检查）。

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
| `openai` | 任意兼容 API | 需 `base_url` + `api_key` + `model`（如 BAAI/bge-m3, dim 1024）；`embedding.dim` 必须与模型实际维度一致，错配首次嵌入即报错 |
| `sentence-transformers` | 本地 | 需 `pip install sentence-transformers`；`model` 如 `BAAI/bge-small-zh-v1.5`（dim 512） |
| `fastembed` | 本地 ONNX int8 | 需 `pip install -e ".[onnx]"`；默认 `Qwen/Qwen3-Embedding-0.6B-Q`，质量与 fp32 持平、批量快 7x；支持 MRL 截维省一半存储 |

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
> `sme/modules/context.py` 的 `ContextManager.build`（模块 11 分层上下文）。

### 7.3 配置项 → 调用映射（会话层语义落实）

引擎本身不消费 `memory.*`，接入聊天程序时按此表手动落实：

| 配置项（会话层约定，见 [docs/接入使用.md](docs/接入使用.md) §3.3） | 对应调用 |
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
# Web 配置中心: http://127.0.0.1:8000/ ；文档（Swagger UI）: http://127.0.0.1:8000/docs
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

### 8.4 MCP 接入（智能体原生）

支持 MCP（Model Context Protocol）的智能体宿主（ZCode / Claude Code / Cursor 等）可以一行配置挂载 SME 长期记忆：

```json
{"mcpServers": {"sme-memory": {
    "command": "python",
    "args": ["/path/to/sme/mcp_server.py"]
}}}
```

- 提供 4 个工具：`remember`（写记忆）/ `recall`（语义检索）/ `reinforce`（命中强化）/ `memory_stats`
- 服务未运行时自动拉起（默认 `127.0.0.1:8760`，`SME_MCP_PORT` 可改），重启自动恢复快照+WAL
- 主会话与子智能体共享同一记忆库（REST 服务单写者，无并发写风险）
- 依赖：`pip install -e ".[mcp]"`（mcp>=2.0）

## 9. 预设场景（一键套用）

在 Web 配置中心的预设下拉中选择即可一键套用；等价的 REST 调用（key 取下方预设键名）：

```bash
curl -X POST http://127.0.0.1:8000/config/preset -H "Content-Type: application/json" \
  -d '{"key": "chat"}'           # 01 聊天助手（默认）：强化/衰减/融合/压缩全开
curl -X POST http://127.0.0.1:8000/config/preset -H "Content-Type: application/json" \
  -d '{"key": "kb_dynamic"}'     # 02 知识库·动态：知识不衰减，越查越重要
curl -X POST http://127.0.0.1:8000/config/preset -H "Content-Type: application/json" \
  -d '{"key": "kb_static"}'      # 03 知识库·静态：纯只读，结果可复现
curl -X POST http://127.0.0.1:8000/config/preset -H "Content-Type: application/json" \
  -d '{"key": "robot"}'          # 04 具身机器人：WAL 崩溃安全 + 多用户隔离
curl -X POST http://127.0.0.1:8000/config/preset -H "Content-Type: application/json" \
  -d '{"key": "minimal"}'        # 05 全关：当普通向量库用
```

> 扩展模块（事实提取/问答对/纠错/图谱/画像/噪音抑制）默认全关 = 基础行为；
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
| 报"embedding 维度不匹配"？ | openai embedding 的模型实际维度与 `embedding.dim` 不一致（如 bge-m3=1024 配了 64），按报错提示改 `embedding.dim` 或换匹配模型；不做静默补齐/截断，详见 [docs/接入使用.md](docs/接入使用.md) Q12 |
| 改配置何时生效？ | Web 配置中心保存后引擎立即热重建（记忆自动迁移），关掉页面持续有效；改 embedding provider/model/dim 且库内有记忆时会被拒绝，需先导出清空 |

## 11. 评测

```bash
python -m sme.benchmark --n-memories 2000                 # 写入/检索压测
```

## 12. 目录结构

```
├── sme/                                # 插件本体
│   ├── engine.py config.py config_items.py config_check.py
│   │   models.py memory_manager.py utils.py     # 引擎 / 配置（内置默认注册表）/ 配置校验
│   ├── import_docs.py benchmark.py visualization.py
│   ├── space/  retrieval/（含 ranking、rerank）  embedding/  index/  llm/
│   ├── api/                            # REST 服务（server / client / static 配置中心前端）
│   ├── dynamics/                       # 记忆动力学：decay / reinforcement / consolidation /
│   │                                   #   compression / archive / policy
│   ├── modules/                        # 扩展模块：extraction / qapair / factversion / profile /
│   │                                   #   context / namespaces / observability / noise /
│   │                                   #   memory_graph / bridge / pipeline / factgraph
│   └── storage/                        # snapshot / backends / wal
├── docs/                               # 接入使用 / 原理解析 / 迭代计划 / 修改记录
├── requirements.txt / pyproject.toml
└── README.md
```

> 公开导入路径保持兼容：`from sme.engine import SpatialMemoryEngine`、
> `from sme.config import SMEConfig`、`from sme.retrieval import SearchQuery`、
> `from sme.storage import EngineSnapshot` 等照旧可用；
> `sme.dynamics` / `sme.modules` / `sme.storage` 提供便捷 re-export。

## 13. 密钥安全

- 默认零密钥：代码内置默认配置中所有密钥字段为空，可安全入库
- **Web 配置中心**：保存的 `data/sme.config.json` 位于已被 .gitignore 排除的 `data/` 目录，不会入库
- **REST 服务模式**：密钥走环境变量 `SME_LLM_API_KEY` / `SME_EMBEDDING_API_KEY` / `SME_API_AUTH_TOKEN`，不落盘（见 5.4）
- **SDK 直连模式**：引擎不读环境变量，运行时从你自己的环境变量读入 `SMEConfig`（示例见 [docs/接入使用.md](docs/接入使用.md) §3.5），
  或把含密钥的配置文件放在不入库的路径（如 `~/.sme/` 下）用 `config_path` 加载
- 引擎运行时状态文件（`data/`）已被 .gitignore 排除

## 14. 文档导航

| 文档 | 内容 |
|---|---|
| [docs/接入使用.md](docs/接入使用.md) | 完整接入参考：89 项配置、环境变量、Python SDK 全 API、REST 端点表、预设、FAQ |
| [docs/原理解析.md](docs/原理解析.md) | 程序原理 + 每个文件/函数的作用（函数级解析） |
| [docs/迭代计划.md](docs/迭代计划.md) | 对标第一梯队（Mem0/Zep/Graphiti/Letta）的迭代记录 |
| [docs/修改记录.md](docs/修改记录.md) | 代码审查修复/完善/清理 + 修复前后对比数据 |

> 注：原 `快速接入指南.md`、`项目介绍.md` 已合并进本文档。

## License

Apache-2.0（见 [LICENSE](LICENSE)）。
