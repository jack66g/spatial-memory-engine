# 对接指南 —— mem3d 3D 空间记忆 × DeepSeek Harness

> 本指南包含两部分：
> **第一章 · 对接提示词** —— 粘贴给任意 AI 会话，AI 即可接管/维护这套记忆系统；
> **第二章 · 对接介绍** —— 面向人的完整架构、安装、使用与排障文档。
>
> 项目位置：你自己的路径位置（`sme/` 为 SME 引擎本体，**禁止修改**；`mem3d/` 为对接层）

---

## 第一章 · 对接提示词（给 AI 自己）

> 以下整段可直接作为提示词注入新会话，或由用户在对话开头粘贴。核心目标：
> 让 AI 明白自己身上挂着一套 3D 空间记忆系统，知道怎么用、怎么维护、崩了怎么修。

```
【角色】你是运行在 DeepSeek Harness 上的 AI，身上挂接了一套 3D 空间记忆系统
（mem3d，基于 SME 空间记忆引擎）。你负责使用和维护它，遵循以下规则。

【系统构成】
1. Python 边车 bridge：http://127.0.0.1:8756（SME 引擎多模式托管 + PCA 三维投影 + HTTP JSON API）
2. 持久化前端：浏览器里右下角 🧠 悬浮球、侧栏「3D 记忆」按钮、设置 →「3D 记忆」分区
   （由 web profile 的 mem3d-plugin bundle 提供，随 DSH 启动自动加载，直连 bridge）
3. 持久化 Host：agent preset「3D记忆模式」（~/.dsh/.agent-presets/sme3d/）里的
   mem3d-host 行加载 mem3d/host-preset-live.js —— 注册记忆工具、会话自动写入、边车管理
4. 数据目录：mem3d/data/（每个模式一个独立子目录 engine.json.gz；modes.json 当前模式；
   auto.json 自动写入开关）
5. 全部对接代码在 mem3d/ 与 mem3d-host/；绝不修改 sme/ 源码（只调用其公开 SDK）

【你的记忆工具】
- sme3d_status()：引擎状态（当前模式/记忆条数/Region 数/自动写入统计）
- sme3d_remember(text, role?)：显式写入一条重要记忆（role 默认 user）
- sme3d_recall(text, top_k?)：检索相关记忆（带相关度分数）
- sme3d_mode(id?)：列出全部模式 / 切换模式
（若这些工具不存在：说明当前会话没有挂 Host 半区，按下方【恢复流程】处理）

【自动写入规则】
- 用户消息自动入库（去重、截断 2000 字符、过滤内部 thinking 块）
- 助手消息一律不自动入库（防止思考/回复污染记忆）
- 需要记住的助手侧结论，用 sme3d_remember 显式写入
- 自动写入可在设置「3D 记忆」分区用开关关闭（持久化到 bridge auto.json）

【8 种记忆模式（存储完全独立，切换即换库）】
chat 对话记忆（默认）/ semantic 语义记忆（本地 BGE 中文向量）/ focus 只记事实 /
v2 深度对话（v2 全模块）/ kb_dynamic 知识库动态 / kb_static 知识库静态 /
robot 具身机器人（WAL+多用户隔离）/ minimal 裸向量库
- 切换到从未写过的模式 = 空白记忆，UI 与 sme3d_mode 都会明确提示
- 切回旧模式记忆仍在；当前模式持久化在 modes.json，重启不丢

【恢复流程（重启/换会话后）】
- 新会话：让用户在启动页选择「3D记忆模式」preset —— 工具/自动写入/边车全部自动生效
- 当前会话临时恢复：用 cordis 动态插件机制重新激活 Host（host-only，无需浏览器批准）
- 浏览器 UI 由 bundle 提供，重启后自动在（若不在，检查 web profile 的 cordis.patch.yml
  是否还有 mem3d-client entry 与 node_modules 里的 mem3d-plugin 链接）
- bridge 是懒启动：第一次工具调用或事件写入会自动拉起；端口 8756 被占则复用已有实例
- 若 sme3d_status 返回 alive:false，先调用一次任意 sme3d_* 工具触发 ensureBridge 重试

【排障速查】
- 工具 unknown → 会话没挂 Host（见恢复流程）
- 悬浮球不在 → 页面缓存/重启后 bundle 未进 boot 图（查页面 HTML 是否含 mem3d-plugin；
  检查 ~/.dsh/profiles/web/ 的 package.json 依赖与 cordis.patch.yml）
- bridge 写入 Permission denied → 沙箱策略：所有 shell 调用必须带
  sandboxPolicy { mode:'workspace-write', workspaceRoot, sessionId }（走 seam 受管 grant）
- semantic 模式卡住 → HF 必须离线（HF_HUB_OFFLINE=1），模型缓存
  ~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5 不存在则回退 hashing
- 页面报错信息 → 从 host-boot.json / bridge 响应 error 字段定位，不要瞎猜

【约定】
- 记忆内容保持简洁事实句（主语归一为"用户"）
- 不向用户展示内部思考；涉及记忆操作时说明"已写入/已检索"
- 系统是单用户本地部署，无需多用户隔离心智负担（robot 模式除外）
```

---

## 第二章 · 对接介绍

### 1. 是什么

`mem3d/` 是把 **SME（Spatial Memory Engine）长期记忆引擎**对接进 **DeepSeek Harness**
的完整实现：聊天对话自动沉淀为空间记忆，浏览器侧提供可拖拽的 3D 记忆空间
（记忆点、Region 聚落、质心、邻居连线、新写入高亮），设置面板提供 8 种
独立记忆模式的切换与维护。

**核心原则：不修改 `sme/` 一行源码** —— bridge 只调用 `sme.engine` / `sme.config` 的公开 SDK。

### 2. 架构总览

```
┌───────────────────────── DeepSeek Harness ─────────────────────────┐
│                                                                     │
│  ┌─ web profile（独立 Context，只放浏览器面）────────────────────┐   │
│  │  cordis.patch.yml → mem3d-plugin（dsh.client 双面包）         │   │
│  │  ├─ host-stub.js（占位，空插件）                              │   │
│  │  └─ client-bundle.js → 注入 __DSH_BOOT__ 图                  │   │
│  │       └─ 悬浮球 / 侧栏按钮 / 设置「3D 记忆」分区               │   │
│  │            │ fetch 直连（CORS 已开）                          │   │
│  └────────────┼───────────────────────────────────────────────────┘   │
│               ▼                                                     │
│  ┌─ agent preset「3D记忆模式」（主进程 ctx，Host 半区）─────────┐   │
│  │  mem3d-host 行 → mem3d/host-preset-live.js                   │   │
│  │  ├─ sme3d_* 记忆工具（tools.register）                       │   │
│  │  ├─ 会话事件自动写入（session/event + inbox/inserted）        │   │
│  │  └─ 边车懒启动/复用（shell + seam 沙箱 grant）                │   │
│  └────────────┼───────────────────────────────────────────────────┘   │
└───────────────┼───────────────────────────────────────────────────────┘
                ▼
   ┌──────────────────────────────┐
   │  mem3d/bridge.py（127.0.0.1:8756）│
   │  8 种模式引擎（各自独立存储）      │
   │  PCA 三维投影 / recall / 模式切换  │
   │  /auto 自动写入开关（持久化）      │
   └──────────────────────────────┘
```

### 3. 组件清单

| 路径 | 作用 |
|---|---|
| `mem3d/bridge.py` | Python 边车：SME 引擎托管、3D 场景、HTTP JSON API（纯标准库 + CORS） |
| `mem3d/host-preset-live.js` | **持久化 Host 半区**（agent preset 加载）：工具/自动写入/边车管理 |
| `mem3d/client-bundle.js` | **持久化浏览器半区**（`__ModuleLoader__` factory bundle，直连 bridge） |
| `mem3d/package.json` / `host-stub.js` | `dsh.client` 双面包声明（web profile 扫描用） |
| `mem3d/host-plugin.js` / `client-plugin.js` | 动态插件版参考实现（会话内临时加载用） |
| `mem3d-host/` | 曾尝试的 web profile Host 包（**已废弃**，见 §6 教训） |
| `mem3d/data/` | 运行时数据：`<mode>/engine.json.gz`、`modes.json`、`auto.json` |
| `~/.dsh/profiles/web/cordis.patch.yml` | web profile patch：`mem3d-client` entry（浏览器 bundle） |
| `~/.dsh/.agent-presets/sme3d/agent.cordis.yml` | 用户 preset「3D记忆模式」：`mem3d-host` 行（Host 半区） |

### 4. 安装与持久化（已完成，作为记录）

**浏览器半区（web profile）**
```bash
dsh plugin --profile web add 路径你自己的
# 在 ~/.dsh/profiles/web/cordis.patch.yml 中加入：
# - insert:
#     - id: mem3d-client
#       name: mem3d-plugin
```
**Host 半区（agent preset）**：复制 `cordis` 预设为 `sme3d`（用户级），在其
`agent.cordis.yml` 末尾挂载：
```yaml
- id: mem3d-host
  name: 'C:/Users/黄小乐/Desktop/mem/mem3d/host-preset-live.js'
```
（preset 加载器支持绝对路径 → file URL；`tool-cordis` 在该 preset 中禁用，
避免与其他 cordis 会话的进程全局 Inspect provider 冲突。）

**使用方式**：新会话在启动页选择「3D记忆模式」预设；浏览器 UI 随 DSH 启动自动加载。

### 5. bridge HTTP API（127.0.0.1:8756）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（含当前模式） |
| GET/POST | `/auto` | 自动写入开关（POST `{on}`；持久化到 auto.json） |
| GET | `/modes` | 8 种模式列表（名称/说明/条数/当前） |
| GET | `/scene` | PCA 三维投影场景（nodes/centroids/edges/recent） |
| GET | `/stats` | 当前模式引擎统计 |
| GET | `/list` | 当前模式记忆列表 |
| POST | `/memory` | 写入 `{text, role, source}`；`source=auto` 受开关拦截 |
| POST | `/recall` | 检索 `{text, top_k}` |
| POST | `/mode` | 切换模式 `{id}`（返回空白提示） |
| POST | `/reinforce` | 命中强化 `{id}` |
| POST | `/delete` / `/clear` | 删单条 / 清空当前模式 |
| POST | `/shutdown` | 关闭服务 |

### 6. 关键设计决策与踩坑记录

1. **web profile 是独立 Context**：`dsh-app-boot` 的 `boot()` 用 `new Context()` 创建隔离
   运行环境，web profile 插件**拿不到** `shell`/`tools`/会话事件。→ Host 逻辑只能放
   agent preset（主进程 ctx）或动态插件；web profile 只放浏览器 bundle。
2. **patch entry 不能写 Windows 绝对路径**：include 加载器把 entry name 原样交给
   ESM `import()`，`C:/...` 抛 `ERR_UNSUPPORTED_ESM_URL_SCHEME` 导致 dsh web 启动崩溃。
   → 用 `dsh plugin --profile web add`（pnpm link）装包后用**包名裸引用**。
3. **dsh.client 包的三个坑**：`exports` 必须放行 `./package.json`；host 面必须导出
   合法插件（`{ apply() {} }`）；bundle 里 `__ModuleLoader__.load` 的 id 必须与
   boot 图 entry id 一致。
4. **沙箱写权限**：所有 shell 调用带 `sandboxPolicy { mode:'workspace-write',
   workspaceRoot, sessionId }` 走 seam 受管 grant（runner 自管 grant 会随进程退出
   撤销 ACE，与其他受限进程互相干扰）。
5. **assistant 消息不入库**：事件里 role 可能藏在 `data.message.role`；助手消息
   （含内部思考）一律跳过，只自动写 user 消息。
6. **semantic 模式强制 HF 离线**：模型未缓存则回退 hashing，绝不联网下载挂起。
7. **自动写入开关在 bridge 层拦截**（`source=auto`），工具显式写入不受限；
   开关持久化，重启保持。

### 7. 使用流程（用户视角）

1. 打开 DeepSeek Harness（悬浮球/侧栏按钮/设置分区随启动出现）
2. 新会话选择「3D记忆模式」预设（获得记忆工具与自动写入）
3. 正常聊天 —— 用户消息自动写入；打开悬浮球可实时观察记忆点生长
4. 设置 →「3D 记忆」：切换 8 种模式（含空白提示）、自动写入开关、清空当前模式
5. 想让我"想起"什么：直接问，我调用 `sme3d_recall` 检索记忆

### 8. 权限与安全说明

- 单用户本地部署；bridge 仅监听 127.0.0.1，CORS 允许任意 Origin（仅本机可访问）
- 自动写入仅来自会话事件，文本截断 2000 字符，去重防重复入库
- 不改动 `sme/` 源码；`~/.dsh` 改动仅限用户级 profile 与 preset
