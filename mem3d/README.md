# mem3d —— 专门对接 DeepSeek Harness 的 3D 空间记忆插件

> 在 DeepSeek Harness 里实时看见你的记忆：把对话自动沉淀进 SME（Spatial Memory
> Engine）空间记忆引擎，并在一个可拖拽的 3D 浮窗中动态呈现记忆轨迹——
> 记忆点、Region 聚落、质心与邻居连线、新增记忆高亮，随聊天实时生长。
>
> 本目录**不修改 `sme/` 任何源码**，只调用其公开 Python SDK。

---

## 1. 是什么

`mem3d/` 是把 **SME 长期记忆引擎**桥接进 **DeepSeek Harness** 的完整插件：

| 文件 | 作用 |
|---|---|
| `bridge.py` | Python 边车服务：托管 9 种独立模式 SME 引擎 + 3D 投影 + HTTP JSON API（纯标准库，含 CORS 与自动写入开关） |
| `host-plugin.js` | 动态插件版 Host 半区（含 `harness` RPC/工具，随会话临时加载用） |
| `host-preset-live.js` | 持久化版 Host 半区（普通 Cordis 插件：`ctx.tools.register` 注册记忆工具，agent preset 加载） |
| `client-plugin.js` | 动态插件版 Client 半区（随会话临时加载用） |
| `client-bundle.js` | **持久化版 Client 半区**（`__ModuleLoader__` factory bundle，直连 bridge，随 DSH 启动自动加载） |
| `package.json` / `host-stub.js` | `dsh.client` 双面包声明（web profile loader 扫描用） |
| `data/` | 运行时数据（每个模式一个独立子目录，互不共享：`engine.json.gz` 快照 + `.wal` 写前日志；`modes.json` 当前模式；`auto.json` 自动写入开关） |
| `README.md` | 本文档 |

数据流：

```
你与 AI 聊天（DeepSeek Harness）
        │ 会话事件（自动捕获 user / assistant 消息，过滤 thinking）
        ▼
host-plugin.js ──懒启动/复用──▶ bridge.py（127.0.0.1:8756）
        ▲                            │ 多模式引擎（各自独立存储）
        │ Package 私有 RPC           ▼
client-plugin.js ◀── 2.5s 轮询 ── PCA 三维投影场景 JSON
        │
        ▼
🧠 3D 记忆浮窗：点 = 记忆 / 星 = Region 质心 / 线 = Region 邻居边
            白色光圈 = 刚写入的记忆 · 悬停显示文本 · 拖拽旋转 / 滚轮缩放 / 双击复位
```

---

## 2. 八种记忆模式（存储完全独立）

模式是**互不相通的独立记忆库**：每种模式有独立配置 + 独立存储目录
（`data/<mode>/engine.json.gz`）。**切换到一个从未写过的模式 = 空白记忆**，
UI 和工具都会明确提示；切回旧模式则恢复原记忆。

| 模式 id | 名称 | 说明 |
|---|---|---|
| `chat` | 对话记忆（默认） | 强化 / 衰减 / Region 演化全开，hashing 离线向量 |
| `semantic` | 语义记忆 | 本地 BGE 中文向量（sentence-transformers，强制离线用缓存）；未缓存自动回退 hashing |
| `focus` | 只记事实 | 事实提取（离线规则）+ 问答对回放 + 噪音抑制：只沉淀干货 |
| `v2` | 深度对话 | v2 全模块：事实提取+问答对+时序知识图谱+用户画像+事实纠错+噪音抑制 |
| `kb_dynamic` | 知识库·动态 | 知识不衰减、越查越重要（命中强化） |
| `kb_static` | 知识库·静态 | 纯只读：衰减/强化/演化全关，结果可复现 |
| `robot` | 具身机器人 | 多用户隔离（模块12）+ 强化全开；WAL 崩溃安全（模块07）为全部模式统一开启 |
| `minimal` | 裸向量库 | 全关：纯向量存取，当普通向量数据库用 |
| `project` | 项目记忆 | 知识不衰减、事实提取+噪音抑制只沉淀干货、命中强化越查越重要；不同项目用【项目:xxx】前缀区分 |

切换入口：设置 →「3D 记忆」分区（8 张模式卡片）、3D 浮窗右上角下拉框、
右下角 🧠 悬浮球内的窗口，或工具 `sme3d_mode`（传 id 切换 / 不传列出）。

---

## 3. 对 AI（我）可见的记忆工具

插件向模型注册了 4 个工具，聊天中即可调用：

| 工具 | 作用 |
|---|---|
| `sme3d_remember(text, role)` | 手动写入一条重要记忆（role: user/assistant/system） |
| `sme3d_recall(text, top_k)` | 检索相关记忆，返回带相关度分数的命中 |
| `sme3d_mode(id?)` | 查看全部模式 / 切换模式（含"新模式为空白"提示） |
| `sme3d_status()` | 运行状态：模式、记忆条数、Region 数、自动写入统计 |

自动写入：用户消息与助手消息（自动跳过内部 thinking 块）会**去重后**自动入库，
无需手动操作——聊天即记忆。

---

## 4. 桥接服务 API（bridge.py）

纯标准库 HTTP JSON 服务，默认 `127.0.0.1:8756`：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（插件据此判断边车是否存活/复用） |
| GET | `/modes` | 全部模式与当前模式（含各自记忆条数） |
| GET | `/scene` | PCA 三维投影场景：nodes/centroids/edges/recent 高亮 |
| GET | `/stats` | 当前模式引擎统计（记忆/Region 动力学计数） |
| POST | `/memory` | `{text, role, source}` 写入记忆 |
| POST | `/recall` | `{text, top_k}` 检索记忆 |
| POST | `/mode` | `{id}` 切换模式 |
| POST | `/reinforce` | `{id}` 命中强化（Ebbinghaus） |
| POST | `/delete` | `{id}` 删除一条记忆 |
| POST | `/shutdown` | 关闭服务 |

手动运行：`python bridge.py --port 8756`（插件会优先复用已运行的实例）。

---

## 5. 持久化安装到 DeepSeek Harness（"启动你就一样打开"）

### 5.1 Host 半区 → agent preset ✅（已完成并挂载校验通过）

已创建用户级 preset `sme3d`（"3D记忆模式"，复制自 `cordis` 预设），在其
`agent.cordis.yml` 末尾挂载本插件：

```yaml
- id: mem3d-host
  name: 'C:/Users/<你的用户名>/Desktop/mem/mem3d/host-preset-live.js'
```

新会话在启动页选择「3D记忆模式」preset 后，记忆工具（sme3d_remember /
sme3d_recall / sme3d_mode / sme3d_status）、对话自动写入、边车懒启动全部
随会话自动运行。注意该 preset 中 cordis 动态插件工具已禁用（其 Inspect
provider 为进程全局注册，与其他 cordis 会话同时挂载会冲突）；需要动态
插件能力时请开一个「创造模式」会话。

### 5.2 Client 半区（3D 浮窗按钮）→ `dsh.client` 持久化 ✅（已配置并验证通过）

本目录已做成 `dsh.client` 双面包（`package.json` 声明 + `host-stub.js` 占位
Host 面 + `client-bundle.js` 手工 factory bundle，`require("react")` 来自前端
静态模块表），已安装进 web profile 并在 `~/.dsh/profiles/web/cordis.patch.yml`
顶层 insert 了 loader entry：

```bash
# 安装（第一次 / mem3d 目录移动后重装）：
dsh plugin --profile web add C:/Users/<你的用户名>/Desktop/mem/mem3d
```

```yaml
# ~/.dsh/profiles/web/cordis.patch.yml
- insert:
    - id: mem3d-client
      name: mem3d-plugin
```

> ⚠️ **必须用包名 `mem3d-plugin` 裸引用，不能写 Windows 绝对路径**
> （`name: 'C:/.../mem3d'`）——CLI 启动的 include 加载器会把 entry name
> 原样传给 ESM `import()`，绝对路径会抛 `ERR_UNSUPPORTED_ESM_URL_SCHEME`
> 导致 `dsh web` 启动直接崩溃（实测）。
> `mem3d/package.json` 的 `exports` 必须放行 `"./package.json"`，
> 否则 client-modules 扫描时 `require.resolve('mem3d-plugin/package.json')`
> 会报 `ERR_PACKAGE_PATH_NOT_EXPORTED`，包不会进浏览器 boot 图。
> `host-stub.js` 必须导出合法 cordis 插件（`{ apply() {} }`），
> 空对象 `{}` 会挂载失败。

**DSH 重启后**，浏览器 boot 图（`window.__DSH_BOOT__`）自动包含该包：
侧栏按钮、右下角 🧠 悬浮球、设置 →「3D 记忆」分区随页面启动即出现，
无需任何激活。持久化版前端**直连 bridge**（CORS 已开），不依赖会话内
动态插件；「自动记录对话」开关由 bridge 持久化（`data/auto.json`，仅拦截
`source=auto` 的会话自动写入，工具显式写入不受限）。

当前会话内的动态插件（memviz-1）与持久化 bundle 可并存：两者注册同一批
Slot id 时以动态插件优先（动态加载晚于 bundle）；若想只留持久化版，
停掉动态插件即可。

---

## 6. 注意事项

- **不碰 sme 源码**：本目录只 import `sme.engine` / `sme.config` 公开 API。
- 边车懒启动：首次需要时自动拉起，若端口上已有健康实例则直接复用；
  插件卸载/更新时其自行拉起的边车进程随 fiber 回收。
- 自动写入有去重与 thinking 过滤；文本最长截断 2000 字符。
- hashing 离线向量对中文短句区分度有限（同类句式余弦约 0.2–0.4），
  故 chat/kb/minimal 模式把 Region 聚合门槛放宽到 0.30 以形成可见聚落；
  追求真实语义请切 `semantic` 模式（需 `pip install sentence-transformers`）。
- 3D 投影为 PCA 三维主成分（SVD），节点与 Region 质心共用同一投影基，
  保证星标不漂移。
