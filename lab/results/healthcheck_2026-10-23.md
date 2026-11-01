# SME 全面体检与用户流程实验 · 2026-10-23

> 历史检查记录，描述 2026-10-23 当时的实现。后续修复、复测及未通过项见 [2026-11-01 验收报告](../../docs/优化验收报告_2026-11-01.md)，不以本报告判断当前版本。

**基础单条/批量 CRUD、恢复、配置网页及无鉴权 MCP 流程能用；现有测试全部通过，但补测确认 10 类实现问题。检索能找到证据，纠错后的优先级、无答案判断和若干扩展路径仍有明显边界。所有问题仅记录，没有修改源码。**

## 运行范围与保护

- 对当前未提交工作区进行测试，未回退此前优化；Git HEAD 为 `1f635e7`。
- 现有 152 个工作区文件逐文件 SHA-256 未变；初次收尾核验时生产 data/ 的 5 个文件哈希均未变。最终复核发现其他项目的并发业务会话向 WAL 新增 1 条记忆，数量由 390 变为 391；原 WAL 前缀哈希一致，其余 4 个生产文件未变。常驻进程仍为 39532。本轮没有向真实库写入测试数据，未回滚或覆盖这条业务新增。
- 用系统临时目录建立独立库；真实 HTTP 服务仅监听临时端口 65461、使用独立绝对存储和配置路径；未碰 8760。实际 MCP 测试定向该临时端口。
- 无安装、无模型下载、无云端 LLM 付费请求、无 commit/push；Qwen3 使用已缓存的本地 CPU 模型。
- 清理状态：已完成；本轮临时目录、测试数据库、日志、脚本、pytest 链接均已删除，临时服务和网页已关闭，仅保留本报告、原始证据与验证截图。

## 测试结果

| 项目 | 结果 |
|---|---|
| 原有全部回归 | 239 通过、4 跳过，70.65 秒；4 项为本地未缓存 ONNX 模型 |
| 补充用户/边界/接口/模块探针 | 77 项：49 符合检查期望、28 未符合；重复参数化和输入策略观察包含在内，不等于 28 个独立 bug |
| 随机状态机 | 3 seed × JSON/SQLite × fsync/grouped，合计 2400 个操作步骤、72 次恢复检查，全部一致 |
| 混合并发 | 4 写线程共 320 条、4 读线程共 400 次、20 次完整保存；重启后 320 条均在，无跨 ns 命中 |
| 配置网页实测 | 页面展示、非法值提示、保存合法值、刷新保留、离线配置体检均通过 |
| 真正 MCP stdio | remember/recall/reinforce/stats 与 ns 查询通过；开启服务鉴权后失败 |

原有回归覆盖了进程退出与 I/O 故障注入，但“套件全绿”不能替代所有用户工作流的验证。输入 top_k=0、负数或未知 region 模式的接受行为属于接口策略观察，未单独列为确定实现 bug。

## 已确认问题

### B01 · P1 · 问答对跨用户捕获

开启 qapair，alice 提问后 bob 的 assistant 消息被存成 alice 的答案，alice 检索可读到 bob 的内容。

证据：`QA_interleaved_users_do_not_cross_capture`；入口：[sme/modules/bridge.py:306](<C:/Users/黄小乐/Desktop/记忆插件/sme/modules/bridge.py:306>)。

### B02 · P1 · 分层上下文画像越过 ns

开启 extraction/rules、profile、context；只把 alice 的检索结果传给 context.build，组装出的用户画像仍含 bob 的事实。

证据：`context_profile_namespace`；入口：[sme/modules/context.py:38](<C:/Users/黄小乐/Desktop/记忆插件/sme/modules/context.py:38>)。

### B03 · P1 · 删除记忆后问答答案仍回放

删除唯一问题记忆后，memories=0，但 QA pair 仍保留并检索出原答案。删除操作未清理相应 QA 状态。

证据：`QA_delete_question_removes_replay`；入口：[sme/memory_manager.py:182](<C:/Users/黄小乐/Desktop/记忆插件/sme/memory_manager.py:182>)。

### B04 · P1 · WAL 未覆盖全部成功变更

WAL 与 autosave 开启；完整 checkpoint 后进行 reinforce/link/consolidate/compress，再重启：强化回退、显式图边和摘要消失。JSON/SQLite × fsync/grouped 下强化和图边均复现；普通 CRUD 和带 link_to 的 add 可恢复。

证据：`reinforcement_survives_json_fsync`；入口：[sme/engine.py:513](<C:/Users/黄小乐/Desktop/记忆插件/sme/engine.py:513>)。

### B05 · P2 · QA 回放绕过标签与元数据过滤

不存在的 metadata_filters 与 tags 本应无命中，QA 的合成答案仍被追加到结果。

证据：`QA_honors_metadata_and_tag_filters`；入口：[sme/modules/qapair.py:142](<C:/Users/黄小乐/Desktop/记忆插件/sme/modules/qapair.py:142>)。

### B06 · P2 · 批量与官方 Client 的 ns 契约不齐

REST batch 每条的顶层 ns 被静默丢弃；默认 SDK 关闭 WAL 和写扩展时 add_many(ns=...) 不打标签；MemoryClient.add(ns=...) 抛 TypeError。metadata={"ns":...} 可绕行；WAL 开启时 SDK 批量 ns 用例通过。

证据：`REST_batch_namespace`；入口：[sme/api/server.py:612](<C:/Users/黄小乐/Desktop/记忆插件/sme/api/server.py:612>)。

### B07 · P2 · MCP 无法连接开启鉴权的服务

通过真正 stdio 协议验证：服务启用 api.auth_token 后，带正确 Bearer 的 HTTP 请求成功，MCP memory_stats 报错，服务日志为 401。向 MCP 进程传入同值 SME_API_AUTH_TOKEN 也未被当请求头使用。

证据：`MCP_authenticated_service`；入口：[sme/mcp_server.py:121](<C:/Users/黄小乐/Desktop/记忆插件/sme/mcp_server.py:121>)。

### B08 · P2 · 上下文总预算未执行

budget_tokens=128、6 条长检索记忆，context.build 输出按项目自身 estimate_tokens 估算为 4188 tokens；不能视为模型精确 token 数，但已远超声明预算。

证据：`context_total_token_budget`；入口：[sme/modules/context.py:74](<C:/Users/黄小乐/Desktop/记忆插件/sme/modules/context.py:74>)。

### B09 · P2 · 基础 add 的 WAL 恢复重置原始时间

单条 add 内容成功恢复，但 created_at 与 last_hit 变为恢复时间；旧式轻量日志没有记录这些字段，影响新鲜度与时序判断。

证据：`WAL_add_preserves_original_timestamps`；入口：[sme/storage/wal.py:494](<C:/Users/黄小乐/Desktop/记忆插件/sme/storage/wal.py:494>)。

### B10 · P2 · 部分非法输入导致 500 和部分写入

空文本、纯空格单写返回 500；batch 的第二条 metadata 非法时先落第一条，再返回 500。这里不推定 batch 承诺全量原子，只确认缺少完整预校验、错误分类与清晰的部分成功回执。

证据：`REST_batch_malformed_metadata`；入口：[sme/api/server.py:612](<C:/Users/黄小乐/Desktop/记忆插件/sme/api/server.py:612>)。

B01/B02/B03/B05 需要开启相应扩展模块才会触发；当前个人服务的配置没有因此被更改。这些结果不表示真实记忆库已经发生串读或删除失败。B04 的手动 `save()` 对照用例能完整保存批量记忆、强化和图边，故问题定位在自动持久化覆盖范围。B06 的单条 REST ns 与带 WAL 的 SDK 批量 ns 对照均通过。

## 检索效果

全新合成种子 101/202/303，每组 24 条目标事实与 72 条紧邻主题干扰，共 96 条文档、72 个可回答问题、8 个远域无答案问题。题型覆盖中文、改写、英文、错误码与否定表达；512 个唯一文本实际由本地 Qwen3-Embedding-0.6B 编码。不同检索参数共享相同固定向量，省去重复编码。

**这里测的是正确证据 ID 的召回，不是最终回答正确率，也不是开放世界、真人长程对话或竞品对战。** 同模板、多种子不等于充分现实覆盖。

| 参数 | Top1 正确证据 | Top5 正确证据 | 远域无答案仍返回 |
|---|---:|---:|---:|
| 默认 | 192/216（88.9%） | 216/216 | 24/24 |
| BM25 独立候选 20 | 192/216（88.9%） | 216/216 | 24/24 |
| 余弦门槛 0.44 | 192/216（88.9%） | 216/216 | 0/24 |

BM25 在这批题上没有增加额外命中，不据此判断其无用；本轮未修改任何生产开关。

| 默认题型 | Top1 | Top5 |
|---|---:|---:|
| 中文维护人 | 29/36 | 36/36 |
| 英文跨语言 | 35/36 | 36/36 |
| 精确错误码 | 36/36 | 36/36 |
| 中文改写 | 70/72 | 72/72 |
| 否定表达 | 22/36 | 36/36 |

### 无答案判断的补测

再对 3 个 seed 各取 4 个项目，询问从未记录的维护人私人手机号、服务器月租，共 24 个近域无答案问题。默认仍返回 24/24，门槛 0.44 仍返回 24/24。

这说明门槛能过滤部分主题无关内容，却不能证明“这条记忆包含所问细节”。返回相关项目记录本身是搜索接口的合理行为；将非空/高相似度当作答案存在的保证则不成立。不能用远域 0/24 误接收的结果宣称可靠拒答。

### 纠错的用户体验

真实 Qwen3 + rules 提取 + factversion，四个独立示例：

| 示例 | 期望新内容 | 新内容排第一 | 新内容在前五 |
|---|---|---|---|
| city | 上海 | 否 | 是 |
| pet | 豆包 | 是 | 是 |
| drink | 绿茶 | 否 | 是 |
| job | 晨光 | 否 | 是 |

北京→上海：旧北京事实已带 `superseded_by/invalid_at`，仍排在上海前面；饮料与工作示例还出现未匹配成纠错链的情况。这是四个具体反例，不把 1/4 写成产品整体纠错准确率。事实版本保留历史是既有设计，但检索第一位不能保证最新有效值。

### 本机时延

- 24 个不重复问题：p50 **514.9 ms**，p95 **589.5 ms**。
- 6 个已预热问题循环 8 次：p50 **2.59 ms**，p95 **3.84 ms**。
- 4 线程读取缓存问题 48 次：约 **0.128 秒**，均返回结果。

CPU 4 线程、96 条文档；时延环节没有同时运行本轮 pytest/压力脚本。重复查询受益于向量缓存；这些数据不能外推到大库、不同硬件或长文本。

## 可用与边界

- 普通单写、SDK/REST 批量内容、更新、归档/恢复、删除、带 link_to 的事务恢复均通过；配置热重建保留记忆，非空库换维度被阻止。
- 导出/导入对照保留了记忆字段，但 REST `/export` 中的 graph 不会被只接收 memories 的 `/import` 还原；不能把该接口对当成整个引擎的完整备份恢复。此项按接口范围限制记录。
- 未使用真实云端 LLM、真实 cross-encoder 或未缓存 ONNX 模型；无物理断电、Linux 或长达数天的持续运行实验。
- 本轮只体检，不修复上述问题。建议后续优先处理跨用户与删除残留，再补全非 CRUD 的 WAL，最后改进纠错优先级和接入体验。

## 最小复现思路

下列流程均在新建临时库中进行，勿指向真实 data/：

```python
# 先创建隔离的 SMEConfig，storage.path 指向 tempfile 目录。
# 1. qapair.enabled=True：
engine.add("我的项目口令是什么？", ns="alice")
engine.add("乙用户口令为 BETA-SECRET-439。", ns="bob", source="assistant")
engine.search("我的项目口令是什么？", ns="alice")  # 可返回 bob 的答案

# 2. WAL 和 autosave 开启：
m = engine.add("用户偏好简短回复。")
engine.save()
engine.reinforce(m.id)
engine.wal.close()
# 用同配置的新引擎 load：hit_count/weight 会回到 save 时的值。
```

完整测试脚本以文本形式保留在 JSON 的 `reproduction_harnesses` 中；固定语料、每题命中、配置和源文件哈希也在同一证据文件中。脚本没有被放入产品源码，临时可执行文件会清理。

原始证据：[healthcheck_2026-10-23.json](<C:/Users/黄小乐/Desktop/记忆插件/lab/results/healthcheck_2026-10-23.json>)。

![隔离服务的配置体检页面](<C:/Users/黄小乐/Desktop/记忆插件/lab/results/healthcheck_2026-10-23_ui.png>)
