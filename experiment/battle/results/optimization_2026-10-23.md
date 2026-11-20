# SME 优化与实验报告 · 2026-10-23

> 此处保留上一轮实验口径。2026-11-01 将这些未提交改动纳入冻结基线，继续补齐体检缺陷；当前状态见 [最新验收报告](../../docs/优化验收报告_2026-11-01.md)。

已完成恢复协议、V2 写入事务、写侧命名空间隔离、分代快照、精排位置与查询缓存优化。默认检索在本次 208 条查询上保持相同 Top5；239 项测试通过、4 项跳过。影响排序的试验功能保留为可选项，没有直接改成生产默认。

## 改动与验收

| 改动 | 验收证据 |
|---|---|
| WAL 先重放、再持久提交、最后清理 | SDK 连续两次重启、checkpoint 失败、重放编码失败和坏日志均验证；失败保留日志，API 不再空库降级 |
| 序号水位与完整事务 | checkpoint 已完成而清理中断后，旧事务不会覆盖新快照；多事实、原文、失效标记和模块状态共同恢复 |
| 写侧 ns 隔离 | 同事实跨 ns 分别保留；纠错不失效其他用户；全局空间与具名空间去重互不干扰 |
| 分代 JSON/NPZ 快照 | 发布中断保留旧文本与旧向量组合；向量缺失/代次不符报错；成功提交后清理自身旧代次 |
| 配置与向量重建 | 问答对、pending 问题、暂时关闭的模块状态继续保留；加载切换模型后写端 provider 同步更新 |
| 查询嵌入与缓存 | 推理移出引擎主锁，模型切换后重算；LRU 128 条只缓存向量，实时重新检索；弱引用不固定旧模型 |
| 精排顺序 | 真正提供 top_n 候选，精排后再取请求 top_k；可控精排模型能救回原先第 8 条答案 |
| 可选检索能力 | 全库 BM25 独立召回、查询指令、原始余弦拒答门槛均有专项测试 |

新配置共 101 项。新增配置默认值：query_prefix 为空，keyword_candidates=0，min_similarity=-1，query_cache_size=128。没有全局改 float64 为 float32，也没有直接替换现有融合算法。

## 测试

- 旧版已有套件：191 通过、4 跳过。
- 修复前新增边界探针：20 失败、1 通过。这里统计用例，不把参数化用例当成 20 个独立 bug。
- 最终完整套件：239 通过、4 跳过，pytest 耗时记录见原始 JSON。
- 4 个跳过项为未缓存 ONNX 模型的可选测试，没有通过联网下载绕过。
- 真实子进程在恢复提交前和清日志前分别 os._exit(73)：JSON/SQLite × fsync/grouped × 两个退出位置，共 8 个组合。
- 注入保存失败、编码失败、fsync 失败、坏日志与探测失败；验证原日志保留、错误可见、后续重启可恢复。
- 混合并发测试：两路写、两路检索、一路保存；120 条确认新增在恢复后全部保留，查询命名空间无串读。
- 旧 suite 中几处“重启”模拟补上原写入者 close；直接 replay 的测试按新契约保留日志，不能把未提交重放当成 checkpoint。
- Python AST 与 git diff --check 通过；没有 commit、push 或服务重启。

## 冻结协议与质量结果

本地 Qwen/Qwen3-Embedding-0.6B、CPU 4 线程。832 个文本的向量只计算一次，两版共享。没有使用 LLM 裁判或付费生成。

3 个固定合成 seed（11/29/47），每组 24 条目标事实、120 条干扰、48 个正例问题和 8 个未知问题；另使用已有 seed=2 资产的 40 个问题与规范化事实。合计 184 个正例、24 个未知问题。资产未重新生成，内容哈希见 JSON。

**指标是证据来源 ID 的命中，不是最终问答准确率；规范化事实实验也不是原擂台的完整对话流复跑。** 所以这里的 Hit@5=100% 不能用于宣称产品问答 100% 或领先竞品。

默认配置下，208 条问题的完整 Top5 ID 与旧版逐条一致，变化数为 0。

| 试验配置 | 题组 | 正例数 | Hit@1 | Hit@5 |
|---|---|---:|---:|---:|
| default | synthetic-11 | 48 | 89.58% | 100.00% |
| default | synthetic-29 | 48 | 85.42% | 100.00% |
| default | synthetic-47 | 48 | 87.50% | 100.00% |
| default | canonical-frozen-2 | 40 | 90.00% | 100.00% |
| bm25_20 | synthetic-11 | 48 | 89.58% | 100.00% |
| bm25_20 | synthetic-29 | 48 | 85.42% | 100.00% |
| bm25_20 | synthetic-47 | 48 | 87.50% | 100.00% |
| bm25_20 | canonical-frozen-2 | 40 | 90.00% | 100.00% |
| query_instruction | synthetic-11 | 48 | 85.42% | 100.00% |
| query_instruction | synthetic-29 | 48 | 87.50% | 100.00% |
| query_instruction | synthetic-47 | 48 | 89.58% | 100.00% |
| query_instruction | canonical-frozen-2 | 40 | 92.50% | 100.00% |
| bm25_and_instruction | synthetic-11 | 48 | 85.42% | 100.00% |
| bm25_and_instruction | synthetic-29 | 48 | 87.50% | 100.00% |
| bm25_and_instruction | synthetic-47 | 48 | 89.58% | 100.00% |
| bm25_and_instruction | canonical-frozen-2 | 40 | 92.50% | 100.00% |

查询指令在 seed=11 上下降 2 题，在其他题组有所改善；净收益很小，不能据此证明普遍改善，因此没有默认开启。BM25 独立召回在这批题上没有带来额外命中；它突破 Region 闸门的价值已在单独构造的错误码专项中验证，仍默认关闭以保持兼容。

拒答门槛只在 seed=11 校准：无指令的原始余弦门槛 0.44，在另外两个 seed 上拒绝 15/16 个未知问题，正例 Hit@5 保持 96/96；它仍有未知误接收。加指令另行校准得到 0.40，合成留出集拒绝 16/16。该规模不足以证明门槛可普遍使用，因此 min_similarity 仍为 -1，不把排序分当置信度。

## CPU 计时与存储成本

正式计时期间没有同时运行本轮 pytest 或其他本轮压测。普通桌面系统与磁盘状态仍会波动；以下是本机样本，没有跨硬件保证。

| 查询负载 | 旧版 | 新版 |
|---|---:|---:|
| 6 个已预热问题循环 8 次，48 次查询 p50 | 306.19 ms | 2.24 ms |
| 同上 p95 | 393.05 ms | 2.61 ms |
| 同上再次计算查询向量次数 | 48 | 0 |
| 24 个不重复问题 p50 | 412.82 ms | 418.25 ms |
| 同上 p95 | 480.73 ms | 489.78 ms |

重复问题 p50 约快 136.5 倍，主要原因是省去重复编码；不重复查询没有被证明加速。

另做 3 seed × 每组每版每模式 2,000 次写入，交替旧/新版顺序，计入 grouped 最后刷盘。表中为 3 次中位数：

| 模式 | 写入条/s（旧→新） | 吞吐变化 | 全量 checkpoint（旧→新） |
|---|---:|---:|---:|
| fsync | 971.7 → 980.2 | +0.88% | 87.85 → 114.59 ms |
| grouped | 1665.9 → 1631.1 | -2.09% | 87.00 → 112.84 ms |

增加序号后曾观察到 grouped 吞吐下降，随后采用无损紧凑 JSON；最终 WAL 体积约缩小 1.8%。最终写入没有稳定加速证据。**全量 checkpoint 确实更慢**，新增文件落盘屏障与提交协议有实际成本；可靠性收益不能表述成“所有指标都更快”。大库 V2 全模块写入吞吐没有在本轮充分评测，不作性能承诺。

## 边界与复现

- 验证了进程退出和软件 I/O 故障，没有实际物理断电、存储控制器缓存故障或 Linux 实测。
- grouped 保留原有异步语义，通常按窗口攒批；未刷盘写入不等于逐条 fsync 持久化，调度和 I/O 故障会延长窗口。
- 本轮以单一服务写入者为使用前提，不提供多个独立进程同时写同一库的事务隔离保证。
- 精排管线用可控模型验证；没有下载并评测真实 cross-encoder 的端到端准确率。
- 新版读取旧快照与旧 WAL；新事务 WAL 需要新版恢复器。SDK load 遇到待恢复日志时会提交 checkpoint，磁盘故障会明确报错。
- 未在真实记忆库上做测试，常驻服务未重启；源文件变更将在下次服务启动时加载。

复现：先取得基线提交 1f635e7 的独立源码目录，再运行（workdir/basetemp 使用新的临时目录）：

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp TEMP_TEST_DIRECTORY
python -B lab/scripts/optimization_experiment.py --baseline BASELINE_SOURCE --workdir TEMP_EXPERIMENT_DIRECTORY --out lab/results/optimization_2026-10-23.json
python -B lab/scripts/storage_optimization_experiment.py --baseline BASELINE_SOURCE --workdir TEMP_EXPERIMENT_DIRECTORY --out TEMP_STORAGE_RESULT.json
```

源码 SHA256：旧 `3cfe94f5201c37ba9e3e2af774f6d253a3f7c79e9baf25ac27f4bfbe6a2383d0`；新 `d1d92824bf1e553581649174642c65f3a7c7eacf0f3602c917c88249334b4ad3`。包含源码注释和文件内容，哈希已与当前目录核对。原始逐题结果、留出门槛评估、存储样本及测试摘要见 [JSON](optimization_2026-10-23.json)。

提交顺序参考 [SQLite WAL](https://www.sqlite.org/wal.html)；Windows 替换使用 [MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw) 的 REPLACE_EXISTING/WRITE_THROUGH，并在发布前 fsync 文件。这些调用不替代真实硬件断电验证。

清理已验证：本轮独立临时目录、源码基线副本、pytest 目录与链接、故障注入产物、批量向量和中间日志已删除；只保留可复用脚本、测试、正式 Markdown/JSON 报告。
