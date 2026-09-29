# SME 优化方向调研（2026-09-28，互联网调研 + 本机压测交叉）

> 调研范围：CPU embedding 提速 / WAL 组提交 / 2026 记忆系统格局 / 检索融合 / 中文 embedding 选型。
> 结论按投入产出排序；所有数字有出处。本文件是决策依据，改造前先在 lab A/B。

## 一页总览（按 ROI 排序）

| # | 方向 | 结论 | 预期收益 | 改造落点 |
|---|---|---|---|---|
| 1 | WAL 组提交 | 强烈建议，改动小 | 写吞吐 1454→~3536 条/s（2.4x），丢失窗口有界 10-20ms | wal.py append 路径 |
| 2 | Embedding ONNX int8 | 强烈建议，生态就绪 | 2-4x 提速（115-250ms → 60-125ms） | sentence_transformers_provider / factory |
| 3 | 融合改造（min-max/去死信号） | 建议做（正确性为主） | 排序稳定性，性能无损 | retriever.py 三处 |
| 4 | 换 CPU 友好 embedding | 可选 profile | 延迟 5-15x，中文 C-MTEB 掉 1-3 分 | 换模型需向量重建脚本 |
| 5 | 格局借鉴 3 点 | 吸收不重写 | bi-temporal / sleep-time / PPR | 中期迭代 |

## 1. CPU embedding 提速

- **Qdrant 官方已出 [Qwen3-Embedding-0.6B-onnx](https://huggingface.co/Qdrant/Qwen3-Embedding-0.6B-onnx)**（含 int8 量化版），fastembed 一行加载，pooling 已正确处理
- int8 CPU 实测口径 2-4x（BERT 级 65ms→38ms；flash-rerank ~2x）
- **关键坑**：Qwen3-Embedding 是 decoder 架构用 last-token pooling，naive ONNX 导出默认 mean pooling 会得到废向量（cosine 全挤 0.99+，网上"Qwen3 质量差"的结论多为此坑污染）。用 Qdrant 官方导出即规避
- sentence-transformers 原生支持：`export_dynamic_quantized_onnx_model(quantization_config="avx512_vnni")` + `backend="onnx"`
- MRL：Qwen3-0.6B 支持 32-1024 维输出，可截 512/768 维省存储并降低质心/矩阵成本
- **房间里的大象**：本机 RTX 4060 装 CUDA 版 torch 是 10-50x 零代码收益（用户已裁定不装，维持 CPU 路线）

## 2. WAL 组提交

- 本机 -59% 的成本 ≈ 每 op 一次 fsync（0.5-5ms/次）——差值 0.43ms/条与此吻合
- SQLite 官方口径：`synchronous=NORMAL`（只在 checkpoint fsync）WAL 模式 3600 writes/s vs 每提交 fsync 291/s（12x）；掉电只丢 checkpoint 窗口、库永不损坏——SME 的 replay 已幂等（add guard），天然兼容
- RocksDB 范式：多 writer 合并一次 log write + 一次 sync；按字节攒批（wal_bytes_per_sync）
- **改造**：后台 flusher 线程，`group_window_ms=10-20ms` 或攒满 256 op 落一次盘；`sync_mode` 加 `grouped` 档保留 `fsync` 档给高重要度记忆；sqlite 分支改攒批事务 + `PRAGMA synchronous=NORMAL`
- 崩溃安全配套：WAL 行加单调 seq；Windows 的 os.fsync=FlushFileBuffers 无影响

## 3. 2026 记忆系统格局（借鉴 3 点 + 1 个警示）

- **Mem0**（2026-04 新算法：ADD-only 单遍抽取、agent 事实一等公民、向量+图混合、User Profiles JSON Schema 化）
- **Zep/Graphiti**：bi-temporal 时间线（事实带 valid_at/invalid_at，非有损更新）——SME 的 factversion 模块已有雏形，可演化
- **Letta**：sleep-time compute（空闲期离线重组织记忆）成行业共识（Anthropic "Dreaming"、Google Nested Learning 同向）——SME 的衰减/融合/压缩本就该挪到后台路径
- **HippoRAG 2**（ICML 2025）：KG + Personalized PageRank 联想式多跳，F1 +7pt——SME graph_expand 从 BFS 改 PPR，同时解决 O(E) 问题
- **警示 [MemBench]**：记忆系统比塞全上下文贵 14-77 倍且召回差 ~30%——价值主张锚定"上下文塞不下/塞不起"场景，自建诚实基准

## 4. 检索融合

- RRF 稳健（免调参免尺度校准，Cormack 2009），但 SME 的 8 信号 Ranker 需要分数而非排名 → **推荐混合方案：RRF/min-max 出候选序 + 候选集内 min-max 归一化出分数**
- 三个 bug 修法：
  1. BM25 峰值归一化（候选集一变分数漂移）→ 候选集内 min-max 或 RRF
  2. vector `(cos+1)/2`（正交白拿 0.5×0.6=0.3）→ 候选集内 min-max；Qwen3 cosine 实际挤在 0.3-0.9，线性映射压扁区分度
  3. metadata 恒 1 死信号 → 删权重并入 Ranker 或改"字段匹配占比"
- ColBERT/late interaction 不上：个人库万级规模候选集小，优势在亿级语料才兑现；CPU 预算花在交叉编码器 rerank（已有 bge-reranker-base 配置）单位质量收益更高；bge-m3 自带 dense+sparse+colbert 三模态可选
- rerank 触发条件：top-k>10 才开，控 CPU 预算

## 5. 中文 embedding 选型（<1B、CPU 友好）

| 模型 | 参数/架构 | C-MTEB | CPU 速度 |
|---|---|---|---|
| Qwen3-Embedding-0.6B（现役） | 0.6B decoder, MRL, 32k ctx | 62.36 | 慢（115-250ms） |
| bge-m3 int8 | 568M encoder | ~63-64（中文略高） | ~30ms 短 query（4核 ARM 实测） |
| gte-multilingual-base (mGTE) | 305M encoder | ~61-62 | dense 比 bge-m3 快最多 14x |
| Conan-embedding-v2 | 大 | SOTA | 不适合 CPU 档 |

- 结论：**保留 Qwen3 + ONNX int8 为默认路线**；提供 speed profile（mGTE int8，延迟再降 5-10x 掉 ~1 分）与 balanced profile（bge-m3 int8，质量平/略升 + 白送 learned sparse 通道可 A/B 替代 BM25）
- 换模型需：一次性向量重建脚本 + config 记 `embedding.revision` 版本号

## 实验优先级（进 lab A/B，数据说话后再合入）

1. WAL `grouped` 档：压测吞吐对比 + 杀进程丢失窗口验证
2. ONNX int8 provider：延迟 A/B + 检索质量 A/B（同一题集 hit@k）
3. 融合 min-max + 去 metadata 死信号：对战题集 acc A/B（vs 现峰值归一化）
4. （中期）bi-temporal / PPR 扩展 / sleep-time 整理
