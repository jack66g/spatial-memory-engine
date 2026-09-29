# SME 实验场（lab/）

> 独立实验工程：**只 import sme，绝不改 sme**。赛果数据与本 README 进 git；
> 语料/工作区/.env 不进。

## 战报

> 数据的判读与完整故事见 **[战报.md](战报.md)**（三幕剧：平地饱和 → 规模分层 → 长程 7.5pp 王炸；含诚实备注与复现命令）。

## 铁律（大实验 1.0 的教训）

1. 所有脚本一键可重跑，seed 固定，数字可复现
2. 对战 3 seed 起步，单 seed 数字不进结论
3. 中文题集为稳定指标；LLM 波动场景看结论模式不看单值
4. 统一协议：同一对话流 + 同一 embedding（本地 Qwen3）+ 同一 LLM（deepseek-flash）+ 同一考问集
5. 密钥只从 `.env` 读（已 gitignore）

## 目录

- `scripts/` 压测与对战脚本（生成资产 seed 固定）
- `baselines/` 各选手统一接口适配器（mem0/langmem/裸RAG/BM25/Zep/Graphiti）
- `assets/` 评测题集（脚本生成）
- `results/` 结果 JSON（进 git；run 工作区不进）
- `UNINSTALL.md` 竞品依赖清单（打完可卸载）

## 状态

- [x] 压测基线（2026-09-28）：hashing 4231写/s；Qwen3 CPU 4写/s·p50 115ms（并发 742ms 零失败）；WAL -59% 吞吐；soak 无泄漏（详见 results/）
- [x] 冒烟赛（2026-09-28，1 seed）：7 选手完赛 2 缺席，bm25 1.00 / sme_chat 0.95 / rag·minimal 0.90 /
      mem0 0.85 / kb_dynamic 0.75 / graphiti(kuzu) 0.55；全量明细 results/battle_smoke.json。
      缺席：langmem（全系需 Python 3.11+，本机 3.10）、zep（需 docker）。
      已知上游坑（适配器内已绕开/修复）：graphiti-0.30.2 kuzu 后端 FTS 索引缺失、group_id 显式传值炸、
      DeepSeek schema 回显；kuzu 0.11.3 Windows 绝对中文路径 Error 3（一律传相对路径）
- [ ] 正赛：3 seed 起步（单 seed 数字不进结论），扩题集，加被带偏专项
- [ ] 优化候选（压测产出）：WAL 组提交 / embedding ONNX 量化 / RRF 融合 / vector 基线 / 图边索引 + 互联网调研
