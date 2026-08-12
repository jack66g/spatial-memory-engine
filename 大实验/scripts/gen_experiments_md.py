# -*- coding: utf-8 -*-
"""生成 EXPERIMENTS.md：全部实验数据汇总（从结果 JSON 自动展开）。

头部「0. 实验原理与方法」为人工维护的实验设计说明，其余章节从
results/*.json 自动展开。重新生成前请保留本脚本（不要手工编辑
EXPERIMENTS.md 的数据表部分）。
"""
import json, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def read(p):
    return (ROOT / p).read_text(encoding="utf-8")


out = []
W = out.append

# =====================================================================
W("# SME 实验全量数据汇总（EXPERIMENTS.md）")
W("")
W(f"> 生成时间：{time.strftime('%Y-%m-%d')} · 全部实验在大实验/ 下完成，SME 插件本体零改动")
W("> 公平协议：同一 DeepSeek（deepseek-v4-flash）+ 同一本地 BGE（bge-small-zh-v1.5）"
  "+ 同一考问集 + 公开语料（民法典全文/维基医学，manifest 可追溯）")
W("")

# =====================================================================
# 实验原理与方法（人工维护章节；数据表章节由 results/*.json 自动展开）
# =====================================================================
W("## 0. 实验原理与方法")
W("")
W("### 0.1 实验目标与研究问题")
W("""以「一对一擂台赛」形式，把 SME 放在与市面主流记忆方案（mem0 / langmem /
裸 RAG / BM25）完全相同的输入、模型与评测口径下，回答三个问题：
1. **检索精度**：SME 是否达到第一梯队（对话记忆、静态知识库两个维度分别验证）；
2. **机制增益**：提取 / 纠错 / 问答对 / 图谱 / 噪音抑制等创新机制是否有真实收益（消融验证）；
3. **稳健性**：针对竞品弱点的攻击题下，SME 是否更不容易被带偏（攻击题验证）。""")
W("")
W("### 0.2 公平协议原理（对比实验为何可信）")
W("""四同原则——任何一条不满足则对比失去意义：
1. **同一对话流**：所有基线回放同一条对话脚本（`data/dialogue_*.json`），
   逐轮存储用户/助手消息，轮序与内容完全一致 → 保证「输入的记忆内容」公平；
2. **同一 embedding**：所有向量类基线使用同一本地模型
   `BAAI/bge-small-zh-v1.5`（512 维）。DeepSeek 无 embedding 接口，统一用
   本地 BGE，任何基线不得使用不同向量源 → 保证「向量质量」公平；
3. **同一 LLM**：对话生成、LLM 提取、端到端判分统一 `deepseek-v4-flash`
   （`reasoning_effort=none`，最便宜档）；API Key 只走环境变量
   `SME_LLM_API_KEY`，绝不写入任何文件 → 保证「模型能力」公平；
4. **同一考问集 + 统一指标**：同一批问句与期望答案；指标统一
   hit@k / sem@k / 端到端 acc / 被带偏 / 一致率 → 保证「评测口径」公平。""")
W("")
W("### 0.3 指标定义")
W("""| 指标 | 定义 | 说明 |
|---|---|---|
| hit@k | 检索 top-k 中存在**文本包含**匹配期望答案的记忆 | 对 LLM 提取的主语归一化（“我叫小林”→“用户叫小林”）天然偏低 |
| sem@k | 检索 top-k 中存在与期望答案 **BGE 余弦 ≥0.60** 的记忆 | 主指标，对归一化鲁棒 |
| 端到端 acc | 问句 → 检索 top-3 → LLM 回答 → 期望答案关键词判分 | 第二轮升级口径，允许多记忆协同 |
| 被带偏（led_astray） | 检索命中纠错前旧说法且回答错误 | 版本管理能力的直接体现 |
| 一致率 | 同题 3 次重问回答一致的比例 | 稳定性 |
| 消融场景 | 按任务 id 前缀分组：直接回忆(01)/长尾细节/纠错(05)/问答(02)/画像(04)/图谱(03)/噪音(06)/同义改写 | 模块收益归因口径 |""")
W("")
W("### 0.4 数据集构建原理")
W("""- **对话脚本**：固定事实剧本（40 条「小林」个人事实：姓名/年龄/工作/爱好/家庭/计划/
  健康等）+ 陈述 / 纠错 / 闲聊 / 提问 4 类语句模板 + 固定 seed（20260809/10/11）
  随机编排 → 300 轮/seed。刻意覆盖：重复陈述、纠错（旧说法→新说法）、闲聊噪音、
  问句——对应记忆系统的四类真实压力。
- **任务集**：从对话脚本自动生成 52 题（直接回忆 15 / 长尾细节 15 / 纠错最新说法 10 /
  问答 10 / 画像 2）；判分关键词从期望答案自动提取（数字 + 非停用词 token）。
- **知识库语料**：公开数据——民法典全文（维基文库，1381 条款）、维基医学
  （1875 句 / 17 主题）；来源 manifest 可追溯（`data/corpus_manifest.json`），
  按条款号/句号切分入库。
- **攻击题**：13 题独立场景（同义改写 6 / 双重纠错 1 / 跨话题关联 3 / 噪音霸榜 3）。""")
W("")
W("### 0.5 攻击题设计原理（针对竞品弱点）")
W("""| 攻击 | 针对弱点 | 设计原理 |
|---|---|---|
| 同义口语化改写（6 题） | 纯关键词系统（BM25 / RAG 文本匹配） | 同一事实换措辞提问，关键词不重叠，逼检索走语义通道 |
| 双重纠错（1 题） | 无版本管理系统 | 陈述旧说法 → 纠错 → 再次纠错（两次反转），考问最新说法；无版本系统会命中旧记忆 |
| 噪音霸榜（3 题） | 无噪音抑制 | 注入 50 条重复模板句，真实记忆被挤出 top-k |
| 跨话题关联（3 题） | 无图谱 / 关联检索 | 答案分散在多个记忆，单点向量命中不足以回答 |""")
W("")
W("### 0.6 消融设计原理（模块贡献归因）")
W("""- **方向 A（叠加）**：v1 全关基线 → 逐模块单独开启，看单项增益；
- **方向 B（剔除）**：知识库全开 → 逐模块关闭，看组合中的边际贡献；
- **交叉验证**：某模块在 A 中无增益、在 B 中剔除有负贡献 → 判定为「依赖组合」
  （实验结论：05/02 单独开无收益，组合 01 后各 +0.08）；
- 提取统一 rules 模式（免费、确定性强、内部自洽），检索统一本地 BGE，
  16 配置 × 8 场景 × 2 次复跑取均值。""")
W("")
W("### 0.7 知识库评测修正原理（为什么有修正版）")
W("""旧实现 SME 知识库导入走 `engine.add`（v2 提取管线），导致：LLM 提取空转 +
factversion 把相似条款误合并（条款号 1217/1260，43 条丢失）。
修正为 `import_memories` 原文直存后条款号恢复 1258/1260（`run_kb_fixed.py`）。
原理：提取 / 纠错 / 问答对是「对话记忆」机制，不应作用于权威原文；修正后
9 选手 kb acc 趋同（0.586-0.690）——知识库纯静态检索各系统能力接近，
SME 的领先维度在动态对话记忆。""")
W("")
W("### 0.8 局限与公平性说明")
W("""- mem0 / langmem 对话仅单 seed（LLM 提取回放慢），SME 为 3 seed 均值；
- 攻击题部分题量偏少（双重纠错 1 题、噪音 3 题）；
- 图谱模块受 rules 提取实体覆盖率限制（仅 3 个实体），价值需 LLM 提取验证；
- Letta 需 server 架构，实验性缺席；
- 判分关键词自动生成，个别开放题（画像聚合）已改为精确问法。""")
W("")

# =====================================================================
W("## 1. 一页摘要")
W("")
W("""- **SME 优势定位**：对话记忆场景（端到端问答 0.421，仅次于 mem0 0.477，
  高于裸 RAG 0.128 的 3.3 倍）；纠错/版本管理/噪音抑制在攻击题中实锤有效。
- **检索精度（端到端口径）**：对话第一梯队（与 mem0 差距 <6pt，且 mem0 单 seed
  vs SME 3 seed）；知识库场景 9 选手趋同（0.586-0.690），SME 与 RAG 持平。
- **创新机制**：空间 Region 密度演化（写入 O(N²)→近线性后 10k 写入 2472/s）、
  双通道检索（向量+中文 bigram BM25）、8 信号可解释排序、事实版本化
  （supersede 而非删除）、QA 对直接回放、噪音抑制、全部可关（全关=v1）。
- **关注度评估**：机制新颖度中上（空间记忆非主流路线），工程完整度高
  （89 项可配项/预设/REST/评测资产），可背书性强；但需公开对标数据与
  论文/博客放大影响（见第 10 节）。
""")

# =====================================================================
W("## 2. 实验资产清单")
W("")
W("| 资产 | 路径 | 规模 |")
W("|---|---|---|")
assets = [
    ("民法典全文（维基文库公开）", "data/law/civil_code.txt", "115,334 字 / 7 编"),
    ("法律条款语料", "data/law/law_chunks.json", "1381 条款"),
    ("医学语料（维基百科中文条目）", "data/medical/medical_chunks.json", "1875 句 / 17 主题"),
    ("语料来源 manifest", "data/corpus_manifest.json", "URL/日期可追溯"),
    ("对话脚本 seed 20260809", "data/dialogue_20260809.json", "300 轮（含纠错/提问/闲聊）"),
    ("对话脚本 seed 20260810", "data/dialogue_20260810.json", "300 轮"),
    ("对话脚本 seed 20260811", "data/dialogue_20260811.json", "300 轮"),
    ("对话任务集", "data/tasks_dialogue.json", "52 题（回忆15/长尾15/纠错10/QA10/画像2）"),
    ("攻击题集", "data/tasks_attack.json", "13 题（同义6/双重纠错1/关联3/噪音3）"),
    ("知识库考问集", "data/tasks_kb.json", "29 题（法律15/医疗14）"),
    ("第一轮检索级结果", "results/battle_300.json", "8 选手全表"),
    ("第二轮端到端结果", "results/battle2.json", "9 选手 × 3 seed 每题明细"),
    ("知识库修正版结果", "results/battle2_kb_fixed.json", "9 选手 + 条款完整性"),
    ("攻击题专项结果", "results/attack_report.json", "13 题 × 9 选手"),
    ("消融矩阵", "results/ablation_matrix.json", "16 配置 × 8 场景"),
]
for name, p, size in assets:
    W(f"| {name} | `{p}` | {size} |")
W("")

# =====================================================================
W("## 3. 第一轮擂台赛：检索级对比（8 选手，39 题中文考问，1 seed）")
W("")
W("指标：hit@1=文本包含匹配；sem@1/3/5=BGE 余弦≥0.60 语义命中（主指标）。")
W("")
b1 = load("results/battle_300.json")
W("| 基线 | hit@1 | sem@1 | sem@3 | sem@5 | 存储ms | 检索ms | 记忆条数 |")
W("|---|---|---|---|---|---|---|---|")
for r in sorted(b1["results"], key=lambda x: -x["sem@1"]):
    W(f"| {r['name']} | {r['hit@1']:.3f} | {r['sem@1']:.3f} | {r['sem@3']:.3f} "
      f"| {r['sem@5']:.3f} | {r['store_avg_ms']:.1f} | {r['search_avg_ms']:.1f} "
      f"| {r.get('memories', 0)} |")
W("")
W("""结论：检索级区分度不足（0.79-0.95 挤在一起）；LLM 提取类系统文本精确匹配
（hit@1）因主语归一化天然偏低。→ 第二轮改为端到端问答判分。""")
W("")

# =====================================================================
W("## 4. 第二轮擂台赛：端到端对话（9 选手 × 3 seed，65 题/seed）")
W("")
W("对话 acc 为 3 seed 均值（mem0/langmem 为单 seed 20260809）；被带偏=检索含纠错"
  "前旧说法且回答错误；一致率=同题 3 次重问一致率。")
W("")
b2 = load("results/battle2.json")
rows = []
for name, r in b2["results"].items():
    d_accs = [r[k]["accuracy"] for k in r if k.startswith("d_") and "accuracy" in r[k]]
    d_acc = sum(d_accs) / len(d_accs) if d_accs else float("nan")
    spread = (max(d_accs) - min(d_accs)) if len(d_accs) > 1 else 0.0
    astray = max((r[k].get("led_astray", 0) for k in r if k.startswith("d_")), default=0)
    cons = r.get("consistency", [])
    cons_avg = sum(c["consistency"] for c in cons) / len(cons) if cons else float("nan")
    n_mem = r.get("d_20260809", {}).get("details", [])
    rows.append((name, d_acc, spread, astray, cons_avg))
rows.sort(key=lambda x: -x[1])
W("| 排名 | 基线 | 对话acc | ±(3seed) | 被带偏 | 一致率 |")
W("|---|---|---|---|---|---|")
for i, (name, d_acc, spread, astray, cons) in enumerate(rows, 1):
    W(f"| {i} | {name} | {d_acc:.3f} | {spread:.3f} | {astray} | {cons:.3f} |")
W("")

# ---- 每题明细（全部） ----
W("### 4.1 每题明细（全部 585 行 = 9 选手 × 65 题）")
W("")
W("格式：`[seed] 题号 正确? 带偏? | 问题 | 回答 | 检索top3`")
W("")
for name in b2["results"]:
    W(f"#### {name}")
    W("")
    W("```")
    for k in sorted(b2["results"][name]):
        if not k.startswith("d_"):
            continue
        r = b2["results"][name][k]
        for det in r.get("details", []):
            mark = "OK " if det.get("correct") else "XX "
            astray = "带偏!" if det.get("led_astray") else "     "
            q = (det.get("question") or "?")[:34]
            a = (det.get("answer") or "")[:28]
            ret = " | ".join((t or "")[:24] for t in (det.get("retrieved") or [])[:3])
            W(f"[{k[2:]}] {mark}{astray} Q:{q} A:{a} R:{ret}")
    W("```")
    W("")

# =====================================================================
W("## 5. 知识库评测（修正版：SME 原文直存，9 选手同语料 3256 条，29 题）")
W("")
W("""**修正记录（2026-08-09）**：旧实现 SME 走 engine.add（v2 管线），导致
LLM 提取空转、factversion 相似条款误合并（条款号 1217/1260，43 条丢失）。
修正为 import_memories 原文直存后条款号 1258/1260，恢复 41 条。
知识库场景下提取/纠错/问答对是对话记忆机制，不应作用于权威原文。""")
W("")
kb = load("results/battle2_kb_fixed.json")
W("| 基线 | kb acc | 记忆条数 | 条款号数(1260) | 导入秒 |")
W("|---|---|---|---|---|")
for name, r in sorted(kb["results"].items(), key=lambda x: -x[1]["accuracy"]):
    W(f"| {name} | {r['accuracy']:.3f} | {r['memories']} | "
      f"{r.get('clause_ids', '-')} | {r['import_seconds']:.0f} |")
W("")
W("""修正前后对比（kb acc）：sme_kb_dynamic 0.724→0.655、sme_kb_static 0.586→0.655、
sme_chat 0.621→0.690、rag 0.655→0.690、mem0 0.724→0.655。修正后 9 选手
0.586-0.690 趋同——知识库纯检索场景各系统能力接近，差异主要在对话记忆场景。""")
W("")

# =====================================================================
W("## 6. 薄弱点攻击题（13 题 × 9 选手，独立场景）")
W("")
attack = load("results/attack_report.json")
groups = {"dialect": "同义口语化改写（打 BM25）",
          "double": "双重纠错（打无版本管理）",
          "rel": "跨话题关联（打无图谱）",
          "noise": "噪音霸榜（打无噪音抑制）"}
W("| 攻击 | 针对 | " + " | ".join(n for n in attack) + " |")
W("|---|---|" + "---|" * len(attack))
for key, label in groups.items():
    cells = []
    for name in attack:
        vals = [1 if d.get("correct") else 0 for d in attack[name].get("details", [])
                if d.get("id", "").startswith(f"attack_{key}")]
        cells.append(f"{sum(vals)}/{len(vals)}" if vals else "-")
    W(f"| {label} | {key} | " + " | ".join(cells) + " |")
W("")
W("### 6.1 攻击题逐题明细")
W("")
for name in attack:
    W(f"#### {name}")
    W("")
    W("```")
    for det in attack[name].get("details", []):
        mark = "OK " if det.get("correct") else "XX "
        astray = "带偏!" if det.get("led_astray") else "     "
        q = (det.get("question") or "?")[:30]
        a = (det.get("answer") or "")[:30]
        W(f"{mark}{astray} [{det.get('id','')}] Q:{q} A:{a}")
    W("```")
    W("")

# =====================================================================
W("## 7. SME 消融矩阵（16 配置 × 8 场景，2 次复跑均值，rules 提取）")
W("")
ab = load("results/ablation_matrix.json")
scene_cols = ["直接回忆(01)", "长尾细节", "纠错版本(05)", "问答回放(02)",
              "图谱关联(03)", "噪音抑制(06)", "画像聚合(04)", "同义改写"]
W("| 配置 | 总acc | " + " | ".join(scene_cols) + " | 记忆 |")
W("|---|---|" + "---|" * len(scene_cols) + "|---|")
for name, m in ab.items():
    g = m.get("groups", {})
    cells = " | ".join(f"{g.get(c, {}).get('acc', 0):.2f}" for c in scene_cols)
    W(f"| {name} | {m['accuracy']:.3f} | {cells} | {m['memories']} |")
W("")
W("### 7.1 模块贡献（方向 B：知识库全开 0.392 逐一剔除）")
W("")
full_b = ab["B0_知识库全开"]["accuracy"]
W("| 剔除模块 | 剔除后 acc | Δ贡献 |")
W("|---|---|---|")
for mod in ["01", "05", "02", "03", "04", "06", "07"]:
    key = f"B_−{mod}"
    if key in ab:
        delta = full_b - ab[key]["accuracy"]
        W(f"| −{mod} | {ab[key]['accuracy']:.3f} | {delta:+.3f} |")
W("")
W("""消融结论：
1. **01 事实提取贡献最大**（A 方向 +0.22 全矩阵最大；B 方向 −01 Δ+0.069）
2. **05 纠错 / 02 问答对依赖 01**：单独开无收益（A 方向 0.15-0.19 < 基线），
   组合后各贡献 +0.077 —— 模块是组合拳
3. **03 图谱在规则提取下贡献≈0**（Δ+0.007）：规则提取实体覆盖率仅 3 个，
   图谱价值需 LLM 提取验证（后续轮次）
4. 方向 A（叠加）与方向 B（剔除）交叉验证一致
5. 噪音抑制场景：攻击注入下无抑制系统全挂，抑制有效""")
W("")

# =====================================================================
W("## 8. 成本明细")
W("")
W("| 项目 | LLM 调用 | token | 估算 ¥ |")
W("|---|---|---|---|")
costs = [
    ("对话生成 ×3 seed（900 轮回复）", "899", "~244k", "~0.73"),
    ("第一轮检索评测（无 LLM）", "0", "0", "0"),
    ("消融（16 配置 × 65 题 × 2 复跑）", "~2100", "~350k", "~1.05"),
    ("第二轮对话评测（9 选手×65 题 + 复跑）", "~2000", "~330k", "~1.0"),
    ("知识库评测（29 题 × 9 选手）", "~260", "~45k", "~0.15"),
    ("攻击题专项（13 × 9）", "~120", "~20k", "~0.06"),
    ("LLM 提取（mem0/langmem/SME-kb 回放）", "~900", "~230k", "~0.7"),
]
for n, c, t, y in costs:
    W(f"| {n} | {c} | {t} | {y} |")
W("| **合计** | | | **≈ ¥3.7-4.2** |")
W("")

# =====================================================================
W("## 9. 选手配置清单")
W("")
W("""| 选手 | 记忆机制 | 配置要点 |
|---|---|---|
| SME·聊天助手 | v1 默认 | 衰减/演化/融合全开，graph_expand=1，无 v2 模块 |
| SME·知识库动态 | 提取+纠错+QA+噪音+强化 | extraction(llm)/factversion/qapair/noise 开，decay/evolve 关，命中强化 |
| SME·知识库静态 | 同上无强化 | 同上但 reinforce 关（纯只读） |
| SME·机器人 | 规则提取+WAL+命名空间 | extraction(rules)/persistence/namespaces 开 |
| SME·全关 | 裸空间存储 | 全模块关，decay/evolve 关 |
| 裸 RAG | BGE 向量库 | 用户消息逐条存原文，余弦 top-k，无任何记忆机制 |
| BM25 | 纯关键词 | 中文 1-2 元切分，无向量 |
| mem0 | LLM 提取+qdrant | deepseek 提取（中文指令对齐），qdrant 本地 |
| langmem | LLM 提取+BGE | LangChain memory manager 提取，统一 BGE 检索 |
| leta | — | server-client 架构需独立服务，实验性缺席 |
""")

# =====================================================================
W("## 10. SME 优势与创新分析")
W("")
W("""### 10.1 优势是什么（实测支撑）
1. **对话记忆端到端第一梯队**：0.421（3 seed 均值）仅次于 mem0 0.477（单 seed），
   是裸 RAG 0.128 的 3.3 倍；被带偏仅 1 次（RAG 10 次）。
2. **纠错/版本管理实锤**：双重纠错攻击题下，无版本系统全部答旧说法，
   SME 知识库全对；纠错场景消融 acc 0.00→0.60（开 01+05）。
3. **噪音抗性**：50 条重复模板句注入后 RAG 0/3 全挂，SME 噪音抑制有效。
4. **中文适配**：bigram 关键词 + 语义双通道，同义改写攻击 6/6（RAG/BM25 1/6）。
5. **性能**：写入 2472/s（10k，O(N²)→近线性修复后）、100k 加载 2.9s、
   检索 p50 <1ms（hashing）/ ~10ms（BGE）。
6. **可解释与可配置**：8 信号 breakdown、89 项可配项、5 预设、全部可关
   （全关=v1，快照兼容）。

### 10.2 检索精度是否第一梯队
- 对话场景：**是**（端到端口径 0.421，与 mem0 差 <6pt；检索级 sem@1 0.846）。
- 知识库场景：**中上非领先**（修正版 0.655，与 RAG 0.690 持平/略低，
  9 选手 0.586-0.690 趋同）——纯静态检索大家差距小，SME 优势在动态记忆。
- 诚实结论：第一梯队=「对话记忆」维度；「静态知识库检索」维度不是领先者，
  需要 rerank/图谱 LLM 提取等后续增强。

### 10.3 有没有创新
机制层面有，且可验证：
1. **空间 Region 密度演化**（主流是向量库 top-k 或图结构；Region 动态
   分裂/融合 + ANN 加速是差异化路线）——写入路径增量几何修复后才是真 O(n)。
2. **8 信号可解释排序**（semantic/importance/freshness/weight/decay/
   hit_count/recency/region，带 breakdown 输出）——同类系统少有。
3. **事实版本化 supersede**（纠错后旧说法降权不删除，保留历史）——
   与 mem0/Graphiti 的时效性设计同向，实现独立。
4. **QA 对直接回放**（问题→已存答案 0.99 置顶，带缓存与 ns 隔离）。
5. **噪音抑制三信号**（重复/模板度/信息密度）——攻击题实测有效。
6. **增量几何 + WAL + sqlite 增量通道** 等工程创新（性能实测）。

### 10.4 创新机制怎么样、厉害吗
- **厉害的点**：组合完整（提取→版本→回放→图谱→画像→噪音→精排全链路），
  且全部模块可独立开关、快照兼容、有攻击题验证——不是演示级。
- **存疑的点**：图谱模块在规则提取下贡献≈0（需 LLM 提取验证）；知识库
  静态检索不领先（需 rerank）；画像聚合题检索弱（BGE 短句局限）。

### 10.5 有人会关注吗（市场/社区评估）
- **会关注的场景**：中文对话陪伴/客服记忆、隐私本地记忆（离线零依赖）、
  可解释记忆（breakdown 审计）、教育演示（空间可视化）。
- **关注度瓶颈**：① 无公开对标论文/博客（P3 计划未执行）；② 中文社区
  记忆库讨论聚焦 mem0/Letta/Graphiti，SME 无名气；③ 需要发布「检索精度
  第一梯队 + 攻击题吊打」的可复现数据包（本目录即素材）。
- **建议**：基于本 EXPERIMENTS.md 发技术博客（对比表 + 攻击题 + 消融），
  开源 benchmarks/ 资产，参与 mem0 等生态对比帖——关注度可撬动。
""")

# =====================================================================
W("## 11. 复现命令")
W("")
W("```bash")
W("cd 大实验")
W("python scripts/fetch_corpus.py --all        # 公开语料（已有 data/ 可跳过）")
W("python scripts/gen_dialogue.py --rounds 300 --seed 20260810 --out data/dialogue_20260810.json")
W("python scripts/make_tasks.py && python scripts/make_kb_tasks.py && python scripts/attack_quiz.py")
W("python scripts/run_battle.py --dialogue data/dialogue_300.json --quiz data/quiz_300.json   # 第一轮检索级")
W("python scripts/ablation.py --out results/ablation_matrix.json")
W("python scripts/run_battle2.py --out results/battle2.json")
W("python scripts/run_attack.py")
W("python scripts/run_kb_fixed.py              # 知识库修正版")
W("python scripts/gen_experiments_md.py        # 重新生成本文档")
W("```")

(ROOT / "EXPERIMENTS.md").write_text("\n".join(out), encoding="utf-8")
print(f"✓ EXPERIMENTS.md 生成完毕（{len(out)} 行）")
