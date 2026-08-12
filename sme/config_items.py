"""Config item registry: the single source of truth between the public
config file ``sme/config.json``, the interactive menu (``python -m sme.menu``)
and ``SMEConfig``.

Pure engineering layer - it contains NO engine logic. It only defines:

- which items a user may tune (the "tunable subset"),
- a Chinese description / validation rule per item,
- helpers to read, validate, complete and atomically write the JSON file,
- the 5 built-in presets (apply => write back to the config file).

The config file uses the exact same group/key layout as ``SMEConfig``
(plus one session-level ``memory`` group consumed by chat programs, and a
top-level ``_help`` mapping that is ignored by the engine).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

# Sentinel used by the chat program for "consolidation/compression disabled"
# (0 would cause a division by zero, see docs/USAGE.md 7.5).
OFF_PERIOD = 10 ** 9

HELP_KEY = "_help"  # in-file inline documentation, ignored by the engine

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)


@dataclass
class ConfigItem:
    path: str                  # dotted path into the JSON, e.g. "llm.model"
    name: str                  # short Chinese name
    group: str                 # config group displayed by the menu
    desc: str                  # what it does (shown when editing)
    kind: str = "str"          # str | int | float | bool | enum | interval
    choices: tuple = ()        # enum options
    minimum: float | None = None
    maximum: float | None = None
    required: bool = False     # empty input is rejected when True
    default: Any = ""


def _it(
    path: str,
    name: str,
    group: str,
    desc: str,
    kind: str = "str",
    choices: tuple = (),
    minimum: float | None = None,
    maximum: float | None = None,
    required: bool = False,
    default: Any = "",
) -> ConfigItem:
    return ConfigItem(path, name, group, desc, kind, choices, minimum, maximum,
                      required, default)


ITEMS: list[ConfigItem] = [
    # ---------------------------- LLM 接入 ---------------------------- #
    _it("llm.base_url", "接口地址", "LLM 接入",
        "OpenAI 兼容 /chat/completions 端点，如 https://api.deepseek.com/v1；留空 = 不使用 LLM（纯离线）"),
    _it("llm.api_key", "API 密钥", "LLM 接入",
        "服务商密钥；留空则读取环境变量 SME_LLM_API_KEY（建议用环境变量，不写进文件）"),
    _it("llm.model", "模型", "LLM 接入",
        "模型名，如 deepseek-v4-flash / gpt-4o-mini / qwen-plus",
        required=True, default=""),
    _it("llm.reasoning_effort", "推理级别", "LLM 接入",
        "推理型模型（如 deepseek-v4-flash）必须填 none，否则小 max_tokens 时返回空回复",
        kind="enum", choices=("", "none", "low", "medium", "high")),
    _it("llm.max_tokens", "最大 token", "LLM 接入",
        "单次回答最大 token 数，越长越贵", kind="int",
        minimum=1, maximum=128000, default=1024),
    _it("llm.temperature", "温度", "LLM 接入",
        "随机度 0~2；知识问答建议 0.3~0.6，越高越发散", kind="float",
        minimum=0, maximum=2, default=0.6),
    _it("llm.timeout", "超时（秒）", "LLM 接入",
        "请求超时秒数；本地模型可调大", kind="float",
        minimum=1, maximum=600, default=60),

    # --------------------------- Embedding ---------------------------- #
    _it("embedding.provider", "向量引擎", "Embedding 向量",
        "hashing=离线零依赖（默认）/ openai=任意 OpenAI 兼容 embedding 接口 / "
        "sentence-transformers=本地模型（需 pip install sentence-transformers）",
        kind="enum", choices=("hashing", "openai", "sentence-transformers"),
        default="hashing"),
    _it("embedding.model", "向量模型", "Embedding 向量",
        "本地如 BAAI/bge-small-zh-v1.5，接口如 BAAI/bge-m3",
        required=True, default="text-embedding-3-small"),
    _it("embedding.dim", "向量维度", "Embedding 向量",
        "必须与模型一致（bge-small-zh=512，bge-m3=1024，hashing 可自定）", kind="int",
        minimum=8, maximum=8192, default=64),
    _it("embedding.base_url", "接口地址", "Embedding 向量",
        "embedding 端点（provider=openai 时必填），如 https://api.siliconflow.cn/v1"),
    _it("embedding.api_key", "API 密钥", "Embedding 向量",
        "留空则读取环境变量 SME_EMBEDDING_API_KEY"),
    _it("embedding.batch_size", "批量大小", "Embedding 向量",
        "每次编码的条数；本地模型可调小省内存", kind="int",
        minimum=1, maximum=512, default=32),

    # ------------------------ 记忆会话（会话层） ---------------------- #
    _it("memory.top_k", "检索条数", "记忆会话",
        "每次检索返回的记忆条数，越大上下文越丰富但越占 token", kind="int",
        minimum=1, maximum=100, default=6),
    _it("memory.window_rounds", "对话窗口", "记忆会话",
        "上下文保留的最近对话轮数", kind="int",
        minimum=1, maximum=1000, default=20),
    _it("memory.consolidate_every", "融合周期", "记忆会话",
        "每 N 轮自动把相似记忆融合成摘要；填 off 关闭（知识库建议关闭）",
        kind="interval", minimum=1, default=8),
    _it("memory.compress_every", "压缩周期", "记忆会话",
        "每 N 轮生成长期摘要；填 off 关闭", kind="interval",
        minimum=1, default=16),
    _it("memory.graph_expand", "图增强跳数", "记忆会话",
        "检索时沿记忆图扩展关联记忆的跳数；0=关闭（纯向量检索）", kind="int",
        minimum=0, maximum=5, default=1),
    _it("memory.reinforce_on", "命中强化", "记忆会话",
        "检索命中后记忆权重增长（Ebbinghaus），常用记忆越来越靠前", kind="bool",
        default=True),
    _it("memory.persist_path", "状态文件", "记忆会话",
        "记忆状态文件路径（相对项目根），留空 = 不持久化"),

    # ---------------------------- 动力学开关 -------------------------- #
    _it("policy.decay_enabled", "时间衰减", "动力学开关",
        "旧记忆随时间降权；关 = 权重永不下降（纯知识库）", kind="bool", default=True),
    _it("policy.full_memory", "全量检索", "动力学开关",
        "关 = 只检索重要度达标的记忆（配合下方阈值）", kind="bool", default=True),
    _it("policy.importance_threshold", "重要度阈值", "动力学开关",
        "full_memory=关 时的检索过滤阈值", kind="float",
        minimum=0, maximum=1, default=0.25),
    _it("region.auto_evolve", "空间演化", "动力学开关",
        "Region 自动拆分/融合；关 = 空间结构稳定", kind="bool", default=True),
    _it("region.min_join_cosine", "聚合门槛", "动力学开关",
        "记忆并入 Region 的相似度门槛；调大分区更细", kind="float",
        minimum=0, maximum=1, default=0.7),
    _it("region.min_region_size", "最小 Region 成员", "动力学开关",
        "成员少于该值的 Region 会被并入邻居", kind="int",
        minimum=1, maximum=10000, default=3),
    _it("region.split_threshold", "拆分阈值", "动力学开关",
        "Region 成员数达到该值且密度达标时拆分", kind="int",
        minimum=1, maximum=100000, default=48),
    _it("region.merge_distance", "融合距离", "动力学开关",
        "邻近 Region 漂移融合距离阈值；0 = 永不融合", kind="float",
        minimum=0, maximum=2, default=0.12),
    _it("region.ann_enabled", "ANN 加速", "动力学开关",
        "Region 数量多时加速最近邻检索；Region 少时可关", kind="bool", default=True),
    _it("region.evolve_interval", "演化间隔", "动力学开关",
        "每 N 次写入触发一次演化检查", kind="int",
        minimum=1, maximum=100000, default=40),
    _it("reinforcement.ebbinghaus_half_life_days", "强化半衰期", "动力学开关",
        "Ebbinghaus 强化半衰期（天），越大强化越温和", kind="float",
        minimum=0.1, maximum=3650, default=7.0),
    _it("reinforcement.ebbinghaus_power", "强化曲线", "动力学开关",
        "强化曲线陡峭度", kind="float",
        minimum=0.1, maximum=10, default=1.2),
    _it("reinforcement.hit_weight_delta", "命中权重增量", "动力学开关",
        "每次命中权重增量；0 = 只计数不加权", kind="float",
        minimum=0, maximum=1, default=0.10),
    _it("decay.half_life_days", "衰减半衰期", "动力学开关",
        "衰减半衰期（天），越大衰减越慢", kind="float",
        minimum=0.1, maximum=3650, default=30.0),
    _it("decay.max_weight_decay", "最大衰减幅度", "动力学开关",
        "最大可衰减比例（1 = 可衰减到 0）", kind="float",
        minimum=0, maximum=1, default=0.85),
    _it("consolidation.similarity_threshold", "融合相似度", "动力学开关",
        "记忆融合的相似度门槛，越高越不容易合并", kind="float",
        minimum=0, maximum=1, default=0.5),
    _it("consolidation.max_group_size", "融合组上限", "动力学开关",
        "单组融合的成员数上限", kind="int", minimum=2, maximum=100, default=12),
    _it("consolidation.summary_source", "摘要方式", "动力学开关",
        "摘要生成方式；llm 需要配置 LLM", kind="enum",
        choices=("template", "llm"), default="template"),
    _it("compression.age_days_threshold", "压缩年龄门槛", "动力学开关",
        "多少天前的旧记忆参与长期压缩", kind="float",
        minimum=0, maximum=3650, default=3.0),
    _it("compression.min_region_compact", "压缩最小成员", "动力学开关",
        "Region 成员数达到该值才参与压缩", kind="int",
        minimum=2, maximum=100000, default=8),
    _it("compression.summary_source", "摘要方式", "动力学开关",
        "压缩摘要生成方式；llm 需要配置 LLM", kind="enum",
        choices=("template", "llm"), default="template"),

    # ---------------------------- 检索与排序 -------------------------- #
    _it("retrieval.top_regions", "Region 检索数", "检索与排序",
        "第一阶段检索的 Region 数量", kind="int",
        minimum=1, maximum=100, default=3),
    _it("retrieval.top_k", "候选上限", "检索与排序",
        "检索候选记忆上限", kind="int",
        minimum=1, maximum=1000, default=10),
    _it("retrieval.vector_weight", "向量通道权重", "检索与排序",
        "语义向量通道权重；三项之和不必为 1", kind="float",
        minimum=0, maximum=1, default=0.60),
    _it("retrieval.keyword_weight", "关键词通道权重", "检索与排序",
        "BM25 关键词通道权重；0 = 纯向量检索", kind="float",
        minimum=0, maximum=1, default=0.30),
    _it("retrieval.metadata_weight", "元数据通道权重", "检索与排序",
        "元数据匹配通道权重；0 = 关闭", kind="float",
        minimum=0, maximum=1, default=0.10),
    _it("retrieval.region_dampening", "区域惩罚分", "检索与排序",
        "全局补充记忆的区域惩罚分；0 = 无区域加分", kind="float",
        minimum=0, maximum=1, default=0.10),
    _it("retrieval.cjk_bigram", "中文二元分词", "检索与排序",
        "BM25 关键词通道对中文启用 1-2 元切分（违约金→违约/约金）；关 = 单字切分",
        kind="bool", default=True),
    _it("retrieval.candidate_window", "候选窗口", "检索与排序",
        "混合打分的最小候选池（实际取 max(top_k×2, 此值)）；模板句霸榜时调大可救回真实记忆",
        kind="int", minimum=4, maximum=1000, default=20),
    _it("ranking.semantic", "排序·语义", "检索与排序",
        "最终分排序权重（八项建议总和 = 1）；语义=检索相似度", kind="float",
        minimum=0, maximum=1, default=0.40),
    _it("ranking.importance", "排序·重要度", "检索与排序",
        "重要度越高越靠前（可被命中强化提升）", kind="float",
        minimum=0, maximum=1, default=0.12),
    _it("ranking.freshness", "排序·新鲜度", "检索与排序",
        "最近创建/命中的记忆更靠前", kind="float",
        minimum=0, maximum=1, default=0.10),
    _it("ranking.weight", "排序·权重", "检索与排序",
        "强化权重（命中增长）的压缩映射", kind="float",
        minimum=0, maximum=1, default=0.10),
    _it("ranking.decay", "排序·衰减", "检索与排序",
        "衰减因子（越旧越低）", kind="float",
        minimum=0, maximum=1, default=0.08),
    _it("ranking.hit_count", "排序·命中数", "检索与排序",
        "累计命中次数越多越靠前", kind="float",
        minimum=0, maximum=1, default=0.08),
    _it("ranking.recency", "排序·近期", "检索与排序",
        "最近命中时间越近越靠前", kind="float",
        minimum=0, maximum=1, default=0.06),
    _it("ranking.region", "排序·区域", "检索与排序",
        "Region 检索得分加成", kind="float",
        minimum=0, maximum=1, default=0.06),

    # ------------------------------ 存储 ------------------------------ #
    _it("storage.path", "快照路径", "存储",
        "引擎快照文件路径", required=True, default="data/engine.json"),
    _it("storage.autosave", "自动保存", "存储",
        "写入量满自动落盘，防丢记忆", kind="bool", default=True),
    _it("storage.autosave_interval", "自动保存间隔", "存储",
        "每 N 次写入自动保存一次", kind="int",
        minimum=1, maximum=100000, default=100),
    _it("storage.compress", "快照压缩", "存储",
        "快照 gzip 压缩（省空间）", kind="bool", default=True),
    _it("storage.backend", "存储后端", "存储",
        "json=默认 / sqlite=一体库（切换后建议重启生效）", kind="enum",
        choices=("json", "sqlite"), default="json"),

    # --------------------------- v2 模块开关 -------------------------- #
    _it("extraction.enabled", "事实提取", "v2 模块",
        "写入前做事实提取，只存干净事实（问题/闲聊/AI 回答默认不入库）", kind="bool", default=False),
    _it("extraction.mode", "提取方式", "v2 模块",
        "llm=大模型提取 / rules=离线规则提取（不调 LLM）", kind="enum",
        choices=("llm", "rules", "off"), default="llm"),
    _it("factversion.enabled", "纠错版本", "v2 模块",
        "重复事实去重、纠正句以最新说法为准（旧版降权不删除）", kind="bool", default=False),
    _it("noise.enabled", "噪音抑制", "v2 模块",
        "模板句/重复短句检索时降权，真实记忆不被霸榜", kind="bool", default=False),
    _it("noise.min_density", "噪音·信息密度阈值", "v2 模块",
        "低于该密度阈值（信息量）的句子受模板惩罚；调大可少误伤真实短记忆", kind="float",
        minimum=0, maximum=1, default=0.10),
    _it("noise.template_penalty", "噪音·模板惩罚", "v2 模块",
        "模板覆盖率惩罚系数；越低越保守（少误伤），越高压制越强", kind="float",
        minimum=0, maximum=1, default=0.3),
    _it("noise.dup_penalty", "噪音·重复惩罚", "v2 模块",
        "同句重复惩罚系数", kind="float",
        minimum=0, maximum=1, default=0.5),
    _it("qapair.enabled", "问答对", "v2 模块",
        "问→答固化为问答对，考问直接回放答案", kind="bool", default=False),
    _it("qapair.max_age_days", "问答对有效期", "v2 模块",
        "问答对多少天内有效", kind="float", minimum=1, maximum=3650, default=180.0),
    _it("factgraph.enabled", "知识图谱", "v2 模块",
        "实体-关系时序图 + 多跳检索（新关系覆盖旧关系）", kind="bool", default=False),
    _it("factgraph.extract_mode", "图谱提取方式", "v2 模块",
        "llm=大模型抽取 / rules=离线规则抽取", kind="enum",
        choices=("llm", "rules", "off"), default="llm"),
    _it("factgraph.max_depth", "图谱多跳深度", "v2 模块",
        "实体关系图检索的最大跳数", kind="int", minimum=1, maximum=10, default=3),
    _it("profile.enabled", "用户画像", "v2 模块",
        "聚合画像事实，\"总结一下我\"类查询加权", kind="bool", default=False),
    _it("profile.snapshot_every", "画像快照间隔", "v2 模块",
        "每写入多少条画像事实自动生成快照", kind="int",
        minimum=1, maximum=100000, default=50),
    _it("persistence.enabled", "WAL 增量落盘", "v2 模块",
        "增量写日志 + 周期快照，大库每轮保存提速（崩溃自动恢复）", kind="bool", default=False),
    _it("persistence.checkpoint_every", "WAL 快照间隔", "v2 模块",
        "每 N 次写入做一次全量快照", kind="int",
        minimum=1, maximum=100000, default=10),
    _it("api.host", "REST 监听地址", "v2 模块",
        "127.0.0.1=仅本机；0.0.0.0=局域网可访问", required=True, default="127.0.0.1"),
    _it("api.port", "REST 端口", "v2 模块",
        "监听端口", kind="int", minimum=1, maximum=65535, default=8000),
    _it("api.auth_token", "REST 鉴权令牌", "v2 模块",
        "非空即启用 Bearer 鉴权（访问 /docs 与 /health 之外的接口需带 "
        "Authorization: Bearer <token>；留空=无鉴权）"),
    _it("observability.enabled", "遥测统计", "v2 模块",
        "记录事件/延迟/命中率，可导出 JSON/CSV 报告", kind="bool", default=False),
    _it("context.enabled", "分层上下文", "v2 模块",
        "画像常驻 + token 预算内注入对话（省 token）", kind="bool", default=False),
    _it("context.budget_tokens", "上下文预算", "v2 模块",
        "分层上下文管理的 token 预算", kind="int",
        minimum=256, maximum=1000000, default=4096),
    _it("namespaces.enabled", "多用户隔离", "v2 模块",
        "A/B 用户/场景记忆互不可见（机器人多人共用）", kind="bool", default=False),
    _it("namespaces.default_ns", "默认命名空间", "v2 模块",
        "未指定时的命名空间名", required=True, default="default"),
    _it("rerank.enabled", "精排 Rerank", "v2 模块",
        "检索后用交叉编码器（bge-reranker）精排；默认关=零影响，开需安装 "
        "sentence-transformers", kind="bool", default=False),
    _it("rerank.model", "精排模型", "v2 模块",
        "交叉编码器模型名（sentence-transformers CrossEncoder 兼容）",
        default="BAAI/bge-reranker-base"),
    _it("rerank.top_n", "精排候选数", "v2 模块",
        "参与精排的候选命中数（超出部分保持原序）", kind="int",
        minimum=1, maximum=1000, default=50),
]

ITEM_BY_PATH: dict[str, ConfigItem] = {it.path: it for it in ITEMS}

_GROUP_ORDER: list[str] = []
for _itm in ITEMS:
    if _itm.group not in _GROUP_ORDER:
        _GROUP_ORDER.append(_itm.group)

GROUPS: list[tuple[str, list[ConfigItem]]] = [
    (g, [i for i in ITEMS if i.group == g]) for g in _GROUP_ORDER
]


# --------------------------------------------------------------------------- #
# 读取 / 取值 / 写值
# --------------------------------------------------------------------------- #
def _split(path: str) -> list[str]:
    return path.split(".")


def _walk(cfg: dict, item: ConfigItem) -> dict:
    """Return the section dict containing the item (creating nothing)."""
    cur: Any = cfg
    for part in _split(item.path)[:-1]:
        if not isinstance(cur, dict) or not isinstance(cur.get(part), dict):
            return {}
        cur = cur[part]
    return cur


def has_value(cfg: dict, item: ConfigItem) -> bool:
    return _walk(cfg, item).get(_split(item.path)[-1], "__missing__") != "__missing__"


def get_value(cfg: dict, item: ConfigItem) -> Any:
    section = _walk(cfg, item)
    return section.get(_split(item.path)[-1], item.default)


def set_value(cfg: dict, item: ConfigItem, value: Any) -> None:
    cur: Any = cfg
    parts = _split(item.path)
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def defaults_config() -> dict:
    cfg: dict = {}
    for item in ITEMS:
        set_value(cfg, item, item.default)
    return cfg


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
def parse_value(item: ConfigItem, raw: str) -> Any:
    """Parse and validate user input; raises ValueError with a Chinese message."""
    raw = (raw or "").strip()
    if item.required and raw == "":
        raise ValueError(f"{item.name} 不能为空，请输入一个值")
    if item.kind == "bool":
        flag = {
            "1": True, "0": False, "true": True, "false": False,
            "yes": True, "no": False, "on": True, "off": False,
            "y": True, "n": False, "开": True, "关": False, "是": True, "否": False,
        }.get(raw.lower())
        if flag is None:
            raise ValueError("请输入 开/关（或 1/0 / true/false / on/off）")
        return flag
    if item.kind == "enum":
        if raw == "":
            return ""
        for choice in item.choices:
            if choice.lower() == raw.lower():
                return choice
        raise ValueError(f"可选值：{' / '.join(c or '空' for c in item.choices)}")
    if item.kind == "interval":
        if raw.lower() in ("off", "none", "关", "关闭"):
            return OFF_PERIOD
        return _parse_number(item, raw, int)
    if item.kind == "int":
        return _parse_number(item, raw, int)
    if item.kind == "float":
        return _parse_number(item, raw, float)
    return raw


def _parse_number(item: ConfigItem, raw: str, cast) -> Any:
    try:
        value = cast(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{item.name} 必须是{'整数' if cast is int else '数字'}") from exc
    if item.minimum is not None and value < item.minimum:
        raise ValueError(f"{item.name} 不能小于 {item.minimum:g}")
    if item.maximum is not None and value > item.maximum:
        raise ValueError(f"{item.name} 不能大于 {item.maximum:g}")
    return value


def validate_value(item: ConfigItem, value: Any) -> str | None:
    """Validate an existing value (e.g. loaded from a file). Returns an error."""
    if value is None:
        return "值为空"
    if item.kind == "bool":
        return None if isinstance(value, bool) else "应为 开/关（true/false）"
    if item.kind in ("int", "interval"):
        if isinstance(value, bool) or not isinstance(value, int):
            return "应为整数"
        if item.minimum is not None and value < item.minimum:
            return f"不能小于 {item.minimum:g}"
        if item.maximum is not None and value > item.maximum:
            return f"不能大于 {item.maximum:g}"
        return None
    if item.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "应为数字"
        if item.minimum is not None and value < item.minimum:
            return f"不能小于 {item.minimum:g}"
        if item.maximum is not None and value > item.maximum:
            return f"不能大于 {item.maximum:g}"
        return None
    if item.kind == "enum":
        if value == "":
            return None
        if value not in item.choices:
            return f"可选值：{' / '.join(item.choices)}"
        return None
    return None if isinstance(value, str) else "应为字符串"


def validate_config(cfg: dict) -> list[str]:
    """Check existing values in a loaded config; returns warning lines."""
    warnings: list[str] = []
    for item in ITEMS:
        if not has_value(cfg, item):
            continue
        error = validate_value(item, get_value(cfg, item))
        if error:
            warnings.append(f"[{item.path}] = {get_value(cfg, item)!r}：{error}")
    return warnings


# --------------------------------------------------------------------------- #
# 展示
# --------------------------------------------------------------------------- #
def render_value(item: ConfigItem, value: Any) -> str:
    if value is None:
        return "（未设置）"
    if item.kind == "bool":
        return "开" if bool(value) else "关"
    if item.kind == "interval" and isinstance(value, (int, float)) and value >= OFF_PERIOD:
        return "关闭"
    if isinstance(value, str) and value == "":
        return "（空）"
    return str(value)


def type_hint(item: ConfigItem) -> str:
    if item.kind == "bool":
        return "开/关（1/0/true/false/on/off 均可）"
    if item.kind == "enum":
        return "可选：" + " / ".join(c or "空" for c in item.choices)
    if item.kind == "interval":
        return "整数，或 off（关闭）"
    if item.kind == "int":
        lo = f" ≥{item.minimum:g}" if item.minimum is not None else ""
        hi = f" ≤{item.maximum:g}" if item.maximum is not None else ""
        return f"整数（{lo}{hi}）"
    if item.kind == "float":
        lo = f" ≥{item.minimum:g}" if item.minimum is not None else ""
        hi = f" ≤{item.maximum:g}" if item.maximum is not None else ""
        return f"数字（{lo}{hi}）"
    return "字符串（回车留空 = 取消）"


# --------------------------------------------------------------------------- #
# 文件读写（原子写）
# --------------------------------------------------------------------------- #
def complete_config(cfg: dict) -> dict:
    """Ensure every registered item exists in the dict (defaults for missing)."""
    for item in ITEMS:
        if not has_value(cfg, item):
            set_value(cfg, item, item.default)
    return cfg


def build_help() -> dict:
    return {item.path: item.desc for item in ITEMS}


def load_config(path: str) -> dict:
    """Load a config file. Missing/corrupt files fall back to defaults."""
    if not os.path.exists(path):
        return defaults_config()
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            cfg = json.load(fh)
        if not isinstance(cfg, dict):
            return defaults_config()
    except (json.JSONDecodeError, OSError):
        return defaults_config()
    return cfg


def save_config(path: str, cfg: dict) -> None:
    """Write the config file atomically (tmp + rename), with inline _help."""
    cfg = complete_config(cfg)
    cfg[HELP_KEY] = build_help()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 预设场景（5 个）：应用后必须写回 config.json
# --------------------------------------------------------------------------- #
PRESETS: list[dict] = [
    {
        "key": "chat", "name": "01 聊天助手（默认）",
        "desc": "动态记忆：强化/衰减/融合/压缩全开（= 第一版默认行为）",
        "values": {
            "policy.decay_enabled": True,
            "region.auto_evolve": True,
            "memory.reinforce_on": True,
            "memory.consolidate_every": 8,
            "memory.compress_every": 16,
            "memory.graph_expand": 1,
            "retrieval.keyword_weight": 0.3,
            "extraction.enabled": False,
            "factversion.enabled": False,
            "noise.enabled": False,
            "qapair.enabled": False,
            "factgraph.enabled": False,
            "profile.enabled": False,
            "persistence.enabled": False,
            "observability.enabled": False,
            "context.enabled": False,
            "namespaces.enabled": False,
        },
    },
    {
        "key": "kb_dynamic", "name": "02 知识库·动态",
        "desc": "知识不衰减不融合不压缩，但越查越重要（命中强化）",
        "values": {
            "policy.decay_enabled": False,
            "region.auto_evolve": False,
            "memory.reinforce_on": True,
            "memory.consolidate_every": OFF_PERIOD,
            "memory.compress_every": OFF_PERIOD,
            "memory.graph_expand": 0,
            "retrieval.keyword_weight": 0.3,
            # 静态知识场景下 freshness/recency 无意义（迭代 2.4）：
            # 语义权重调高、动态权重调低
            "ranking.semantic": 0.45,
            "ranking.freshness": 0.04,
            "ranking.recency": 0.02,
            "extraction.enabled": True,
            "factversion.enabled": True,
            "noise.enabled": True,
            "qapair.enabled": True,
            "factgraph.enabled": False,
            "profile.enabled": False,
            "persistence.enabled": True,
            "observability.enabled": False,
            "context.enabled": False,
            "namespaces.enabled": False,
        },
    },
    {
        "key": "kb_static", "name": "03 知识库·静态",
        "desc": "纯只读知识库：检索不改变任何状态，结果可复现",
        "values": {
            "policy.decay_enabled": False,
            "region.auto_evolve": False,
            "memory.reinforce_on": False,
            "memory.consolidate_every": OFF_PERIOD,
            "memory.compress_every": OFF_PERIOD,
            "memory.graph_expand": 0,
            "retrieval.keyword_weight": 0.3,
            # 静态知识场景下 freshness/recency 无意义（迭代 2.4）：
            # 语义权重调高、动态权重调低
            "ranking.semantic": 0.45,
            "ranking.freshness": 0.04,
            "ranking.recency": 0.02,
            "extraction.enabled": True,
            "factversion.enabled": True,
            "noise.enabled": True,
            "qapair.enabled": True,
            "factgraph.enabled": False,
            "profile.enabled": False,
            "persistence.enabled": True,
            "observability.enabled": False,
            "context.enabled": False,
            "namespaces.enabled": False,
        },
    },
    {
        "key": "robot", "name": "04 具身机器人",
        "desc": "崩溃安全落盘 + 多用户隔离 + 低资源离线（规则提取）+ 记忆稳定",
        "values": {
            "policy.decay_enabled": False,
            "region.auto_evolve": False,
            "memory.reinforce_on": False,
            "memory.consolidate_every": OFF_PERIOD,
            "memory.compress_every": OFF_PERIOD,
            "memory.graph_expand": 1,
            "retrieval.keyword_weight": 0.3,
            "extraction.enabled": True,
            "extraction.mode": "rules",
            "factversion.enabled": False,
            "noise.enabled": True,
            "qapair.enabled": True,
            "factgraph.enabled": False,
            "profile.enabled": False,
            "persistence.enabled": True,
            "observability.enabled": True,
            "context.enabled": False,
            "namespaces.enabled": True,
        },
    },
    {
        "key": "minimal", "name": "05 全关（裸存取）",
        "desc": "所有动态机制关闭：纯存储/纯检索，当普通向量库用",
        "values": {
            "policy.decay_enabled": False,
            "region.auto_evolve": False,
            "memory.reinforce_on": False,
            "memory.consolidate_every": OFF_PERIOD,
            "memory.compress_every": OFF_PERIOD,
            "memory.graph_expand": 0,
            "retrieval.keyword_weight": 0.3,
            "extraction.enabled": False,
            "factversion.enabled": False,
            "noise.enabled": False,
            "qapair.enabled": False,
            "factgraph.enabled": False,
            "profile.enabled": False,
            "persistence.enabled": False,
            "observability.enabled": False,
            "context.enabled": False,
            "namespaces.enabled": False,
        },
    },
]

PRESET_BY_KEY: dict[str, dict] = {p["key"]: p for p in PRESETS}


def apply_preset(cfg: dict, preset: dict) -> list[tuple[ConfigItem, Any, Any]]:
    """Apply a preset to the in-memory config; returns (item, old, new) changes."""
    changes: list[tuple[ConfigItem, Any, Any]] = []
    for path, value in preset["values"].items():
        item = ITEM_BY_PATH[path]
        old = get_value(cfg, item)
        if old == value:
            continue
        set_value(cfg, item, value)
        changes.append((item, old, value))
    return changes


def current_preset_key(cfg: dict) -> str:
    for preset in PRESETS:
        if all(
            get_value(cfg, ITEM_BY_PATH[k]) == v
            for k, v in preset["values"].items()
        ):
            return preset["key"]
    return "custom"


def preset_name(key: str) -> str:
    preset = PRESET_BY_KEY.get(key)
    return preset["name"] if preset else "自定义"
