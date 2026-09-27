"""配置校验 + LLM/embedding 连通性检查（纯函数核心，供 REST API / Web 前端复用）。

从原终端配置菜单（已删除）的 ``_cmd_check`` 迁移而来：逻辑保持一致，
但不再做任何 print/exit，而是返回结构化结果。配置路径不在此处解析，
调用方先自行加载配置文件并传入 :class:`sme.config.SMEConfig` 对象。

返回结构::

    {
        "config_valid": bool,   # 配置本身是否可用（值校验 + 维度匹配 + 凭证齐全）
        "errors": [str, ...],   # 配置层面的全部错误明细
        "engine_build_ok": bool,
        "llm":       {"configured": bool, "ok": bool, "detail": str},
        "embedding": {"configured": bool, "ok": bool, "detail": str},
    }

语义约定：
- ``config_valid`` 只反映配置本身的问题（值校验失败、embedding 维度不匹配、
  embedding=openai 但 base_url/api_key 未填）；真实请求失败不改动它，
  只体现在 ``llm.ok`` / ``embedding.ok`` 上。
- LLM 未配置（``llm.base_url`` 为空）不算失败：纯离线可用，``llm.ok`` 为 True。
- ``embedding.configured`` 表示是否配置了真实提供方（openai / sentence-transformers）；
  内置离线兜底 hashing 记为未配置。
- ``engine_build_ok`` 为 False 时无法探测连通性，直接提前返回。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sme.config import SMEConfig


def _probe(configured: bool, ok: bool, detail: str) -> dict:
    return {"configured": configured, "ok": ok, "detail": detail}


def run_config_check(config: "SMEConfig", ping: bool = False) -> dict:
    """验证配置：值合法性 + 引擎构建 + 可选的真实 LLM/embedding 连通性。

    Args:
        config: 调用方已加载好的配置对象（本函数不读配置文件、不解析路径）。
        ping: 为 True 时对已配置的 LLM / openai embedding 发真实请求测连通。
    """
    result: dict = {
        "config_valid": True,
        "errors": [],
        "engine_build_ok": False,
        "llm": _probe(False, False, ""),
        "embedding": _probe(False, False, ""),
    }

    # 惰性导入：避免在模块加载期引入重依赖（与原 _cmd_check 一致）
    from sme.config_items import validate_config
    from sme.engine import SpatialMemoryEngine

    # -- 1) 值合法性（类型/枚举/范围） ------------------------------------- #
    # to_dict 里"未设置"的值可能是 None（如 llm.reasoning_effort=None），
    # 而注册表的合法空值是 ""/缺键（validate_config 会把 None 报为"值为空"）。
    # 校验前把 None 叶子视为未设置剔除，与原菜单基于配置文件字典的行为一致。
    probe: dict = {}
    for section, body in config.to_dict().items():
        if isinstance(body, dict):
            probe[section] = {k: v for k, v in body.items() if v is not None}
        else:
            probe[section] = body
    warnings = validate_config(probe)
    if warnings:
        result["config_valid"] = False
        result["errors"].extend(warnings)

    # -- 2) 引擎构建 ------------------------------------------------------- #
    try:
        engine = SpatialMemoryEngine(config)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"引擎构建失败：{exc}")
        result["llm"] = _probe(False, False, f"引擎构建失败，无法检测：{exc}")
        result["embedding"] = _probe(False, False, f"引擎构建失败，无法检测：{exc}")
        return result

    result["engine_build_ok"] = True
    emb = engine.embeddings
    llm = engine.llm

    # -- 3) embedding 维度校验 --------------------------------------------- #
    cfg_dim = engine.config.embedding.dim
    if emb.name != "hashing" and emb.dim and emb.dim != cfg_dim:
        result["config_valid"] = False
        result["errors"].append(
            f"embedding 维度不匹配：模型实际 {emb.dim} 维，"
            f"配置 embedding.dim={cfg_dim}（应改为 {emb.dim}，"
            "否则空间/ANN 索引维度错位）"
        )

    # -- 4) LLM ------------------------------------------------------------ #
    if llm.configured:
        key_state = "已填" if llm.config.api_key else "空（无鉴权头，多数服务会 401）"
        detail = f"LLM 已配置：{llm.base_url} model={llm.config.model} key={key_state}"
        ok = True
        if ping:
            try:
                out = llm.chat([{"role": "user", "content": "ping"}], max_tokens=8)
                detail += f"；连通：{out[:40]}"
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail += f"；请求失败：{exc}"
        result["llm"] = _probe(True, ok, detail)
    else:
        result["llm"] = _probe(False, True, "LLM 未配置（llm.base_url 为空 → 纯离线可用）")

    # -- 5) embedding ------------------------------------------------------ #
    if emb.name == "openai":
        if not emb.base_url or not emb.api_key:
            detail = f"embedding=openai 但 base_url/api_key 未填（将请求 {emb.base_url}）"
            result["config_valid"] = False
            result["errors"].append(detail)
            result["embedding"] = _probe(True, False, detail)
        elif ping:
            try:
                v = emb.embed(["连通性测试"])
                result["embedding"] = _probe(
                    True, True,
                    f"embedding=openai（{emb.base_url}）连通：dim={len(v[0])}",
                )
            except Exception as exc:  # noqa: BLE001
                result["embedding"] = _probe(True, False, f"embedding 请求失败：{exc}")
        else:
            result["embedding"] = _probe(
                True, True, f"embedding=openai（{emb.base_url}，ping 可测连通）"
            )
    else:
        result["embedding"] = _probe(
            emb.name != "hashing", True,
            f"embedding={emb.name}（{emb.model_name}，无需网络）",
        )

    return result
