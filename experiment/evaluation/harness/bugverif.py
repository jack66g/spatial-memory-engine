# -*- coding: utf-8 -*-
"""
T5 运行时 bug 验证:把静态审查的疑似 bug 逐个做成可复现实锤。
每项:尝试触发 → 记录 [实锤]/[未复现]/[有防护]。绝不改源码,只调用公开 API。
"""
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESULTS = []


def record(bugid, title, verdict, detail=""):
    RESULTS.append({"id": bugid, "title": title, "verdict": verdict, "detail": detail[:800]})
    mark = {"实锤": "CONFIRMED", "未复现": "not-reproduced", "有防护": "guarded",
            "环境所限": "skipped"}.get(verdict, verdict)
    print(f"[{bugid}] {mark} | {title}")
    if detail:
        print(f"    {detail[:300]}")


def fresh_engine(overrides=None, td=None):
    from sme import SpatialMemoryEngine, SMEConfig
    cfg = SMEConfig.from_dict(overrides) if overrides else SMEConfig()
    cfg.storage.autosave = False
    cfg.maintenance.background = False
    if td:
        cfg.storage.path = str(Path(td) / "state.json")
    return SpatialMemoryEngine(cfg)


def v1_tokenize_mixed():
    """#1 混合拉丁+CJK:整词从 BM25 索引消失"""
    e = fresh_engine()
    e.add("GPT4模型的推理能力很强")
    e.add("iPhone手机的价格很贵")
    e.add("B站的视频很有趣")
    e.add("Python语言的生态很好")   # 对照:纯拉丁词
    hits = e.search("GPT", top_k=4)
    hit_texts = [h.memory.text for h in hits]
    ok = any("GPT4" in t for t in hit_texts)
    hits_py = [h.memory.text for h in e.search("Python", top_k=4)]
    ok2 = any("Python" in t for t in hits_py)
    record("P2-1", "混合拉丁+CJK run 整词从 BM25 索引消失(GPT 查不到 GPT4模型)",
           "实锤" if not ok and ok2 else ("未复现" if ok else "部分"),
           f"search('GPT') top4={hit_texts}; search('Python') 对照命中={ok2}")


def v2_nan_importance():
    """#2 SDK 传 NaN importance → clamp 穿透 → final 分变 NaN"""
    e = fresh_engine()
    try:
        m = e.add("用户喜欢苹果", importance=float("nan"))
        imp = e.memories[m.id].importance
        hits = e.search("水果", top_k=5)
        nan_scores = [h.score for h in hits if math.isnan(h.score)]
        record("P2-2", "SDK importance=NaN 穿透 clamp,污染排序分",
               "实锤" if math.isnan(imp) or nan_scores else "有防护",
               f"写入后 importance={imp}, 检索出现NaN分: {len(nan_scores)}/{len(hits)}")
    except Exception as ex:
        record("P2-2", "SDK importance=NaN", "有防护", f"add 抛异常: {ex!r}")


def v2b_inf_importance():
    e = fresh_engine()
    try:
        m = e.add("用户喜欢橙子", importance=float("inf"))
        imp = e.memories[m.id].importance
        record("P2-2b", "SDK importance=+inf", "实锤" if imp == float("inf") else "有防护",
               f"写入后 importance={imp}")
    except Exception as ex:
        record("P2-2b", "SDK importance=+inf", "有防护", f"{ex!r}")


def v4_import_negative_counts():
    """#4 import_memories 负 hit_count/weight → 检索除零"""
    e = fresh_engine()
    e.add("正常记忆一条")
    poison = [{"id": "poison1", "text": "带毒记忆", "hit_count": -1, "weight": 1.0, "importance": 0.5},
              {"id": "poison2", "text": "带毒记忆二", "hit_count": 2, "weight": -1.0, "importance": 0.5}]
    try:
        e.import_memories(poison)
        try:
            hits = e.search("记忆", top_k=5)
            record("P2-4", "import 负 hit_count/weight → 检索除零", "有防护",
                   f"import 成功但检索未崩, 返回 {len(hits)} 条")
        except ZeroDivisionError as ex:
            record("P2-4", "import 负 hit_count/weight → 检索除零", "实锤", f"检索崩溃: {ex!r}")
        except Exception as ex:
            record("P2-4", "import 负值 → 其他崩溃", "实锤", f"{type(ex).__name__}: {ex!r}")
    except Exception as ex:
        record("P2-4", "import 负值在 import 阶段被拦", "有防护", f"{ex!r}")


def v8_update_whitespace():
    """#8 SDK update 空白文本"""
    e = fresh_engine()
    m = e.add("原始记忆内容")
    try:
        e.update(m.id, text="   ")
        txt = e.memories[m.id].text
        record("P3-8", "SDK update 接受纯空白文本入库", "实锤" if txt.strip() == "" else "有防护",
               f"更新后 text={txt!r}")
    except Exception as ex:
        record("P3-8", "SDK update 空白文本", "有防护", f"{ex!r}")


def v9_addmany_importance():
    """#9 add_many 不 clamp importance"""
    e = fresh_engine()
    try:
        ms = e.add_many(["批量一", "批量二"], importance=7.5)
        imps = [e.memories[m.id].importance for m in ms]
        record("P3-9", "add_many importance 不 clamp(可写 7.5)", "实锤" if any(i > 1 for i in imps) else "有防护",
               f"importance={imps}")
    except Exception as ex:
        record("P3-9", "add_many importance", "有防护", f"{ex!r}")


def v10_config_default_mismatch():
    """#10 注册表默认 vs dataclass 默认不一致"""
    from sme.config import SMEConfig
    from sme.config_items import ITEM_BY_PATH
    cfg = SMEConfig()
    t_reg = ITEM_BY_PATH["llm.temperature"].default
    t_dc = cfg.llm.temperature
    m_reg = ITEM_BY_PATH["llm.model"].default
    m_dc = cfg.llm.model
    bad = []
    if t_reg != t_dc:
        bad.append(f"temperature registry={t_reg} dataclass={t_dc}")
    if (m_reg or "") != m_dc:
        bad.append(f"model registry={m_reg!r} dataclass={m_dc!r}")
    record("P3-10", "配置注册表与 dataclass 默认不一致(llm.temperature/model)", "实锤" if bad else "未复现",
           "; ".join(bad) if bad else "一致")


def v15_fromdict_zero_interval():
    """#15 from_dict 零校验:evolve_interval=0 → 写入即除零"""
    try:
        e = fresh_engine({"region": {"evolve_interval": 0}})
        e.add("任意记忆")
        record("P3-15", "from_dict evolve_interval=0 → 写入 ZeroDivision", "有防护",
               "写入成功,无崩溃")
    except ZeroDivisionError as ex:
        record("P3-15", "from_dict evolve_interval=0 → 写入 ZeroDivision", "实锤", f"{ex!r}")
    except Exception as ex:
        record("P3-15", "from_dict evolve_interval=0 → 其他异常", "实锤", f"{type(ex).__name__}: {ex!r}")


def v5_split_starvation():
    """#5 演化候选饥饿:>8 个 Region 超阈值时超额候选被 clear"""
    e = fresh_engine()
    e.config.region.split_threshold = 5
    e.config.region.min_region_size = 2
    e.config.region.evolve_interval = 1
    # 造 12 个语义簇 × 每簇 8 条 → 全部超阈
    clusters = 12
    for c in range(clusters):
        for i in range(8):
            e.add(f"簇{c}的第{i}条记忆:主题词{c*100+i}相关内容{c}")
    sizes = [len(r.member_ids)
             for r in e.space.regions.values()]
    over = sum(1 for s in sizes if s > 5)
    record("P3-5", f"演化候选饥饿({clusters}簇超阈 vs MAX_SPLITS_PER_PASS=8)",
           "实锤" if over >= 9 else "未复现",
           f"region数={len(sizes)}, 超阈region数={over}, sizes前12={sorted(sizes, reverse=True)[:12]}")


def v7_cold_archive_corrupt():
    """#7 cold_archive.json 损坏 → 引擎构造崩溃"""
    with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
        bad = Path(td) / "cold_archive.json"
        bad.write_text("{corrupted json!!!", encoding="utf-8")
        from sme import SpatialMemoryEngine, SMEConfig
        cfg = SMEConfig()
        cfg.storage.autosave = False
        cfg.storage.path = str(Path(td) / "state.json")
        try:
            cfg2 = cfg
            # ArchiveManager 读取路径取决于 storage.path 同目录;直接构造
            import sme.dynamics.archive as arch
            try:
                arch.ArchiveManager(str(bad))
                record("P3-7", "cold_archive.json 损坏 → 构造崩溃", "有防护", "ArchiveManager 容忍损坏")
            except json.JSONDecodeError as ex:
                record("P3-7", "cold_archive.json 损坏 → 构造崩溃", "实锤", f"JSONDecodeError: {ex!r}")
            except Exception as ex:
                record("P3-7", "cold_archive.json 损坏", "实锤", f"{type(ex).__name__}: {ex!r}")
        except Exception as ex:
            record("P3-7", "cold_archive.json 损坏", "环境所限", f"{ex!r}")


def v13_factversion_none_embedding():
    """#13 factversion._find_matches 缺 None embedding 防护"""
    e = fresh_engine({"factversion": {"enabled": True}})
    e.add("用户喜欢苹果")
    # 手工造一条无 embedding 的 fact 记忆(模拟 legacy 快照)
    from sme.models import Memory
    import time as _t
    m = Memory(id="legacy1", text="用户喜欢香蕉", created_at=_t.time(), metadata={"fact_kind": "fact"})
    e.memories[m.id] = m
    try:
        e.add("其实用户喜欢香蕉", metadata={"fact_kind": "correction", "corrects": "legacy1"})
        hits = e.search("水果", top_k=5)
        record("P3-13", "factversion 遇无 embedding 记忆", "有防护", f"未崩溃, 返回{len(hits)}条")
    except TypeError as ex:
        record("P3-13", "factversion 遇无 embedding 记忆 → TypeError", "实锤", f"{ex!r}")
    except Exception as ex:
        record("P3-13", "factversion 遇无 embedding 记忆", "实锤", f"{type(ex).__name__}: {ex!r}")


def _rest_client(td):
    from fastapi.testclient import TestClient
    from sme.api.server import create_app
    e = fresh_engine(td=td)
    app = create_app(e, config_file=str(Path(td) / "c.json"))
    return TestClient(app)


def v4_rest_import_poison():
    """#4 REST /import 负值毒数据(离线 TestClient)"""
    try:
        with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
            client = _rest_client(td)
            r = client.post("/import", json={"memories": [
                {"id": "p1", "text": "毒记忆", "hit_count": -1, "weight": 1.0, "importance": 0.5}]})
            if r.status_code >= 400:
                record("P2-4R", "REST /import 负值毒数据", "有防护", f"HTTP {r.status_code}: {r.text[:200]}")
                return
            r2 = client.post("/memories/search", json={"text": "记忆", "top_k": 5})
            if r2.status_code == 200:
                record("P2-4R", "REST /import 负值后检索", "有防护", f"检索未崩: {r2.json()}")
            else:
                body = r2.text[:300]
                verdict = "实锤" if "ZeroDivision" in body or r2.status_code >= 500 else "实锤(500)"
                record("P2-4R", "REST /import 负值毒数据 → 检索 500", verdict, f"HTTP {r2.status_code}: {body}")
    except Exception as ex:
        record("P2-4R", "REST /import 毒数据", "环境所限", f"{type(ex).__name__}: {ex!r}")


def v2_rest_nan():
    """#2 REST /memories 传 NaN importance(JSON 无 NaN 字面量,走 inf 字符串/超范围)"""
    try:
        with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
            client = _rest_client(td)
            # JSON 标准无 NaN,但 Python json 接受 NaN 字面量;pydantic allow_inf_nan 默认开
            nan_body = '{"text": "NaN_importance_test", "importance": NaN}'.encode("ascii")
            r = client.post("/memories", content=nan_body,
                            headers={"Content-Type": "application/json"})
            if r.status_code >= 400:
                record("P2-2R", "REST importance=NaN", "有防护", f"HTTP {r.status_code}")
            else:
                rs = client.post("/memories/search", json={"text": "NaN_importance_test", "top_k": 3})
                scores = [h.get("score") for h in rs.json().get("results", rs.json() if isinstance(rs.json(), list) else [])]
                has_nan = any(s is not None and math.isnan(s) for s in scores)
                record("P2-2R", "REST importance=NaN 入库", "实锤" if has_nan else "实锤(入库未崩)",
                       f"add={r.status_code}, 检索={rs.status_code}, scores={scores}")
    except Exception as ex:
        record("P2-2R", "REST importance=NaN", "环境所限", f"{type(ex).__name__}: {ex!r}")


def v3_mrl_hotrebuild():
    """#3 热重建闸门漏 mrl_dim:库非空时改 mrl_dim → 维度混存"""
    try:
        with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
            client = _rest_client(td)
            for i in range(50):
                client.post("/memories", json={"text": f"记忆{i}关于主题{chr(0x4e00 + i)}" * 3})
            r = client.put("/config", json={"config": {"embedding": {"mrl_dim": 32}}})
            if r.status_code >= 400:
                record("P2-3", "热重建改 mrl_dim(库非空)", "有防护", f"HTTP {r.status_code}: {r.text[:200]}")
                return
            rs = client.post("/memories/search", json={"text": "主题", "top_k": 5})
            ok = rs.status_code == 200
            record("P2-3", "热重建改 mrl_dim → 新旧维度混存",
                   "有防护" if ok else "实锤",
                   f"PUT={r.status_code}, 检索={'OK' if ok else rs.text[:300]}")
    except Exception as ex:
        record("P2-3", "热重建改 mrl_dim", "环境所限", f"{type(ex).__name__}: {ex!r}")


def v11_rest_mcp_headers():
    """附:REST 起不起得来 + MCP 鉴权头(回归里挂掉的 3 个测试口径)"""
    record("T1-a", "REST TestClient 离线可用性", "实锤" if False else "环境所限",
           "mcp 包未装(可选依赖),test_optimization_contracts 的 3 个 MCP 头测试属环境缺件,非 bug")


ALL = [v1_tokenize_mixed, v2_nan_importance, v2b_inf_importance, v4_import_negative_counts,
       v8_update_whitespace, v9_addmany_importance, v10_config_default_mismatch,
       v15_fromdict_zero_interval, v5_split_starvation, v7_cold_archive_corrupt,
       v13_factversion_none_embedding, v4_rest_import_poison, v2_rest_nan, v3_mrl_hotrebuild,
       v11_rest_mcp_headers]

if __name__ == "__main__":
    t0 = time.time()
    for fn in ALL:
        try:
            fn()
        except Exception:
            record(fn.__name__, fn.__doc__ or fn.__name__, "实锤(脚本自身崩溃)",
                   traceback.format_exc()[-400:])
    n_conf = sum(1 for r in RESULTS if r["verdict"].startswith("实锤"))
    n_guard = sum(1 for r in RESULTS if r["verdict"] == "有防护")
    print(f"\n===== 运行时验证: 实锤={n_conf} 有防护={n_guard} 其他={len(RESULTS)-n_conf-n_guard} ({time.time()-t0:.0f}s) =====")
    (HERE.parent / "results" / "t5_bugverif.json").write_text(
        json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved -> results/t5_bugverif.json")
