# -*- coding: utf-8 -*-
"""
DeepSeek LLM 链路复试:密钥经环境变量注入(不落任何文件)。
覆盖:基础 chat / chat_json / 事实提取模块(extraction.mode=llm) /
LLM 融合摘要(consolidation.summary_source=llm) / REST config check ping。
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KEY = os.environ.get("SME_LLM_API_KEY", "")
BASE = "https://api.deepseek.com/v1"
MODELS = [m for m in os.environ.get("SME_LLM_MODELS", "deepseek-flash,deepseek-chat").split(",") if m]

if not KEY:
    print("SKIP: SME_LLM_API_KEY 未设置")
    sys.exit(0)

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append({"name": name, "ok": ok, "detail": str(detail)[:300]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  | {str(detail)[:180]}" if detail else ""))


def llm_config(model):
    from sme.config import SMEConfig
    cfg = SMEConfig()
    cfg.llm.base_url = BASE
    cfg.llm.api_key = KEY
    cfg.llm.model = model
    cfg.llm.timeout = 60
    return cfg


def pick_model():
    """探测可用模型名:优先用户指定的 deepseek-flash"""
    from sme.llm.client import LLMClient
    for m in MODELS:
        try:
            client = LLMClient(llm_config(m).llm)
            text = client.chat([{"role": "user", "content": "回复两个字:在线"}], max_tokens=2000)
            if text:
                return m, str(text).strip()[:40]
        except Exception as ex:
            print(f"  模型 {m} 探测失败: {type(ex).__name__}: {str(ex)[:150]}")
    return None, None


def main():
    t0 = time.time()
    model, hello = pick_model()
    record("LLM 基础 chat(模型探测)", bool(model), f"model={model}, reply={hello!r}")
    if not model:
        print(json.dumps(RESULTS, ensure_ascii=False, indent=1))
        sys.exit(1)

    from sme.llm.client import LLMClient
    client = LLMClient(llm_config(model).llm)

    # 1) chat_json 结构化输出
    try:
        data = client.chat_json([{"role": "user",
                                  "content": '从这句话提取JSON: {"fruit": "苹果", "person": "小明"}。只返回该JSON。'}])
        record("LLM chat_json 结构化输出", isinstance(data, dict) and "fruit" in str(data), str(data)[:120])
    except Exception as ex:
        record("LLM chat_json 结构化输出", False, f"{type(ex).__name__}: {ex}")

    # 2) extraction 模块(mode=llm)端到端
    try:
        from sme import SpatialMemoryEngine
        cfg = llm_config(model)
        cfg.storage.autosave = False
        cfg.maintenance.background = False
        cfg.extraction.enabled = True
        cfg.extraction.mode = "llm"
        with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
            cfg.storage.path = str(Path(td) / "ext.json")
            e = SpatialMemoryEngine(cfg)
            n0 = len(e.memories)
            facts = e.extraction.extract("用户: 我叫黄小乐,最喜欢吃苹果,每周六去打篮球。\n助手: 记住了!")
            n1 = len(e.memories)
            record("extraction(llm) 抽取事实入库", n1 > n0 or bool(facts),
                   f"facts={len(facts) if facts else 0}, 库新增 {n1 - n0} 条" +
                   (f", 首条: {facts[0].text[:50]}" if facts else ""))
    except Exception as ex:
        record("extraction(llm) 抽取事实入库", False, f"{type(ex).__name__}: {str(ex)[:200]}")

    # 3) LLM 融合摘要
    try:
        from sme import SpatialMemoryEngine
        cfg = llm_config(model)
        cfg.storage.autosave = False
        cfg.maintenance.background = False
        cfg.consolidation.summary_source = "llm"
        cfg.consolidation.similarity_threshold = 0.3
        with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
            cfg.storage.path = str(Path(td) / "cons.json")
            e = SpatialMemoryEngine(cfg)
            for i in range(6):
                e.add(f"用户喜欢苹果,每天吃{i}个")
            summaries = e.consolidate()
            ok = bool(summaries) and any(len(getattr(s, "text", "")) > 5 for s in summaries)
            record("consolidation(llm) 摘要生成", ok,
                   f"{len(summaries)} 条摘要" + (f": {summaries[0].text[:60]}" if summaries else ""))
    except Exception as ex:
        record("consolidation(llm) 摘要生成", False, f"{type(ex).__name__}: {str(ex)[:200]}")

    # 4) REST /config/check 真实 ping
    try:
        from fastapi.testclient import TestClient
        from sme.api.server import create_app
        from sme import SpatialMemoryEngine
        cfg = llm_config(model)
        cfg.storage.autosave = False
        with tempfile.TemporaryDirectory(dir=str(HERE.parent / "work")) as td:
            cfg.storage.path = str(Path(td) / "api.json")
            e = SpatialMemoryEngine(cfg)
            client_api = TestClient(create_app(e, config_file=str(Path(td) / "c.json")))
            r = client_api.post("/config/check", json={"ping": True})
            body = r.json()
            llm_ping = body.get("llm", {}).get("ok", body.get("llm"))
            record("REST /config/check ping LLM", r.status_code == 200 and bool(llm_ping),
                   json.dumps(body.get("llm", body), ensure_ascii=False)[:150])
    except Exception as ex:
        record("REST /config/check ping LLM", False, f"{type(ex).__name__}: {str(ex)[:200]}")

    n_ok = sum(1 for r in RESULTS if r["ok"])
    print(f"\n===== DeepSeek LLM 链路: {n_ok}/{len(RESULTS)} 通过 (model={model}, {time.time()-t0:.0f}s) =====")
    (HERE.parent / "results" / "reexam_llm.json").write_text(
        json.dumps({"model": model, "results": RESULTS}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
