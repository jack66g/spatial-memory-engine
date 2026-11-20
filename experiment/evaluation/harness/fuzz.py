# -*- coding: utf-8 -*-
"""
T5a 输入模糊测试:恶意/边界输入打引擎公开 API,期待"不崩溃"(可拒绝、可兜底,不可崩)。
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sme import SpatialMemoryEngine, SMEConfig  # noqa: E402

ISSUES = []


def fresh():
    cfg = SMEConfig()
    cfg.storage.autosave = False
    cfg.maintenance.background = False
    cfg.storage.path = str(HERE.parent / "work" / "fuzz_state.json")
    return SpatialMemoryEngine(cfg)


NASTY_TEXTS = [
    ("empty", ""),
    ("whitespace", "   \n\t  "),
    ("null_byte", "abc\x00def"),
    ("emoji_only", "😀🎭🚀🧬"),
    ("mixed_emoji_cjk", "用户😀喜欢🍎跑步🤖"),
    ("rtl", "משתמש אוהב תפוחים 用户"),
    ("long_word", "A" * 100000),
    ("long_text", "这是超长文本。" * 20000),
    ("newline_storm", "行1\n" * 5000),
    ("control_chars", "".join(chr(i) for i in range(1, 32))),
    ("surrogate_pair", "𝕐𝕦𝕤𝕖𝕣 中文"),
    ("sql_inject", "'; DROP TABLE memories; --"),
    ("json_inject", '{"hack": "overflow"}'),
    ("path_inject", "../../etc/passwd"),
    ("f_string_inject", "{__class__.__init__.__globals__}"),
    ("regex_bomb", "(a+)+$1" * 20),
    ("deep_repeat", "重复" * 50000),
    ("lone_surrogate", "\ud800"),
    ("combining", "u" + "\u0301" * 500),
    ("huge_number_text", "9" * 9999),
]

NASTY_QUERIES = NASTY_TEXTS + [("query_10MB", "搜索词" * 1000000)]


def trial(fn, label, expect_crash=False):
    try:
        fn()
        return "ok"
    except Exception as ex:
        ISSUES.append({"op": label, "exc": f"{type(ex).__name__}: {ex}"})
        return f"CRASH: {type(ex).__name__}: {str(ex)[:150]}"


def main():
    e = fresh()
    t0 = time.time()
    print("=== add() 模糊 ===")
    for name, payload in NASTY_TEXTS:
        r = trial(lambda p=payload: e.add(p), f"add({name})")
        print(f"  [{name}] {r}")

    print("\n=== add() 极端 importance/metadata ===")
    for name, imp in [("neg", -5), ("big", 1e308), ("nan", float("nan")), ("inf", float("inf")), ("-inf", float("-inf"))]:
        r = trial(lambda i=imp: e.add(f"importance测试{name}", importance=i), f"add(importance={name})")
        print(f"  [{name}] {r}")
    trial(lambda: e.add("meta测试", metadata={"k" * 10000: "v" * 10000}), "add(huge_metadata)")
    trial(lambda: e.add("meta测试2", metadata={"nested": {"deep": ["list", {"a": 1}]}}), "add(nested_metadata)")
    trial(lambda: e.add("meta测试3", metadata={"non_serializable": object()}), "add(non_json_metadata)")
    print("  metadata 三连完成(崩溃会记录)")

    print("\n=== search() 模糊 ===")
    for name, payload in NASTY_QUERIES:
        r = trial(lambda p=payload: e.search(p, top_k=5), f"search({name})")
        print(f"  [{name}] {r}")
    for name, tk in [("zero", 0), ("neg", -1), ("huge", 10**9), ("float", 3.7)]:
        r = trial(lambda t=tk: e.search("测试", top_k=t), f"search(top_k={name})")
        print(f"  [{name}] {r}")
    trial(lambda: e.search("测试", metadata_filters={"不存在键": "值"}), "search(无命中filter)")
    trial(lambda: e.search("测试", metadata_filters={"idx": [1, 2, {"x": "y"}]}), "search(嵌套filter值)")
    trial(lambda: e.search("测试", tags=[]), "search(空tags)")

    print("\n=== update/delete/archive 模糊 ===")
    trial(lambda: e.update("不存在的id", text="x"), "update(ghost_id)")
    trial(lambda: e.delete("不存在的id"), "delete(ghost_id)")
    trial(lambda: e.archive("不存在的id"), "archive(ghost_id)")
    trial(lambda: e.restore("不存在的id"), "restore(ghost_id)")
    trial(lambda: e.reinforce("不存在的id"), "reinforce(ghost_id)")
    trial(lambda: e.link("a", "b"), "link(ghost_ids)")
    trial(lambda: e.get(""), "get(empty_id)")

    print("\n=== 状态健康复查 ===")
    try:
        n = len(e.memories)
        hits = e.search("用户", top_k=5)
        print(f"  库内存活 {n} 条,检索返回 {len(hits)} 条")
    except Exception as ex:
        ISSUES.append({"op": "post-fuzz health", "exc": f"{type(ex).__name__}: {ex}"})
        print(f"  [FAIL] fuzz 后库不可用: {ex!r}")

    print(f"\n===== fuzz 完成: 崩溃/异常 {len(ISSUES)} 个 ({time.time()-t0:.0f}s) =====")
    for i in ISSUES:
        print("  -", i["op"], "->", i["exc"][:120])
    (HERE.parent / "results" / "t5a_fuzz.json").write_text(
        json.dumps(ISSUES, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
