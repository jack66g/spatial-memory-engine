"""SME 快速上手示例（iteration 3.4）。

    python examples/quickstart.py

三个场景：离线记忆 / 知识库导入 / 中文考问。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # SME 插件根（大实验/examples → 大实验 → 项目根）

from sme.engine import SpatialMemoryEngine
from sme.import_docs import import_documents


def main() -> None:
    # ---------- 1. 离线记忆（零配置，hashing embedding） ---------- #
    print("== 1. 离线记忆 ==")
    engine = SpatialMemoryEngine()
    engine.config.storage.autosave = False
    engine.add("用户喜欢喝咖啡")
    engine.add("用户周末打篮球")
    engine.add("用户在公司上班")
    hits = engine.search("用户喜欢什么饮品", top_k=3)
    for h in hits:
        print(f"  {h.score:.3f}  {h.memory.text}")

    # ---------- 2. 知识库导入（条款切分 + 摘要-原文两级） ---------- #
    print("== 2. 知识库导入 ==")
    law = (
        "第五百八十五条 当事人可以约定一方违约时应当根据违约情况向对方"
        "支付一定数额的违约金。约定的违约金低于造成的损失的，可以请求"
        "人民法院予以增加；过分高于损失的，可以请求适当减少。\n"
        "第七百零五条 租赁期限不得超过二十年。超过二十年的，超过部分无效。"
    )
    import_documents(engine, law, title="民法典示例", source="法律")
    hits = engine.search("违约金怎么约定", top_k=5)
    for h in hits:
        print(f"  {h.score:.3f}  {h.memory.text[:40]}...")
    # 命中摘要后可沿 children 取原文
    summary = next((h for h in hits if h.memory.source == "summary"), None)
    if summary:
        print(f"  摘要记忆 children（原文条款）: {len(summary.memory.children)} 条")
    else:
        print("  （示例条款较短，摘要未进入 top5；换更强 embedding 后可见两级结构）")

    # ---------- 3. 保存 / 加载 ---------- #
    engine.save("example_state.json.gz")
    engine2 = SpatialMemoryEngine()
    engine2.load("example_state.json.gz")
    print(f"== 3. 持久化 ==")
    print(f"  保存后加载: {len(engine2.memories)} 条记忆, "
          f"{engine2.region_stats().count} 个 Region")
    os.remove("example_state.json.gz")
    os.remove("example_state.embeddings.npz")


if __name__ == "__main__":
    main()
