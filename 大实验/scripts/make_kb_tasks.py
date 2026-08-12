"""知识库考问集：基于公开语料（民法典/医学）的条款精确命中评测。

判分（scene=kb）：回答必须包含正确条款号或关键条文片段——
不允许多条记忆协同充数（知识库场景要求精确命中，与对话场景分开）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# 法律考问：问题 -> (期望条款号, 关键条文片段, 补充关键词)
LAW_QUIZ = [
    ("违约金怎么约定", "第五百八十五条", "违约金"),
    ("约定的违约金低于损失可以请求增加吗", "第五百八十五条", "增加"),
    ("损失赔偿额怎么计算", "第五百八十四条", "损失赔偿额"),
    ("不履行合同义务要承担什么责任", "第五百七十七条", "违约责任"),
    ("买卖合同的定义是什么", "第五百九十六条", "买卖合同"),
    ("标的物交付前毁损灭失风险谁承担", "第六百零四条", "风险"),
    ("租赁期限最长不得超过多少年", "第七百零五条", "二十年"),
    ("承揽合同的定义", "第七百七十条", "承揽合同"),
    ("人格权包括哪些权利", "第九百八十九条", "人格权"),
    ("夫妻之间有相互继承遗产的权利吗", "第一千零六十一条", "继承"),
    ("过错侵害他人民事权益要承担什么责任", "第一千一百六十五条", "侵权责任"),
    ("侵害人身权益造成财产损失怎么赔偿", "第一千一百八十二条", "赔偿"),
    ("定金数额有什么限制", "第五百八十六条", "定金"),
    ("订立保证合同的形式要求", "第六百八十一条", "保证"),
    ("高空抛物造成损害谁担责", "第一千二百五十四条", "高空抛物"),
]

# 医疗考问：问题 -> (来源主题, 语料内关键片段)
MEDICAL_QUIZ = [
    ("血压持续超过多少算高血压", "高血压", "130/80"),
    ("高血压患者服用药物可以控制血压吗", "高血压", "藥物"),
    ("2型糖尿病的旧称是什么", "2型糖尿病", "非胰岛素依赖"),
    ("糖尿病长期并发症有哪些", "2型糖尿病", "視網膜病變"),
    ("流感由什么病毒引起", "流行性感冒", "流感病毒"),
    ("流感常见症状有哪些", "流行性感冒", "高燒"),
    ("骨质疏松容易在哪些部位骨折", "骨质疏松", "脊椎"),
    ("心绞痛是什么引起的", "冠状动脉疾病", "胸痛"),
    ("手足口病的病程大约多久", "手足口病", "7–10天"),
    ("睡眠障碍大致分为几类", "睡眠障碍", "异态睡眠"),
    ("带状疱疹的发病机制", "带状疱疹", "水痘"),
    ("对乙酰氨基酚中毒的解毒剂", "对乙酰氨基酚", "乙酰半胱氨酸"),
    ("哮喘的常见症状", "哮喘", "喘息"),
    ("偏头痛的病因假说", "偏头痛", "血管"),
    ("心肌梗死发作时怎么办", "心肌梗死", "胸痛"),
]


def clause_from(quiz: tuple) -> str:
    return quiz[1]


def build_law_tasks(chunks: list[dict]) -> list[dict]:
    """把法律考问绑定到语料中的真实条款文本（答案=条款号+关键片段）。"""
    by_clause = {c["clause"]: c["text"] for c in chunks if c.get("clause")}
    tasks = []
    for q, clause, kw in LAW_QUIZ:
        text = by_clause.get(clause, "")
        if not text:
            print(f"  ! 未找到条款 {clause}，跳过：{q}")
            continue
        tasks.append({
            "id": f"law_{clause}", "scene": "kb",
            "question": q,
            "answer_keywords": [clause, kw],
            "expect_text": text[:120],
            "conflict_text": None, "top_k": 5,
        })
    return tasks


def build_medical_tasks(chunks: list[dict]) -> list[dict]:
    """医疗考问：期望片段需在语料中真实存在。"""
    texts = [c["text"] for c in chunks if c.get("source")]
    tasks = []
    for q, topic, frag in MEDICAL_QUIZ:
        # 找包含关键片段的真实句子（简繁均可）
        candidates = [t for t in texts if frag in t]
        if not candidates:
            print(f"  ! 未找到 {topic}/{frag}，跳过：{q}")
            continue
        expect = candidates[0]
        # 判分关键词：关键片段 + 主题词（简繁双写）
        kws = [frag, topic]
        if frag == "130/80":
            kws.append("140/90")
        tasks.append({
            "id": f"med_{topic}_{frag[:4]}", "scene": "kb",
            "question": q,
            "answer_keywords": kws,
            "expect_text": expect[:120],
            "conflict_text": None, "top_k": 5,
        })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="生成知识库考问集")
    parser.add_argument("--law-chunks", default="data/law/law_chunks.json")
    parser.add_argument("--medical-chunks", default="data/medical/medical_chunks.json")
    parser.add_argument("--out", default="data/tasks_kb.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    law = json.loads((root / args.law_chunks).read_text(encoding="utf-8"))["chunks"]
    med = json.loads((root / args.medical_chunks).read_text(encoding="utf-8"))["chunks"]
    tasks = build_law_tasks(law) + build_medical_tasks(med)
    out = root / args.out
    out.write_text(json.dumps({"name": "kb_tasks", "tasks": tasks},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ 知识库考问集 {len(tasks)} 题 → {out}")


if __name__ == "__main__":
    main()
