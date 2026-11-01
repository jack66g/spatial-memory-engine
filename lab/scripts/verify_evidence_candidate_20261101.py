"""Evaluate an intentionally optional, local field-evidence checker.

This is an experimental candidate, NOT an engine confidence API. Its frozen
acceptance thresholds are those in docs/待优化计划_2026-10-23.md. Failed
thresholds mean rejection, never a change to search's default behavior.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser(__doc__)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
source = Path(__file__).resolve().parents[2]
historical = json.loads((source/"lab/results/healthcheck_2026-10-23.json").read_text(encoding="utf-8"))
quality = json.loads((args.out/"quality_current.json").read_text(encoding="utf-8"))


def check(question, memories):
    if re.search(r"手机|电话",question):
        pattern = r"(?:手机号|电话)(?:为|是|：|:)?\s*(1[3-9]\d{9})"
    elif re.search(r"月租|租金",question):
        pattern = r"(?:月租|租金)(?:为|是|：|:)?\s*(\d+(?:\.\d+)?)\s*元"
    elif re.search(r"谁.*维护|谁负责|Who maintains",question,re.I):
        pattern = r"由([^，。；\s]{1,12})维护"
    elif "配置文件" in question or "线上参数" in question:
        pattern = r"[\w\u4e00-\u9fff]+\.yaml"
    elif "重装" in question:
        pattern = r"不能直接重装服务"
    elif re.search(r"哪个位置|先做什么|目录",question):
        pattern = r"/srv/\d+/\d+/cache"
    else:
        return {"status":"uncertain"}
    subjects = re.findall(r"[\u4e00-\u9fff]{2}-\d{3}|H[ABC]_\d+|ERR_DSK_\d+",question)
    if not subjects:
        return {"status":"uncertain"}
    for memory in memories:
        text = memory["text"]
        if not all(s in text for s in subjects):
            continue
        if re.search(r"未登记|未记录|未知|不是|未确认",text):
            continue
        match = re.search(pattern,text)
        if match:
            return {"status":"supported","source":memory["id"],"snippet":match.group(0)}
    return {"status":"uncertain"}


docs = {case["seed"]:{d["id"]:d for d in case["docs"]} for case in historical["quality_fixture"]}
original = []
for row in quality["rows"]:
    evidence = [docs[row["seed"]][mid] for mid in row["ids"]]
    original.append({"seed":row["seed"],"question":row["text"],"positive":row["gold"] is not None,
                     "result":check(row["text"],evidence)})
near = []
for seed, memories in docs.items():
    # Same objects as the historical near-domain probe: detail never stored.
    for index in range(4):
        text = memories[f"f-{index}-owner"]["text"]
        subject = re.search(r"[\u4e00-\u9fff]{2}-\d{3}",text).group(0)
        for field in ["维护人的私人手机号", "服务器月租"]:
            q = f"{subject}的{field}是多少？"
            near.append({"question":q,"positive":False,"result":check(q,list(memories.values()))})

heldout = []
for split, seed in [("debug",177),("validation",288),("holdout",399)]:
    rows = []
    letter = {177:"A",288:"B",399:"C"}[seed]
    for index in range(12):
        subject = f"H{letter}_{index}"
        phone = f"138{seed:03d}{index:05d}"
        phone_doc = {"id":f"{subject}-phone","text":f"{subject}的负责人手机号为{phone}。"}
        rent_doc = {"id":f"{subject}-rent","text":f"{subject}的服务器月租为{1200+index}元。"}
        for field, evidence in [("负责人手机号",phone_doc),("服务器月租",rent_doc)]:
            question=f"{subject}的{field}是多少？"
            rows.append({"question":question,"positive":True,"result":check(question,[evidence])})
        # Missing field on a known object; topic relevance alone is not proof.
        q=f"{subject}的负责人手机号是多少？"
        rows.append({"question":q,"positive":False,"kind":"missing_field",
                     "result":check(q,[rent_doc])})
        # There is a real numeric value in a historical statement, but the
        # current value is explicitly no longer established by that statement.
        variant = index % 4
        if variant == 0:
            text=f"{subject}的历史资料：负责人手机号为{phone}，号码后来改过。"
            kind="expired_phone"
        elif variant == 1:
            text=f"另一项目的负责人手机号为{phone}。{subject}只留有项目名称。"
            kind="wrong_object"
        elif variant == 2:
            text=f"{subject}的负责人手机号不是{phone}。"
            kind="negation"
        else:
            text=f"{subject}的上期服务器月租为{1200+index}元，本期价格已经调整。"
            q=f"{subject}的服务器月租是多少？"
            kind="expired_rent"
        historical_doc={"id":f"{subject}-boundary", "text":text}
        rows.append({"question":q,"positive":False,"kind":kind,
                     "result":check(q,[historical_doc])})
    positives=[r for r in rows if r["positive"]]
    negatives=[r for r in rows if not r["positive"]]
    heldout.append({"split":split,"seed":seed,"positive_count":len(positives),"negative_count":len(negatives),
                    "supported_positive":sum(r["result"]["status"]=="supported" for r in positives),
                    "false_supported":sum(r["result"]["status"]=="supported" for r in negatives),"rows":rows})
positive=[r for r in original if r["positive"]]
negative=[r for r in original if not r["positive"]]+near
count=sum(r["result"]["status"]=="supported" for r in positive)
false=sum(r["result"]["status"]=="supported" for r in negative)
accepted=count>=195 and false==0 and all(g["supported_positive"]>=22 and g["false_supported"]==0 for g in heldout)
report={"candidate":"local field checker; isolated experimental code only", "accepted":accepted,
        "frozen_thresholds":{"original_positive_min":195,"original_negative_false_supported_max":0,
                             "each_holdout_positive_min":22,"each_holdout_negative_false_supported_max":0},
        "summary":{"original_positive_count":len(positive),"original_supported":count,
                   "original_negative_count":len(negative),"original_false_supported":false},
        "original":original,"near":near,"groups":heldout}
(args.out/"evidence_candidate.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"accepted":accepted,"summary":report["summary"],"groups":[{k:v for k,v in g.items() if k!="rows"} for g in heldout]}))
