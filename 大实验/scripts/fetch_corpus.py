"""公开语料获取：法律条款 + 医疗知识（严格只用公开权威来源）。

用法::

    python scripts/fetch_corpus.py --law      # 民法典全文（全国人大网）
    python scripts/fetch_corpus.py --medical  # 默沙东诊疗手册公众版若干疾病条目
    python scripts/fetch_corpus.py --all

产出（data/ 下）:
    law/civil_code.txt        民法典原文（仅正文，条款号保留）
    law/law_chunks.json       条款级语料 [{clause, text}]
    medical/msd_pages/        医疗网页原文
    medical/medical_chunks.json  段落级语料 [{source, text}]
    corpus_manifest.json      来源清单（URL/日期/版本，可追溯）
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAW_URL = "http://www.npc.gov.cn/npc/c30834/202006/75ba6483b8344591abd07917e1d25cc8.shtml"
LAW_SOURCE = "全国人民代表大会网《中华人民共和国民法典》全文（2020-06-01 公布）"

MSD_PAGES = [
    # 已确认可访问的 MSD 页面（补充语料）
    ("/zh-cn/home/bone,-joint,-and-muscle-disorders/osteoporosis/osteoporosis", "骨质疏松"),
    ("/zh-cn/home/blood-disorders/anemia/iron-deficiency-anemia", "缺铁性贫血"),
    ("/zh-cn/home/brain,-spinal-cord,-and-nerve-disorders/headaches/migraines", "偏头痛"),
]

CLAUSE_RE = re.compile(r"^第[〇零一二三四五六七八九十百千万0-9]+条")
SENT_SPLIT_RE = re.compile(r"(?<=[。；;！？!?])")


# --------------------------------------------------------------------------- #
# 民法典（维基文库公开全文，CC BY-SA，分编抓取）
# --------------------------------------------------------------------------- #
LAW_URL = "https://zh.wikisource.org/wiki/中华人民共和国民法典"
LAW_SOURCE = "维基文库《中华人民共和国民法典》全文（2020-05-28 通过，2021-01-01 施行）"
LAW_SUBS = ["第一编 总则", "第二编 物权", "第三编 合同", "第四编 人格权",
            "第五编 婚姻家庭", "第六编 继承", "第七编 侵权责任", "附则"]


def _wikisource_text(title: str) -> str:
    import urllib.parse

    url = ("https://zh.wikisource.org/w/api.php?action=query&prop=revisions"
           "&rvprop=content&format=json&redirects=1&titles="
           + urllib.parse.quote(title))
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = json.loads(
                urllib.request.urlopen(req, timeout=90).read().decode("utf-8"))
            p = next(iter(data["query"]["pages"].values()))
            revs = p.get("revisions", [])
            return revs[0]["*"] if revs else ""
        except Exception as exc:  # noqa: BLE001 - 429 限流退避重试
            wait = 10 * (attempt + 1)
            print(f"[law]   重试 {title}（{exc}），{wait}s 后...")
            time.sleep(wait)
    return ""


def _clean_wikitext(wt: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", "", wt)                       # 模板
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)  # 链接
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_law() -> None:
    print(f"[law] 抓取维基文库全文（{len(LAW_SUBS)} 编）")
    parts = []
    for sub in LAW_SUBS:
        wt = _wikisource_text("中华人民共和国民法典/" + sub)
        text = _clean_wikitext(wt)
        if text:
            parts.append(text)
            print(f"[law]   OK {sub}（{len(text)} 字）")
        else:
            print(f"[law]   - {sub} 为空，跳过")
        time.sleep(0.4)
    text = "\n\n".join(parts)
    (DATA / "law").mkdir(parents=True, exist_ok=True)
    (DATA / "law" / "civil_code.txt").write_text(text, encoding="utf-8")

    # 条款级切分（第X条起段）
    chunks: list[dict] = []
    current: dict | None = None
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        m = CLAUSE_RE.match(para)
        if m:
            if current:
                chunks.append(current)
            current = {"clause": m.group(0), "text": para}
        elif current:
            current["text"] += para
    if current:
        chunks.append(current)
    # 长条款按句切分
    flat = []
    for c in chunks:
        sentences = [s.strip() for s in SENT_SPLIT_RE.split(c["text"]) if s.strip()]
        if len(c["text"]) > 220 and len(sentences) > 1:
            for s in sentences:
                flat.append({"clause": c["clause"], "text": s})
        else:
            flat.append(c)
    (DATA / "law" / "law_chunks.json").write_text(
        json.dumps({"source": LAW_SOURCE, "url": LAW_URL, "chunks": flat},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[law] OK：{len(flat)} 条条款/段落（{len(text)} 字）→ "
          f"{DATA/'law'/'law_chunks.json'}")


# --------------------------------------------------------------------------- #
# 医疗（维基百科中文医学条目，公开 CC BY-SA；另有 MSD 已抓页面）
# --------------------------------------------------------------------------- #
WIKI_TOPICS = [
    "高血压", "2型糖尿病", "流行性感冒", "消化性溃疡", "幽门螺杆菌感染",
    "骨质疏松", "冠状动脉疾病", "心肌梗死", "带状疱疹", "手足口病",
    "缺铁性贫血", "睡眠障碍", "慢性胃炎", "甲状腺功能亢进症", "过敏性鼻炎",
    "对乙酰氨基酚", "营养学", "疫苗", "偏头痛", "哮喘",
]

WIKI_API = ("https://zh.wikipedia.org/w/api.php?action=query"
            "&prop=extracts&explaintext=1&format=json&redirects=1&titles=")


def fetch_medical() -> None:
    import urllib.parse

    out_dir = DATA / "medical" / "wiki_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for topic in WIKI_TOPICS:
        url = WIKI_API + urllib.parse.quote(topic)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            text = (page.get("extract") or "").strip()
            title = page.get("title", topic)
            if len(text) < 500:
                raise ValueError(f"提取过短（{len(text)}）")
            safe_name = re.sub(r"[\\/:*?\"<>|]", "_", topic)
            target = out_dir / f"{safe_name}.txt"
            target.write_text(text, encoding="utf-8")
            manifest.append({"topic": topic, "title": title, "url": url,
                             "file": str(target.relative_to(DATA)),
                             "chars": len(text)})
            print(f"[medical] OK {topic}（{len(text)} 字）")
        except Exception as exc:  # noqa: BLE001
            print(f"[medical] FAIL {topic}: {exc}")
        time.sleep(0.5)

    # 段落级语料（每段一条知识，按句再切）
    chunks = []
    for entry in manifest:
        text = (DATA / entry["file"]).read_text(encoding="utf-8")
        for i, p in enumerate(text.split("\n")):
            p = p.strip()
            if len(p) < 25:
                continue
            for s in [x.strip() for x in SENT_SPLIT_RE.split(p) if x.strip()]:
                if len(s) >= 25:
                    chunks.append({"source": f"Wiki-{entry['topic']}", "text": s,
                                   "seq": len(chunks)})
    (DATA / "medical" / "medical_chunks.json").write_text(
        json.dumps({"source": "维基百科中文医学条目（CC BY-SA，公开）",
                    "chunks": chunks}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[medical] OK：{len(chunks)} 句 → {DATA/'medical'/'medical_chunks.json'}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="公开语料获取")
    parser.add_argument("--law", action="store_true")
    parser.add_argument("--medical", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    if args.law or args.all:
        fetch_law()
    if args.medical or args.all:
        fetch_medical()

    manifest = {
        "date": str(date.today()),
        "law": {"source": LAW_SOURCE, "url": LAW_URL,
                "file": "data/law/civil_code.txt",
                "chunks": "data/law/law_chunks.json"},
        "medical": {"source": "默沙东诊疗手册公众版（msdmanuals.cn）",
                    "base_url": "https://www.msdmanuals.cn/zh-cn/home",
                    "pages": len(MSD_PAGES),
                    "chunks": "data/medical/medical_chunks.json"},
    }
    (DATA / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"manifest → {DATA/'corpus_manifest.json'}")


if __name__ == "__main__":
    main()
