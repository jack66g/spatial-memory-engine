"""Module: document import for knowledge-base scenarios (iteration 1.3).

Turns SME from a pure memory engine into a "memory + knowledge base"
engine: a law / medical / policy document is split into clause/sentence
chunks, each chunk becomes a memory with structured metadata (``来源`` /
``文档名`` / ``条款号``), and a summary memory per document is wired as
parent (parent/children + summary edges) so retrieval hits the short,
precise summary vector and the caller can follow the edges to the original
text (the SME-flavored RAG chunk -> original mapping).

Usage::

    from sme.import_docs import import_documents
    engine = SpatialMemoryEngine()
    import_documents(engine, "民法典.txt", title="民法典", source="法律")
    engine.save()

CLI::

    python -m sme.import_docs --file law.txt --title 民法典 --source 法律
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Optional

from sme.graph import KIND_SUMMARY

CLAUSE_RE = re.compile(r"第[〇零一二三四五六七八九十百千万0-9]+[条款章节目]")
SENT_SPLIT_RE = re.compile(r"(?<=[。；;！？!?])")

# metadata keys (retrievable via metadata_filters)
META_SOURCE = "来源"
META_TITLE = "文档名"
META_CLAUSE = "条款号"
META_KIND = "doc_kind"


def split_document(text: str, max_chunk: int = 200) -> list[dict]:
    """Split a document into (clause, text) chunks.

    Strategy: split on clause markers first (第X条/第X章/第X节/第X目),
    then split long bodies at sentence boundaries; each output chunk
    carries its clause number. Returns a list of dicts:
    ``{"clause": str, "text": str}``.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?=第[〇零一二三四五六七八九十百千万0-9]+[条款章节目])", text)
    chunks: list[dict] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = CLAUSE_RE.match(part)
        clause = m.group(0) if m else ""
        body = part[len(clause):].strip() if clause else part
        if not body:
            continue
        # split long bodies at sentence boundaries
        sentences = [s.strip() for s in SENT_SPLIT_RE.split(body) if s.strip()]
        if not sentences:
            sentences = [body]
        buf = ""
        for sent in sentences:
            if buf and len(buf) + len(sent) > max_chunk:
                chunks.append({"clause": clause, "text": buf})
                buf = ""
            buf = (buf + " " + sent).strip() if buf else sent
        if buf:
            chunks.append({"clause": clause, "text": buf})
    # no clause markers at all: return the sentence chunks
    if not chunks:
        for sent in [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]:
            chunks.append({"clause": "", "text": sent})
    return chunks


def import_documents(
    engine: Any,
    text_or_path: str,
    title: str = "",
    source: str = "document",
    summary_text: Optional[str] = None,
) -> list[Any]:
    """Import one document into the engine as chunks + a summary parent.

    Returns the created memories (summary first, then chunks). Every chunk
    gets ``metadata_filters``-friendly metadata; the summary memory links to
    its chunks via parent/children + summary edges, so a hit on the summary
    can be expanded to the original text via ``memory.children``.

    NOTE: a ``metadata_filters={"文档名": ...}`` query matches BOTH the
    summary memory and its chunks; use ``"doc_kind": "doc_chunk"`` to
    restrict a search to the original text only.
    """
    if os.path.exists(text_or_path):
        with open(text_or_path, "r", encoding="utf-8-sig") as fh:
            text = fh.read()
        if not title.strip():
            title = os.path.basename(text_or_path)
    else:
        text = text_or_path
    title = title.strip()
    if not title:
        title = source

    chunks = split_document(text)
    if not chunks:
        return []

    mm = engine.memory_manager
    # 1) summary / title memory (short vector, precise retrieval target)
    summary_text = summary_text or f"【{title}】{source}文档（{len(chunks)} 条）"
    summary = mm.add_memory(
        text=summary_text,
        metadata={META_KIND: "doc_summary", META_TITLE: title, META_SOURCE: source},
        tags=["summary", "doc"],
        importance=0.7,
        source="summary",
    )
    created: list[Any] = [summary]
    # 2) chunk memories wired as children of the summary
    for i, chunk in enumerate(chunks):
        mem = mm.add_memory(
            text=chunk["text"],
            metadata={
                META_KIND: "doc_chunk",
                META_TITLE: title,
                META_SOURCE: source,
                META_CLAUSE: chunk["clause"],
                "chunk_index": i,
            },
            tags=["doc"],
            importance=0.5,
            source=source,
        )
        created.append(mem)
        mm.set_parent(mem.id, summary.id)
        mm.graph.add_edge(summary.id, mem.id, KIND_SUMMARY, weight=1.0, note="covers")
    return created


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: ``python -m sme.import_docs``."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="SME 文档导入（知识库场景）")
    parser.add_argument("--file", required=True, help="文档文件路径（txt/md）")
    parser.add_argument("--title", default="", help="文档名（写入 metadata.文档名）")
    parser.add_argument("--source", default="document", help="来源（写入 metadata.来源）")
    parser.add_argument("--config", default="", help="配置文件路径（默认内置默认值）")
    parser.add_argument("--save", default="", help="导入后保存的快照路径（默认 config 的 storage.path）")
    args = parser.parse_args(argv)

    from sme.config import SMEConfig
    from sme.config_items import load_config
    from sme.engine import SpatialMemoryEngine

    if args.config:
        cfg = SMEConfig.from_dict(
            {k: v for k, v in load_config(args.config).items() if k != "_help"}
        )
    else:
        cfg = SMEConfig()
    cfg.storage.autosave = False
    engine = SpatialMemoryEngine(cfg)
    created = import_documents(engine, args.file, title=args.title, source=args.source)
    path = args.save or cfg.storage.path
    if path:
        engine.save(path)
    print(f"导入 {len(created) - 1} 条内容 + 1 条摘要 → {path or '（未保存）'}")
    if created:
        print(f"摘要: {created[0].text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
