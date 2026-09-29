"""P0 迭代测试：qapair 向量缓存（1.1）/ BM25 CJK bigram（1.2）/ 文档导入（1.3）。"""

from __future__ import annotations

import time

import numpy as np


# ------------------------- 1.1 qapair vector cache -------------------------- #
def test_qapair_cache_used_on_put(fresh_engine, zh):
    e = fresh_engine
    e.config.qapair.enabled = True
    e.config.qapair.similarity_threshold = 0.80
    e.add(zh["q_name"])
    e.add(zh["a_name"], source="assistant")
    assert zh["q_name"] in e.qapair._vec_cache
    # a second lookup uses the cache (no per-pair re-embedding)
    pairs = e.qapair.lookup(zh["q_name"] + "？", e)
    assert pairs
    assert zh["q_name"] in e.qapair._vec_cache


def test_qapair_cache_refilled_after_load(fresh_engine, zh, new_engine, tmp_path):
    e = fresh_engine
    e.config.storage.path = str(tmp_path / "qc.json")
    e.config.qapair.enabled = True
    e.config.qapair.similarity_threshold = 0.80
    e.add(zh["q_name"])
    e.add(zh["a_name"], source="assistant")
    e.save(e.config.storage.path)

    e2 = new_engine()
    e2.config.storage.path = e.config.storage.path
    e2.config.qapair.enabled = True
    e2.config.qapair.similarity_threshold = 0.80
    e2.load(e.config.storage.path)
    assert e2.qapair._vec_cache == {}  # lazily refilled
    hits = e2.search(zh["q_name"] + "？", top_k=5)
    assert any("小明" in h.memory.text for h in hits)
    assert zh["q_name"] in e2.qapair._vec_cache


def test_qapair_cache_many_pairs_single_batch(fresh_engine, monkeypatch):
    e = fresh_engine
    e.config.qapair.enabled = True
    e.config.qapair.similarity_threshold = 0.80
    embed_calls = {"n": 0}
    orig_embed = e.embeddings.embed

    def spy(texts):
        embed_calls["n"] += 1
        return orig_embed(texts)

    monkeypatch.setattr(e.embeddings, "embed", spy)
    for i in range(50):
        e.add(f"问题编号是多少{i}")
    e.add("答案是编号", source="assistant")
    # put-time caching embeds each question individually (50 calls), but a
    # fresh store with 187 pairs must look up with ONE batched fill + matmul
    store = type(e.qapair)(e.qapair.config)
    import sme.models as m
    from sme.utils import now
    for i in range(187):
        store.pairs.append(m.QAPair(
            question=f"缓存问题{i}", answer_text=f"答案{i}", created_at=now()))
    store._vec_cache = {}
    embed_calls["n"] = 0
    t0 = time.perf_counter()
    store.lookup("缓存问题9", e)
    elapsed = (time.perf_counter() - t0) * 1000
    # exactly 2 embed calls: one for the query, one batched fill for all 187
    assert embed_calls["n"] == 2, f"expected 2 embeds, got {embed_calls['n']}"
    assert elapsed < 100, f"187-pair lookup took {elapsed:.1f}ms"


# ------------------------- 1.2 BM25 CJK bigram ------------------------------ #
def test_tokenize_bigram_cjk():
    from sme.utils import tokenize

    toks = tokenize("违约金怎么约定", cjk_bigram=True)
    assert "违" in toks and "违约" in toks and "约金" in toks
    # unigram mode keeps the historical behavior
    assert tokenize("违约金") == list("违约金")
    # latin is untouched by bigram mode
    assert tokenize("user likes tea", cjk_bigram=True) == ["user", "likes", "tea"]


def test_bm25_bigram_chinese_hit(fresh_engine):
    e = fresh_engine
    e.config.retrieval.cjk_bigram = True
    e.add("违约金是指按照当事人的约定或法律规定，一方违约时应向另一方支付的金钱")
    e.add("用户喜欢喝咖啡和打篮球")
    hits = e.search("违约金怎么约定", top_k=5)
    assert hits and "违约" in hits[0].memory.text


def test_bm25_bigram_toggle_matches_config(fresh_engine):
    e = fresh_engine
    e.add("违约金条款示例")
    e.config.retrieval.cjk_bigram = False
    # refresh_keywords re-tokenizes with the new setting
    e.retriever.refresh_keywords(e)
    assert e.retriever.bm25.cjk_bigram is False
    assert "违约" not in e.retriever.bm25._doc_tokens[next(iter(e.memories))]
    e.config.retrieval.cjk_bigram = True
    e.retriever.refresh_keywords(e)
    assert "违约" in e.retriever.bm25._doc_tokens[next(iter(e.memories))]


# ------------------------- 1.3 document import ------------------------------ #
LAW_TEXT = (
    "中华人民共和国民法典\n"
    "第五百七十七条 当事人一方不履行合同义务或者履行合同义务不符合约定的，"
    "应当承担继续履行、采取补救措施或者赔偿损失等违约责任。\n"
    "第五百八十四条 当事人一方不履行合同义务或者履行合同义务不符合约定，"
    "造成对方损失的，损失赔偿额应当相当于因违约所造成的损失。\n"
)


def test_split_document_clauses():
    from sme.import_docs import split_document

    chunks = split_document(LAW_TEXT)
    clauses = [c["clause"] for c in chunks if c["clause"]]
    assert "第五百七十七条" in clauses
    assert "第五百八十四条" in clauses
    assert all(c["text"] for c in chunks)


def test_split_document_clause_with_zero():
    """第七百零五条 - the 零 numeral must be recognized (regression)."""
    from sme.import_docs import split_document

    text = ("第七百零五条 租赁期限不得超过二十年。超过二十年的，超过部分无效。\n"
            "第一千零六十一条 夫妻有相互继承遗产的权利。")
    clauses = [c["clause"] for c in split_document(text)]
    assert "第七百零五条" in clauses
    assert "第一千零六十一条" in clauses


def test_import_documents_structure(fresh_engine):
    from sme.import_docs import import_documents

    e = fresh_engine
    created = import_documents(e, LAW_TEXT, title="民法典", source="法律")
    assert created
    summary = created[0]
    assert summary.metadata["文档名"] == "民法典"
    assert summary.metadata["来源"] == "法律"
    # chunks carry clause metadata and are children of the summary
    chunks = [m for m in created[1:] if m.metadata.get("条款号") == "第五百七十七条"]
    assert chunks
    for chunk in created[1:]:
        assert chunk.parent_id == summary.id
        assert chunk.id in summary.children


def test_import_documents_retrieval_and_expansion(fresh_engine):
    from sme.retrieval import SearchQuery

    e = fresh_engine
    e.import_documents(LAW_TEXT, title="民法典", source="法律")
    # a Chinese legal question hits the right clause chunk
    hits = e.search("违约金怎么约定", top_k=10)
    texts = [h.memory.text for h in hits]
    assert any("违约责任" in t for t in texts), texts[:3]
    # metadata filters scope the search to one document
    hits = e.search(SearchQuery(text="违约", metadata_filters={"文档名": "民法典"}))
    assert hits


def test_import_documents_cli(fresh_engine, tmp_path, capsys):
    import sme.import_docs as mod

    doc = str(tmp_path / "law.txt")
    with open(doc, "w", encoding="utf-8") as fh:
        fh.write(LAW_TEXT)
    out = str(tmp_path / "law_state.json")
    rc = mod.main(["--file", doc, "--title", "民法典", "--source", "法律",
                   "--save", out])
    assert rc == 0
    captured = capsys.readouterr()
    assert "导入" in captured.out
    import os

    assert os.path.exists(out) or os.path.exists(out + ".gz")
