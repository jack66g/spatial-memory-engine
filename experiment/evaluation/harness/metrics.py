# -*- coding: utf-8 -*-
"""检索质量指标:hit@k / Recall@k / MRR / nDCG@k + 延迟分位数。纯 Python,零依赖。"""
import math
import statistics


def hit_at_k(ranked_ids, gold, k):
    """top-k 里是否命中任一 gold"""
    return 1.0 if set(ranked_ids[:k]) & set(gold) else 0.0


def recall_at_k(ranked_ids, gold, k):
    g = set(gold)
    if not g:
        return 0.0
    return len(set(ranked_ids[:k]) & g) / len(g)


def mrr(ranked_ids, gold):
    g = set(gold)
    for i, rid in enumerate(ranked_ids, 1):
        if rid in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids, gold, k):
    """二值相关度的 nDCG"""
    g = set(gold)
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(ranked_ids[:k]) if rid in g)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(g), k)))
    return dcg / ideal if ideal > 0 else 0.0


def percentile(values, p):
    if not values:
        return 0.0
    sv = sorted(values)
    idx = min(int(p / 100 * len(sv)), len(sv) - 1)
    return sv[idx]


def summarize(per_query, latencies_ms):
    """per_query: list of dict(ranked=[...], gold=[...])"""
    n = len(per_query)
    out = {"n_queries": n}
    for k in (1, 3, 5, 10):
        out[f"hit@{k}"] = round(sum(hit_at_k(pq["ranked"], pq["gold"], k) for pq in per_query) / n, 4) if n else 0.0
        out[f"recall@{k}"] = round(sum(recall_at_k(pq["ranked"], pq["gold"], k) for pq in per_query) / n, 4) if n else 0.0
    out["mrr"] = round(sum(mrr(pq["ranked"], pq["gold"]) for pq in per_query) / n, 4) if n else 0.0
    out["ndcg@10"] = round(sum(ndcg_at_k(pq["ranked"], pq["gold"], 10) for pq in per_query) / n, 4) if n else 0.0
    # 综合分:排序质量为主(hit@3 + mrr + ndcg 加权),供寻优单目标
    out["composite"] = round(0.4 * out["hit@3"] + 0.35 * out["mrr"] + 0.25 * out["ndcg@10"], 4)
    if latencies_ms:
        out["lat_p50_ms"] = round(percentile(latencies_ms, 50), 2)
        out["lat_p95_ms"] = round(percentile(latencies_ms, 95), 2)
        out["lat_mean_ms"] = round(statistics.fmean(latencies_ms), 2)
    return out


def by_type(per_query, latencies_ms):
    """分查询类型统计(直查/时序/多gold),看权重对不同查询的差异化影响"""
    groups = {}
    for i, pq in enumerate(per_query):
        t = pq.get("type", "unknown")
        groups.setdefault(t, []).append(pq)
    return {t: {"n": len(v),
                "hit@3": round(sum(hit_at_k(p["ranked"], p["gold"], 3) for p in v) / len(v), 4),
                "mrr": round(sum(mrr(p["ranked"], p["gold"]) for p in v) / len(v), 4)}
            for t, v in groups.items()}
