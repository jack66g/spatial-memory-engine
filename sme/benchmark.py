"""Benchmark suite for the Spatial Memory Engine.

Measures:
    - write throughput (memories/sec)
    - search latency (p50 / p95 / p99, ms)
    - region statistics (count, sizes, density)
    - hit rate    - how often the expected memory lands in the top-k
    - recall@k    - fraction of related memories retrieved
    - memory utilization (hot / cold / covered by regions)
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from sme.retrieval import SearchQuery

TOPICS = {
    "apple": [
        "user likes apple fruit",
        "user likes apple juice",
        "user likes apple pie",
        "user likes apple orchard",
        "user likes apple tree",
    ],
    "sports": [
        "user plays basketball sport",
        "user plays football sport",
        "user plays tennis sport",
        "user plays swimming sport",
        "user plays volleyball sport",
    ],
    "coding": [
        "user codes python language",
        "user codes rust language",
        "user codes go language",
        "user codes java language",
        "user codes c language",
    ],
    "travel": [
        "user plans travel to japan",
        "user plans travel to norway",
        "user plans travel to iceland",
        "user plans travel to italy",
        "user plans travel to spain",
    ],
    "cooking": [
        "user cooks spicy curry dish",
        "user cooks spicy pasta dish",
        "user cooks spicy soup dish",
        "user cooks spicy rice dish",
        "user cooks spicy bread dish",
    ],
}


_SUFFIXES = [
    "for breakfast",
    "with friends",
    "on weekends",
    "during summer",
    "at night",
    "in the morning",
    "when traveling",
    "after work",
    "on holidays",
    "usually",
]

@dataclass
class BenchmarkResult:
    write_count: int = 0
    write_seconds: float = 0.0
    write_throughput: float = 0.0
    search_count: int = 0
    search_p50_ms: float = 0.0
    search_p95_ms: float = 0.0
    search_p99_ms: float = 0.0
    hit_rate: float = 0.0
    recall_at_k: float = 0.0
    region_count: int = 0
    avg_region_size: float = 0.0
    max_region_size: int = 0
    avg_region_density: float = 0.0
    memory_utilization: float = 0.0
    archived_count: int = 0
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    def summary_lines(self) -> list[str]:
        return [
            f"wrote {self.write_count} memories in {self.write_seconds:.2f}s "
            f"({self.write_throughput:.0f}/s)",
            f"searched {self.search_count} queries, p50={self.search_p50_ms:.1f}ms "
            f"p95={self.search_p95_ms:.1f}ms p99={self.search_p99_ms:.1f}ms",
            f"hit rate={self.hit_rate:.3f} recall@{self.details.get('k', 10)}="
            f"{self.recall_at_k:.3f}",
            f"regions={self.region_count} avg_size={self.avg_region_size:.1f} "
            f"max={self.max_region_size} avg_density={self.avg_region_density:.2f}",
            f"memory utilization={self.memory_utilization:.2%} "
            f"(archived={self.archived_count})",
        ]


class BenchmarkRunner:
    def __init__(self, engine: object, seed: int = 42) -> None:
        self.engine = engine
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    @staticmethod
    def topic_texts(n: int, rng: random.Random | None, topics: Optional[dict] = None) -> list[str]:
        topics = topics or TOPICS
        names = list(topics)
        rng = rng if rng is not None else random.Random(0)
        texts: list[str] = []
        for _ in range(n):
            topic = rng.choice(names)
            variants = topics[topic]
            texts.append(
                rng.choice(variants)
                + " "
                + rng.choice(_SUFFIXES)
                + f" (note #{rng.randint(0, 9999)})"
            )
        return texts

    # ------------------------------------------------------------------ #
    def write_benchmark(self, n: int = 1000, batch: bool = True) -> BenchmarkResult:
        engine = self.engine
        texts = self.topic_texts(n, self.rng)
        start = time.perf_counter()
        if batch:
            engine.add_many(texts)
        else:
            for text in texts:
                engine.add(text)
        elapsed = time.perf_counter() - start
        result = BenchmarkResult(
            write_count=n,
            write_seconds=elapsed,
            write_throughput=n / elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    def search_benchmark(
        self,
        n_queries: int = 100,
        top_k: int = 10,
        hits_per_query: int = 3,
    ) -> BenchmarkResult:
        """Search latency, hit rate and recall.

        - hit rate: fraction of queries whose own topic-memory appears in
          the top-k results (end-to-end retrieval quality);
        - recall@k: fraction of the expected texts (query variant + topic
          siblings) that appear in the top-k results (diversity of recall).
        """
        engine = self.engine
        topics = TOPICS
        names = list(topics)
        latencies: list[float] = []
        hits = 0
        recalls: list[float] = []

        for _ in range(n_queries):
            topic = self.rng.choice(names)
            q_variant = self.rng.choice(topics[topic])
            query = q_variant.replace("user ", "the user ")
            siblings = [v for v in topics[topic] if v != q_variant]
            expected = [q_variant] + self.rng.sample(
                siblings, min(hits_per_query - 1, len(siblings))
            )
            start = time.perf_counter()
            results = engine.search(
                SearchQuery(text=query, top_k=top_k, top_regions=3)
            )
            latencies.append((time.perf_counter() - start) * 1000.0)

            retrieved_texts = [r.memory.text for r in results]
            # hit: the query's own variant is retrieved in the top-k
            if any(expected[0] in t for t in retrieved_texts):
                hits += 1
            found = sum(
                1 for exp in expected if any(exp in t for t in retrieved_texts)
            )
            recalls.append(found / len(expected))

        latencies.sort()
        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

        stats = engine.region_stats()
        mem_stats = engine.memory_stats()
        total = max(1, mem_stats.total)
        return BenchmarkResult(
            search_count=n_queries,
            search_p50_ms=pct(0.50),
            search_p95_ms=pct(0.95),
            search_p99_ms=pct(0.99),
            hit_rate=hits / n_queries,
            recall_at_k=sum(recalls) / n_queries,
            region_count=stats.count,
            avg_region_size=stats.avg_size,
            max_region_size=stats.max_size,
            avg_region_density=stats.avg_density,
            memory_utilization=mem_stats.active / total,
            archived_count=mem_stats.archived,
            details={"k": top_k},
        )

    # ------------------------------------------------------------------ #
    def eval_set(self, engine: object, path: str) -> BenchmarkResult:
        """Evaluate one benchmark asset (iteration 3.1).

        Loads ``{documents, queries}`` from ``path``, stores the documents,
        then measures hit@1/3/5 and search latency over the queries.
        """
        import json as _json

        with open(path, "r", encoding="utf-8") as fh:
            data = _json.load(fh)
        docs = data.get("documents", [])
        queries = data.get("queries", [])
        # add documents one by one and map ONLY the ones that actually
        # entered the store: with v2 presets the write pipeline may drop a
        # document (extraction noise etc.), and a zip() against the returned
        # list would silently misalign every following expectation.
        # The bulk flag suspends live region evolution exactly like
        # engine.add_many, so the resulting space structure is identical.
        id_of: dict[str, str] = {}
        engine.space.set_bulk(True)
        try:
            for d in docs:
                if not d.get("text"):
                    continue
                mem = engine.add(d["text"])
                if d.get("id") and mem.id in engine.memories:
                    id_of[d["id"]] = mem.id
        finally:
            engine.space.set_bulk(False)
        if engine.config.region.auto_evolve and engine.space.write_ops > 0:
            engine.space.manager.evolution_pass(engine.space)

        latencies: list[float] = []
        hits_at: dict[int, int] = {1: 0, 3: 0, 5: 0}
        misses: list[str] = []
        for item in queries:
            q = item["q"]
            top_k = int(item.get("top_k", 5))
            expected = {id_of.get(e, e) for e in item.get("expect", [])}
            start = time.perf_counter()
            results = engine.search(SearchQuery(text=q, top_k=top_k))
            latencies.append((time.perf_counter() - start) * 1000.0)
            found = {h.memory.id for h in results}
            for k in (1, 3, 5):
                if k <= len(results) and found & expected:
                    hits_at[k] += 1
            if not (found & expected):
                misses.append(q)
        n = len(queries)
        latencies.sort()

        def pct(p: float) -> float:
            if not latencies:
                return 0.0
            return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

        return BenchmarkResult(
            search_count=n,
            search_p50_ms=pct(0.50),
            search_p95_ms=pct(0.95),
            search_p99_ms=pct(0.99),
            hit_rate=hits_at[1] / max(1, n),
            region_count=len(engine.space.regions),
            avg_region_size=engine.region_stats().avg_size,
            memory_utilization=(
                sum(1 for m in engine.memories.values()
                    if engine.space.region_for(m.id) is not None)
                / max(1, len(engine.memories))
            ),
            details={
                "set": data.get("name", path),
                "documents": len(docs),
                "hit@1": hits_at[1] / max(1, n),
                "hit@3": hits_at[3] / max(1, n),
                "hit@5": hits_at[5] / max(1, n),
                "misses": misses[:20],
            },
        )

    # ------------------------------------------------------------------ #
    def full(
        self,
        n_memories: int = 2000,
        n_queries: int = 200,
        top_k: int = 10,
        save: Optional[str] = None,
    ) -> BenchmarkResult:
        write = self.write_benchmark(n_memories)
        search = self.search_benchmark(n_queries=n_queries, top_k=top_k)
        merged = write.to_dict()
        merged.update({k: v for k, v in search.to_dict().items() if k not in (
            "write_count", "write_seconds", "write_throughput",
        )})
        result = BenchmarkResult(**merged)
        if save:
            result.to_json(save)
        return result


def main() -> None:
    """Run the benchmark: ``python -m sme.benchmark``."""
    import argparse

    parser = argparse.ArgumentParser(description="SME 写入/检索压测 + 标准评测")
    parser.add_argument("--n-memories", type=int, default=2000)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save", default="", help="结果 JSON 输出路径")
    parser.add_argument("--eval", default="",
                        help="评测资产 JSON（benchmarks/*.json），替换压测为考问评测")
    parser.add_argument("--preset", default="",
                        help="评测时套用的预设（chat/kb_dynamic/kb_static/robot/minimal）")
    args = parser.parse_args()

    from sme.engine import SpatialMemoryEngine

    engine = SpatialMemoryEngine()
    runner = BenchmarkRunner(engine)
    if args.eval:
        if args.preset:
            from sme.config_items import PRESET_BY_KEY, apply_preset, load_config

            cfg = load_config("")
            apply_preset(cfg, PRESET_BY_KEY[args.preset])
            # apply the preset's engine-level knobs directly
            from sme.config import SMEConfig

            engine = SpatialMemoryEngine(
                SMEConfig.from_dict({k: v for k, v in cfg.items() if k != "_help"})
            )
            runner = BenchmarkRunner(engine)
        result = runner.eval_set(engine, args.eval)
        lines = [
            f"set={result.details.get('set')} documents={result.details.get('documents')} "
            f"queries={result.search_count}",
            f"hit@1={result.details['hit@1']:.3f} "
            f"hit@3={result.details['hit@3']:.3f} "
            f"hit@5={result.details['hit@5']:.3f}",
            f"latency p50={result.search_p50_ms:.1f}ms p95={result.search_p95_ms:.1f}ms "
            f"p99={result.search_p99_ms:.1f}ms",
        ]
        if result.details.get("misses"):
            lines.append("misses: " + " | ".join(result.details["misses"][:10]))
        print("\n".join(lines))
        if args.save:
            result.to_json(args.save)
        return
    result = runner.full(
        args.n_memories, args.n_queries, top_k=args.top_k,
        save=args.save or None,
    )
    print("\n".join(result.summary_lines()))


if __name__ == "__main__":
    main()
