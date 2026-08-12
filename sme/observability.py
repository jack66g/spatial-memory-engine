"""Module 10 - Observability: memory telemetry and reports.

Records an event stream (add / search / hit / reinforce / save / ...),
computes aggregates (hit rate, latency, distribution) and exports
JSON / CSV reports. Disabled => no events are recorded (zero overhead).
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Optional

from sme.config import ObservabilityConfig
from sme.utils import now


class MemoryTelemetry:
    def __init__(self, config: ObservabilityConfig) -> None:
        self.config = config
        self.events: list[dict[str, Any]] = []
        self.started_at = now()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------ #
    def record(self, event: str, **data: Any) -> None:
        if not self.enabled:
            return
        import random

        if self.config.sample_rate < 1.0 and random.random() > self.config.sample_rate:
            return
        entry = {"ts": now(), "event": event}
        entry.update(data)
        self.events.append(entry)
        if len(self.events) > self.config.max_events:
            self.events = self.events[-self.config.max_events :]

    def search(self, query: str, top_k: int, hits: int, ms: float,
               final_scores: Optional[list[float]] = None) -> None:
        self.record("search", query=query[:100], top_k=top_k, hits=hits,
                    latency_ms=round(ms, 1),
                    scores=([round(s, 3) for s in final_scores] if final_scores else []))

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        evs = self.events
        searches = [e for e in evs if e["event"] == "search"]
        adds = [e for e in evs if e["event"] == "add"]
        drops = [e for e in evs if e["event"] == "drop"]
        hits_ev = [e for e in evs if e["event"] == "reinforce"]
        latency = sorted(e.get("latency_ms", 0.0) for e in searches)
        empty = sum(1 for e in searches if e.get("hits", 0) == 0)
        zero_hits = sum(1 for e in searches if not e.get("scores"))
        avg = (sum(latency) / len(latency)) if latency else 0.0

        def pct(p: float) -> float:
            if not latency:
                return 0.0
            return round(latency[min(len(latency) - 1, int(p * len(latency)))], 1)

        return {
            "events": len(evs),
            "searches": len(searches),
            "adds": len(adds),
            "drops": len(drops),
            "reinforces": len(hits_ev),
            "avg_search_ms": round(avg, 1),
            "p50_search_ms": pct(0.50),
            "p95_search_ms": pct(0.95),
            "p99_search_ms": pct(0.99),
            "max_search_ms": round(max(latency), 1) if latency else 0.0,
            "empty_searches": empty,
            "no_result_searches": zero_hits,
            "uptime_s": round(now() - self.started_at, 1),
        }

    def report(self, engine: Any = None) -> dict[str, Any]:
        """Full report: summary + token cost + memory distribution."""
        out: dict[str, Any] = {
            "summary": self.summary(),
            "events": list(self.events[-2000:]),  # keep the report bounded
        }
        if engine is not None:
            llm = getattr(engine, "llm", None)
            if llm is not None:
                out["cost"] = {
                    "calls": llm.calls,
                    "prompt_tokens": llm.total_usage.get("prompt_tokens", 0),
                    "completion_tokens": llm.total_usage.get("completion_tokens", 0),
                    "total_tokens": llm.total_usage.get("total_tokens", 0),
                }
            out["memory_distribution"] = self._memory_distribution(engine)
        return out

    @staticmethod
    def _memory_distribution(engine: Any) -> dict[str, Any]:
        memories = engine.memories
        by_source: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        archived = 0
        for m in memories.values():
            by_source[m.source] = by_source.get(m.source, 0) + 1
            for t in m.tags:
                by_tag[t] = by_tag.get(t, 0) + 1
            if m.archived:
                archived += 1
        return {
            "total": len(memories),
            "active": len(memories) - archived,
            "archived": archived,
            "by_source": by_source,
            "by_tag": dict(sorted(by_tag.items(), key=lambda kv: -kv[1])[:20]),
        }

    # ------------------------------------------------------------------ #
    def export_json(self, path: str, engine: Any = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.report(engine), fh, ensure_ascii=False, indent=2)
        return path

    def export_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if not self.events:
            return path
        keys = sorted({k for e in self.events for k in e})
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for e in self.events:
                writer.writerow({k: e.get(k, "") for k in keys})
        return path
