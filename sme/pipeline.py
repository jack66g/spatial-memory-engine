"""Write / search pipelines for the v2 modules (v2 模块设计).

Every module participates as a *stage*. A stage is inert unless its
module's ``enabled`` flag is True, so with all modules disabled the pipeline
is a transparent no-op and the engine behaves exactly like v1.

Write flow (v2 模块设计 5.1) — stages run in registration order::

    engine.add(text)
      -> extraction     (01) : filter raw text -> list[Fact] (+ correction
                               marker detection, cosine dedup)
      -> canonical      (01/05/02) : versioning/correction resolution and
                               the "what gets stored" decision
      -> storage        (02/03/04) : store canonical items, wire QA pairs,
                               fact-graph entities and profile facts
      -> answer_capture (02) : assistant answer -> QA pair replay entry

Search flow (v2 模块设计 5.2)::

    engine.search(q)
      -> v1 two-stage retrieval (unchanged)
      -> A qapair.lookup   : direct answer replay
      -> B factgraph.multi_hop : graph multi-hop candidates
      -> C noise.apply     : noise/duplicate re-ranking
      -> D profile.boost   : profile facts weighted up
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class WriteContext:
    """Mutable state flowing through the write pipeline stages."""

    engine: Any = None
    text: str = ""
    metadata: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    source: str = "user"
    link_to: str | None = None
    link_kind: str = "reference"
    embedding: Any = None
    facts: list = field(default_factory=list)   # list[Fact]
    drop: bool = False                          # do not store the raw text
    assistant: bool = False
    primary: Any = None                         # the primary stored memory
    pending_question: dict | None = None        # qapair capture between turns
    extra: dict = field(default_factory=dict)   # module scratch space


class WriteStage(Protocol):
    name: str

    def enabled(self, engine: Any) -> bool:
        """Whether this stage does anything under the engine's config."""

    def run(self, engine: Any, ctx: WriteContext) -> WriteContext:
        """Process the context; return it (possibly mutated)."""


class WritePipeline:
    """Ordered list of write stages; disabled stages are skipped."""

    def __init__(self) -> None:
        self.stages: list[WriteStage] = []

    def register(self, stage: WriteStage) -> None:
        self.stages.append(stage)

    def active(self, engine: Any) -> bool:
        return any(getattr(s, "enabled")(engine) for s in self.stages)

    def names(self, engine: Any) -> list[str]:
        return [s.name for s in self.stages if getattr(s, "enabled")(engine)]

    def run(self, engine: Any, ctx: WriteContext) -> WriteContext:
        for stage in self.stages:
            if getattr(stage, "enabled")(engine):
                ctx = stage.run(engine, ctx)
        return ctx
