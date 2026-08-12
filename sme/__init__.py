"""Spatial Memory Engine (SME).

A next-generation AI long-term memory system. Unlike a traditional vector
database that does Embedding -> TopK, SME organizes the embedding space into
dynamic density-based *regions*:

    Memory -> Embedding Space -> Spatial Region -> Region Retrieval
            -> Memory Retrieval -> LLM

Modules:
    embedding   - pluggable embedding providers (OpenAI-compatible, local, hashing)
    space       - the spatial memory space: nodes, regions, region graph
    managers    - region manager and memory manager (CRUD, archive, versioning)
    retrieval   - two-stage retrieval: region retrieval + hybrid memory retrieval
    ranking     - final score combining semantic/importance/freshness/weight/...
    policy      - global switches (full memory, decay)
    reinforcement - Ebbinghaus-style hit reinforcement
    decay       - time-based decay that never deletes memories
    consolidation - auto-fusion of similar memories into summaries
    graph       - memory graph (reference/cause/conversation/summary edges)
    compression - long-term summarization
    archive     - hot/cold storage
    visualization - 2D projection plotting
    benchmark   - write/search performance benchmark
    api         - FastAPI REST server
    llm         - OpenAI-compatible chat client
    engine      - SpatialMemoryEngine facade
"""

__version__ = "1.2.0"

from sme.config import SMEConfig
from sme.models import Memory, Region, SearchHit, ScoreBreakdown
from sme.engine import SpatialMemoryEngine

__all__ = [
    "__version__",
    "SMEConfig",
    "Memory",
    "Region",
    "SearchHit",
    "ScoreBreakdown",
    "SpatialMemoryEngine",
]
