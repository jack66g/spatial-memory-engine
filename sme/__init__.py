"""Spatial Memory Engine (SME).

A next-generation AI long-term memory system. Unlike a traditional vector
database that does Embedding -> TopK, SME organizes the embedding space into
dynamic density-based *regions*:

    Memory -> Embedding Space -> Spatial Region -> Region Retrieval
            -> Memory Retrieval -> LLM

Package layout:
    engine       - SpatialMemoryEngine facade (top-level)
    config       - SMEConfig and presets (top-level)
    models/utils - shared dataclasses and helpers (top-level)
    space        - the spatial memory space: nodes, regions, region graph
    embedding    - pluggable embedding providers (OpenAI-compatible, local, hashing)
    index        - ANN index abstraction
    llm          - OpenAI-compatible chat client
    retrieval    - two-stage retrieval (region + hybrid), ranking, rerank
    dynamics     - memory dynamics: decay, reinforcement, consolidation,
                   compression, archive, policy
    modules      - v2 extension modules: extraction, qapair, factversion,
                   profile, context, namespaces, observability, noise,
                   memory_graph, bridge (v2), pipeline, factgraph
    storage      - snapshots, pluggable storage backends, write-ahead log
    memory_manager - CRUD/region maintenance orchestration
    visualization - 2D projection plotting
    benchmark    - write/search performance benchmark
    api          - FastAPI REST server
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
