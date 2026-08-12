"""Module 03 - FactGraph: entity/relation temporal knowledge graph.

Models the Zep/Graphiti-style temporal knowledge graph: named entities
(person / place / item / org) and predicates between them, each with a
validity window (``valid_at`` / ``invalid_at``). A new statement about an
entity pair invalidates the old relation (the newer company wins).

Retrieval: ``multi_hop_query`` walks valid relations from the entities found
in the query and surfaces the memories behind them (hop-decayed), upgrading
the v1 ``graph_expand`` into a temporal multi-hop graph walk.

Disabled => no graph is built and no multi-hop candidates are added.
"""

from sme.factgraph.extractor import FactGraphExtractor
from sme.factgraph.store import FactGraph

__all__ = ["FactGraph", "FactGraphExtractor"]
