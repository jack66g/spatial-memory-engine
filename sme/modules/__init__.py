"""v2 extension modules: extraction, qapair, factversion, profile, context,
namespaces, observability, noise, memory graph, v2 bridge and write pipeline."""

from sme.modules.bridge import (
    AnswerCaptureStage,
    CanonicalStage,
    ExtractionStage,
    QUESTION_RANK_PENALTY,
    StorageStage,
    V2Bridge,
)
from sme.modules.context import ContextManager, estimate_tokens
from sme.modules.extraction import (
    EMOTION_WORDS,
    EXTRACTION_PROMPT,
    QUESTION_END,
    QUESTION_WORDS,
    TRAILING_PARTICLES,
    ExtractionEngine,
    is_pure_emotion,
    is_question,
    parse_llm_facts,
)
from sme.modules.factversion import FactVersion, STALE_TAG, SUPERSEDE_TAG
from sme.modules.memory_graph import (
    KIND_CAUSE,
    KIND_CONVERSATION,
    KIND_NEIGHBOR,
    KIND_PARENT,
    KIND_REFERENCE,
    KIND_SUMMARY,
    MemoryGraph,
)
from sme.modules.namespaces import NS_KEY, NamespaceView, Namespaces
from sme.modules.noise import NoiseScorer
from sme.modules.observability import MemoryTelemetry
from sme.modules.pipeline import WriteContext, WritePipeline, WriteStage
from sme.modules.profile import PROFILE_TAG, SNAPSHOT_TAG, UserProfile
from sme.modules.qapair import QUESTION_TAG, QAPairStore, looks_like_question

from sme.modules.factgraph import FactGraph, FactGraphExtractor

__all__ = [
    "ExtractionEngine",
    "QAPairStore",
    "FactVersion",
    "UserProfile",
    "ContextManager",
    "Namespaces",
    "NS_KEY",
    "MemoryTelemetry",
    "NoiseScorer",
    "MemoryGraph",
    "KIND_CAUSE",
    "KIND_CONVERSATION",
    "KIND_NEIGHBOR",
    "KIND_PARENT",
    "KIND_REFERENCE",
    "KIND_SUMMARY",
    "V2Bridge",
    "WriteContext",
    "WritePipeline",
    "WriteStage",
    "FactGraph",
    "FactGraphExtractor",
    # bridge stages / constants
    "ExtractionStage",
    "CanonicalStage",
    "StorageStage",
    "AnswerCaptureStage",
    "QUESTION_RANK_PENALTY",
    # context helpers
    "estimate_tokens",
    # extraction helpers / constants
    "is_question",
    "is_pure_emotion",
    "parse_llm_facts",
    "QUESTION_WORDS",
    "QUESTION_END",
    "EMOTION_WORDS",
    "EXTRACTION_PROMPT",
    "TRAILING_PARTICLES",
    # factversion tags
    "STALE_TAG",
    "SUPERSEDE_TAG",
    # namespaces
    "NamespaceView",
    # profile tags
    "PROFILE_TAG",
    "SNAPSHOT_TAG",
    # qapair
    "QUESTION_TAG",
    "looks_like_question",
]
