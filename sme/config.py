"""Configuration for the Spatial Memory Engine.

Every subsystem is configurable. Sensible defaults make the engine runnable
out of the box with a deterministic offline hashing embedding provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    provider: str = "hashing"  # hashing | openai | sentence-transformers
    model: str = "text-embedding-3-small"
    dim: int = 64
    base_url: str = ""          # e.g. https://api.openai.com/v1 or any compatible
    api_key: str = ""
    batch_size: int = 32
    normalize: bool = True
    # for the hashing fallback provider
    hash_factors: int = 3
    hash_window: int = 3        # n-gram window for the hashing embedding


@dataclass
class RegionConfig:
    """Density-based region evolution tuning."""

    min_region_size: int = 3
    min_join_cosine: float = 0.70   # join a region when cos(centroid, v) >= this
    split_threshold: int = 48      # members above this may trigger a split
    max_density: float = 120.0     # members / (1 + avg dist to centroid)
    merge_distance: float = 0.12   # centroid distance below which regions may merge
    neighbor_factor: float = 1.8   # neighbor edge if dist < (r1+r2)*factor
    auto_evolve: bool = True
    evolve_interval: int = 40      # run an evolution pass every N write ops
    # ANN acceleration (hnswlib optional; falls back to exact scanning)
    ann_enabled: bool = True
    ann_min_regions: int = 256     # use ANN once regions exceed this count


@dataclass
class RetrievalConfig:
    top_regions: int = 3
    region_options: list[int] = field(default_factory=lambda: [1, 3, 5])
    top_k: int = 10
    vector_weight: float = 0.60
    keyword_weight: float = 0.30
    metadata_weight: float = 0.10
    region_dampening: float = 0.10  # penalty for memories outside top regions
    summary_penalty: float = 0.5  # score multiplier for source="summary" (1.0 = off)
    # BM25 keyword channel: CJK 1-2 grams (iteration 1.2; English unaffected)
    cjk_bigram: bool = True
    # minimum candidate pool for hybrid scoring (iteration 2.2); the pool is
    # max(top_k * 2, candidate_window), so the default keeps top_k=10 behavior
    candidate_window: int = 20


@dataclass
class RankingConfig:
    """Weights of the final score. Must sum to 1.0."""

    semantic: float = 0.40
    importance: float = 0.12
    freshness: float = 0.10
    weight: float = 0.10
    decay: float = 0.08
    hit_count: float = 0.08
    recency: float = 0.06
    region: float = 0.06


@dataclass
class PolicyConfig:
    """Global switches.

    full_memory: True  -> record and keep everything, never filtered out.
                  False -> memories below importance_threshold are excluded.
    decay_enabled:     -> whether weight/importance decay over time.
    """

    full_memory: bool = True
    importance_threshold: float = 0.25
    decay_enabled: bool = True


@dataclass
class ReinforcementConfig:
    """Ebbinghaus-style reinforcement on memory hits."""

    hit_weight_delta: float = 0.10
    hit_importance_delta: float = 0.05
    max_weight: float = 5.0
    max_importance: float = 1.0
    min_importance: float = 0.0
    ebbinghaus_half_life_days: float = 7.0
    ebbinghaus_power: float = 1.2


@dataclass
class DecayConfig:
    half_life_days: float = 30.0
    max_weight_decay: float = 0.85   # weight can decay down to (1 - this)
    max_importance_decay: float = 0.85


@dataclass
class ConsolidationConfig:
    similarity_threshold: float = 0.50
    min_group_size: int = 3
    max_group_size: int = 12
    summary_source: str = "template"  # template | llm


@dataclass
class CompressionConfig:
    age_days_threshold: float = 3.0
    min_region_compact: int = 8
    summary_source: str = "template"


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout: float = 60.0
    temperature: float = 0.3
    max_tokens: int = 1024
    reasoning_effort: str | None = None  # e.g. "none" for reasoning-capable models
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class StorageConfig:
    path: str = "data/engine.json"
    autosave: bool = True
    autosave_interval: int = 100
    compress: bool = True
    backend: str = "json"  # json | sqlite (module 08, default json = v1 behavior)


# --------------------------------------------------------------------------- #
# v2 modules (v2 模块设计). Every module has `enabled` defaulting to False;
# with all modules disabled the engine behaves exactly like v1.
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionConfig:
    """Module 01 - LLM/rules fact extraction on the write path."""

    enabled: bool = False
    mode: str = "llm"            # llm | rules | off
    store_assistant: bool = False  # AI answers are NOT stored by default
    dedup_threshold: float = 0.92  # cosine above this => skip as duplicate
    fallback_rules: bool = True  # fall back to rules when LLM is unconfigured


@dataclass
class FactVersionConfig:
    """Module 05 - fact versioning and correction handling."""

    enabled: bool = False
    correct_markers: list[str] = field(
        default_factory=lambda: [
            "其实", "不对", "更正", "纠正", "说错", "搞错", "记错",
            "其实不是", "改一下", "不是的", "错了",
        ]
    )
    stale_penalty: float = 0.3      # score multiplier lost by superseded facts
    dedup_threshold: float = 0.92   # cosine for "same fact, repeated statement"
    correction_threshold: float = 0.72  # cosine for "rephrased correction"


@dataclass
class NoiseConfig:
    """Module 06 - retrieval-side noise suppression."""

    enabled: bool = False
    dup_penalty: float = 0.5        # same-sentence repetition penalty
    template_penalty: float = 0.3   # n-gram template coverage penalty
    min_density: float = 0.10       # info-density floor below which penalize
    ngram_window: int = 4           # template n-gram size
    dup_window: int = 300           # how many memories are scanned for dup counts


@dataclass
class QAPairConfig:
    """Module 02 - question/answer pairs with direct replay."""

    enabled: bool = False
    max_age_days: float = 180.0
    answer_top_k: int = 3
    similarity_threshold: float = 0.85  # question-match cosine for replay


@dataclass
class FactGraphConfig:
    """Module 03 - entity/relation temporal knowledge graph."""

    enabled: bool = False
    extract_mode: str = "llm"       # llm | rules | off
    max_depth: int = 3
    hop_decay: float = 0.5
    max_entities: int = 50


@dataclass
class ProfileConfig:
    """Module 04 - per-user profile facts and snapshots."""

    enabled: bool = False
    snapshot_every: int = 50        # profile facts written before auto-snapshot
    top_facts: int = 20             # facts per snapshot
    boost_weight: float = 0.15      # retrieval boost for profile facts


@dataclass
class PersistenceConfig:
    """Module 07 - write-ahead log + periodic full checkpoints."""

    enabled: bool = False
    wal_path: str = ""              # empty => <storage.path>.wal
    checkpoint_every: int = 10      # full snapshot every N write ops
    sync_mode: str = "fsync"        # fsync | off


@dataclass
class ApiConfig:
    """Module 09 - REST server + SDK.

    The server is started explicitly via ``python -m sme.api`` (the legacy
    ``enabled`` flag was removed; the key is tolerated on old snapshots and
    simply ignored by ``from_dict``'s hasattr guard).
    """

    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str = ""


@dataclass
class ObservabilityConfig:
    """Module 10 - telemetry events and reports."""

    enabled: bool = False
    sample_rate: float = 1.0
    max_events: int = 20000


@dataclass
class ContextConfig:
    """Module 11 - layered context management."""

    enabled: bool = False
    budget_tokens: int = 4096
    reserve_profile: int = 500
    recent_rounds: int = 20


@dataclass
class RerankConfig:
    """Module 13 (optional) - cross-encoder re-ranking (iteration 2.3)."""

    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"   # any CrossEncoder-compatible model
    top_n: int = 50                         # hits considered for re-ranking


@dataclass
class NamespaceConfig:
    """Module 12 - multi-user / multi-scene isolation."""

    enabled: bool = False
    default_ns: str = "default"


@dataclass
class VisualizationConfig:
    projection: str = "pca"  # pca | random | tsne(fallback to pca)
    figure_size: tuple[int, int] = (12, 9)
    dpi: int = 120
    color_map: str = "tab20"
    show_graph: bool = True


@dataclass
class SMEConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reinforcement: ReinforcementConfig = field(default_factory=ReinforcementConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    # v2 modules (v2 模块设计) - all disabled by default
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    factversion: FactVersionConfig = field(default_factory=FactVersionConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    qapair: QAPairConfig = field(default_factory=QAPairConfig)
    factgraph: FactGraphConfig = field(default_factory=FactGraphConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    namespaces: NamespaceConfig = field(default_factory=NamespaceConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    seed: int | None = 42
    # keys explicitly present in the source dict (from_dict): used by
    # engine._calibrate_region_threshold to tell "user wrote 0.70" from
    # "default 0.70" (never serialized, runtime-only bookkeeping)
    _explicit_keys: set = field(default_factory=set, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict) -> "SMEConfig":
        """Build a config from a nested dict (JSON friendly)."""
        cfg = cls()
        mapping = {
            "embedding": EmbeddingConfig,
            "region": RegionConfig,
            "retrieval": RetrievalConfig,
            "ranking": RankingConfig,
            "policy": PolicyConfig,
            "reinforcement": ReinforcementConfig,
            "decay": DecayConfig,
            "consolidation": ConsolidationConfig,
            "compression": CompressionConfig,
            "llm": LLMConfig,
            "storage": StorageConfig,
            "visualization": VisualizationConfig,
            "extraction": ExtractionConfig,
            "factversion": FactVersionConfig,
            "noise": NoiseConfig,
            "qapair": QAPairConfig,
            "factgraph": FactGraphConfig,
            "profile": ProfileConfig,
            "persistence": PersistenceConfig,
            "api": ApiConfig,
            "observability": ObservabilityConfig,
            "context": ContextConfig,
            "namespaces": NamespaceConfig,
            "rerank": RerankConfig,
        }
        for key, ctor in mapping.items():
            if key in data and isinstance(data[key], dict):
                current = getattr(cfg, key)
                for k, v in data[key].items():
                    if hasattr(current, k):
                        setattr(current, k, v)
                        cfg._explicit_keys.add(f"{key}.{k}")
        if "seed" in data:
            cfg.seed = data["seed"]
        return cfg

    def to_dict(self) -> dict:
        out: dict = {"seed": self.seed}
        for key in (
            "embedding",
            "region",
            "retrieval",
            "ranking",
            "policy",
            "reinforcement",
            "decay",
            "consolidation",
            "compression",
            "llm",
            "storage",
            "visualization",
            "extraction",
            "factversion",
            "noise",
            "qapair",
            "factgraph",
            "profile",
            "persistence",
            "api",
            "observability",
            "context",
            "namespaces",
            "rerank",
        ):
            out[key] = {k: v for k, v in getattr(self, key).__dict__.items()}
        return out
