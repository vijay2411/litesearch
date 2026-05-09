from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    dimensions: int = 768
    model: str = "embeddinggemma"
    ollama_url: str = "http://localhost:11434"


@dataclass
class RerankerConfig:
    llm_judge_backend: str = "gemini"
    llm_judge_model: str = "gemma4"
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    gemini_api_key_file: str = ""
    mmr_lambda: float = 0.7
    rerank_cli: str = "rerank"


@dataclass
class SearchConfig:
    default_mode: str = "hybrid"
    default_reranker_str: str = "none"
    candidate_pool_size: int = 50
    default_top_k: int = 10


@dataclass
class TimeDecayConfig:
    default_half_life_days: int = 90
    default_weight: float = 0.5


@dataclass
class ChunkingConfig:
    soft_max_chars: int = 1500


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8900
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class LiteSearchConfig:
    db_path: str = "litesearch.db"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    time_decay: TimeDecayConfig = field(default_factory=TimeDecayConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    vault_path: Optional[str] = None
