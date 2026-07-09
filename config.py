"""
Configuration classes for RAG system
Provides typed configuration with environment variable support
"""

from typing import Optional
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class VectorStoreConfig:
    """Configuration for vector store (Pinecone)"""
    api_key: str = os.getenv("PINECONE_API_KEY", "")
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "squad-rag-integrated")
    embedding_model: str = "multilingual-e5-large"
    chunk_size: int = 500
    chunk_overlap: int = 50

@dataclass
class LLMConfig:
    """Configuration for LLM (Generation)"""
    model: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    provider: str = os.getenv("LLM_PROVIDER", "").lower()
    temperature: float = 0.1
    max_tokens: int = 300
    api_key: str = os.getenv("GOOGLE_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    openrouter_site_url: str = os.getenv("OPENROUTER_SITE_URL", "")
    openrouter_app_name: str = os.getenv("OPENROUTER_APP_NAME", "RAG Pipeline")

@dataclass
class RedisConfig:
    """Configuration for Upstash Redis"""
    url: str = os.getenv("UPSTASH_REDIS_REST_URL", "")
    token: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL chat logs"""
    dsn: str = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN", "")
    retention_days: int = int(os.getenv("CHAT_LOG_RETENTION_DAYS", "7"))

@dataclass
class DataConfig:
    """Configuration for data loading"""
    dataset_name: str = "squad_v2"
    split: str = "validation"
    max_samples: Optional[int] = None

@dataclass
class RAGConfig:
    """Main configuration for RAG system"""
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    data: DataConfig = field(default_factory=DataConfig)
    top_k: int = 3

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables"""
        return cls()
