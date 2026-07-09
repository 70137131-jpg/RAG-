"""
Configuration classes for RAG system
Provides typed configuration with environment variable support
"""

from typing import Optional
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    """Read an environment variable while treating empty strings as unset."""
    return os.getenv(name) or default


@dataclass
class VectorStoreConfig:
    """Configuration for vector store (Pinecone)"""
    api_key: str = ""
    index_name: str = "squad-rag-integrated"
    embedding_model: str = "multilingual-e5-large"
    chunk_size: int = 500
    chunk_overlap: int = 50

@dataclass
class LLMConfig:
    """Configuration for LLM (Generation)"""
    model: str = "gemini-1.5-flash"
    provider: str = ""
    temperature: float = 0.1
    max_tokens: int = 300
    api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_site_url: str = ""
    openrouter_app_name: str = "RAG Pipeline"

@dataclass
class RedisConfig:
    """Configuration for Upstash Redis"""
    url: str = ""
    token: str = ""

@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL chat logs"""
    dsn: str = ""
    retention_days: int = 7

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
        return cls(
            vector_store=VectorStoreConfig(
                api_key=_env("PINECONE_API_KEY"),
                index_name=_env("PINECONE_INDEX_NAME", "squad-rag-integrated"),
            ),
            llm=LLMConfig(
                model=_env("LLM_MODEL", "gemini-1.5-flash"),
                provider=_env("LLM_PROVIDER").lower(),
                api_key=_env("GOOGLE_API_KEY"),
                openrouter_api_key=_env("OPENROUTER_API_KEY"),
                openrouter_base_url=_env(
                    "OPENROUTER_BASE_URL",
                    "https://openrouter.ai/api/v1/chat/completions",
                ),
                openrouter_site_url=_env("OPENROUTER_SITE_URL"),
                openrouter_app_name=_env("OPENROUTER_APP_NAME", "RAG Pipeline"),
            ),
            redis=RedisConfig(
                url=_env("UPSTASH_REDIS_REST_URL"),
                token=_env("UPSTASH_REDIS_REST_TOKEN"),
            ),
            postgres=PostgresConfig(
                dsn=_env("DATABASE_URL") or _env("POSTGRES_DSN"),
                retention_days=int(_env("CHAT_LOG_RETENTION_DAYS", "7")),
            ),
        )
