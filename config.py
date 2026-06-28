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
    temperature: float = 0.1
    max_tokens: int = 300
    api_key: str = os.getenv("GOOGLE_API_KEY", "")

@dataclass
class RedisConfig:
    """Configuration for Upstash Redis"""
    url: str = os.getenv("UPSTASH_REDIS_REST_URL", "")
    token: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

@dataclass
class DataConfig:
    """Configuration for data loading"""
    dataset_name: str = "squad_v2"
    split: str = "validation"
    max_samples: int = 100

@dataclass
class AppConfig:
    """Configuration for FastAPI app"""
    auth_token: str = os.getenv("API_AUTH_TOKEN", "super-secret-student-key")

@dataclass
class RAGConfig:
    """Main configuration for RAG system"""
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    data: DataConfig = field(default_factory=DataConfig)
    app: AppConfig = field(default_factory=AppConfig)
    top_k: int = 3

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables"""
        return cls()