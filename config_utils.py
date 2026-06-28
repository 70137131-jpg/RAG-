"""
Utility functions to create components from configuration
"""

from config import RAGConfig, VectorStoreConfig, LLMConfig
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from data_loader import SQuADLoader, CapitalsLoader


def create_vector_store_from_config(config: VectorStoreConfig) -> VectorStore:
    """
    Create a VectorStore instance from configuration
    """
    return VectorStore(config)


def create_rag_pipeline_from_config(config: RAGConfig, vector_store: VectorStore = None) -> RAGPipeline:
    """
    Create a RAGPipeline instance from configuration
    """
    if vector_store is None:
        vector_store = create_vector_store_from_config(config.vector_store)

    return RAGPipeline(
        vector_store=vector_store,
        llm_model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )


def create_data_loader_from_config(config: RAGConfig):
    """
    Create a data loader instance from configuration.
    Supports 'squad'/'squad_v2' and 'capitals'.
    """
    ds_name = (config.data.dataset_name or "squad_v2").lower()
    if ds_name in ("squad", "squad_v2"):
        loader = SQuADLoader(
            dataset_name=config.data.dataset_name,
            split=config.data.split
        )
        loader.load(max_samples=config.data.max_samples)
        return loader
    elif ds_name == "capitals":
        loader = CapitalsLoader()
        loader.load(max_samples=config.data.max_samples)
        return loader
    else:
        raise ValueError(f"Unsupported dataset_name '{config.data.dataset_name}'. Use 'squad_v2' or 'capitals'.")
