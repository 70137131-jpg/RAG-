"""
Quick start script for the current FastAPI/Pinecone/RAG stack.
"""

import asyncio
import os
import sys


def check_requirements() -> bool:
    """Check if the packages used by the current architecture are installed."""
    required_packages = [
        "datasets",
        "pinecone",
        "google.genai",
        "upstash_redis",
        "dotenv",
        "requests",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"OK {package}")
        except ImportError:
            print(f"MISSING {package}")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False

    return True


def check_service_config() -> bool:
    """Check environment variables required by Pinecone, the selected LLM, and Redis."""
    from dotenv import load_dotenv

    load_dotenv()
    required_env = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
    ]
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    model = os.getenv("LLM_MODEL", "")
    if not provider:
        provider = "gemini" if "gemini" in model.lower() else "openrouter"

    if provider == "gemini":
        required_env.append("GOOGLE_API_KEY")
    elif provider == "openrouter":
        required_env.append("OPENROUTER_API_KEY")
    else:
        print(f"\nUnsupported LLM_PROVIDER: {provider}")
        return False

    missing = [name for name in required_env if not os.getenv(name)]

    if missing:
        print(f"\nMissing required environment variables: {', '.join(missing)}")
        print("Set them in .env or in your shell before running quickstart.py.")
        return False

    return True


async def run_simple_demo() -> None:
    """Load a small SQuAD sample, ensure Pinecone has data, and query the configured LLM."""
    print("=" * 70)
    print("RAG QUICKSTART")
    print("=" * 70)

    if not check_requirements() or not check_service_config():
        sys.exit(1)

    from config import RAGConfig
    from data_loader import SQuADLoader
    from rag_pipeline import RAGPipeline
    from vector_store import VectorStore

    config = RAGConfig.from_env()

    print("\n[1/4] Loading SQuAD sample...")
    loader = SQuADLoader(dataset_name=config.data.dataset_name, split=config.data.split)
    loader.load(max_samples=50)
    contexts = loader.get_contexts()
    qa_pairs = loader.get_qa_pairs()
    print(f"Loaded {len(contexts)} contexts and {len(qa_pairs)} QA pairs")

    print("\n[2/4] Connecting to Pinecone...")
    vector_store = VectorStore(config.vector_store)
    if vector_store.get_stats()["total_documents"] == 0:
        vector_store.add_documents(contexts)
    print(f"Indexed documents: {vector_store.get_stats()['total_documents']}")

    provider = config.llm.provider or (
        "gemini" if "gemini" in config.llm.model.lower() else "openrouter"
    )
    print(f"\n[3/4] Initializing {provider} RAG pipeline...")
    rag = RAGPipeline(
        vector_store=vector_store,
        llm_model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        llm_provider=config.llm.provider,
        openrouter_api_key=config.llm.openrouter_api_key,
        openrouter_base_url=config.llm.openrouter_base_url,
        openrouter_site_url=config.llm.openrouter_site_url,
        openrouter_app_name=config.llm.openrouter_app_name,
    )

    print("\n[4/4] Running example queries...")
    for i, qa in enumerate(qa_pairs[:3], 1):
        print(f"\nQuery {i}: {qa['question']}")
        result = await rag.query_async(qa["question"], top_k=2)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved contexts: {len(result['metadata']['retrieved_docs'])}")


if __name__ == "__main__":
    asyncio.run(run_simple_demo())
