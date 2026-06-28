"""
Data Ingestion Script
Load SQuAD data and populate the vector store
"""

from data_loader import SQuADLoader
from vector_store import VectorStore
from config import RAGConfig
from config_utils import create_vector_store_from_config, create_data_loader_from_config
import argparse


def main():
    parser = argparse.ArgumentParser(description="Ingest data into vector store")
    parser.add_argument("--config", type=str, choices=["default"],
                        default="default", help="Configuration profile")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to ingest (default: from config)")
    parser.add_argument("--collection", type=str, default=None,
                        help="Collection name (overrides config)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to use: 'squad_v2' (default) or 'capitals'")
    parser.add_argument("--chunk", action="store_true",
                        help="Chunk documents before embedding for better relevance")
    args = parser.parse_args()

    print("="*60)
    print("RAG DATA INGESTION")
    print("="*60)

    config = RAGConfig.from_env()

    if args.samples:
        config.data.max_samples = args.samples

    if args.collection:
        config.vector_store.index_name = args.collection

    if args.dataset:
        config.data.dataset_name = args.dataset

    print(f"\nConfiguration: {args.config}")
    print(f"Index: {config.vector_store.index_name}")
    print(f"Embedding Model: {config.vector_store.embedding_model}")
    print(f"Dataset: {config.data.dataset_name}")
    if args.chunk:
        print("Chunking: enabled")

    loader = create_data_loader_from_config(config)
    contexts = loader.get_contexts()

    print(f"\nLoaded {len(contexts)} unique contexts")

    vector_store = create_vector_store_from_config(config.vector_store)

    # Pinecone specific
    stats = vector_store.get_stats()
    doc_count = stats["total_documents"]
    if doc_count > 0:
        print(f"Collection already has {doc_count} documents")
        response = input("Reset collection? (y/n): ")
        if response.lower() == 'y':
            vector_store.reset()
        else:
            print("Adding documents to existing collection...")
            vector_store.add_documents(contexts, chunk_documents=args.chunk)
            print(f"Total documents: {vector_store.get_stats()['total_documents']}")
            return

    print("\nAdding documents to vector store...")
    vector_store.add_documents(contexts, chunk_documents=args.chunk)

    print(f"\n✓ Successfully ingested {len(contexts)} documents")
    print(f"  Index: {config.vector_store.index_name}")
    print(f"  Total documents: {vector_store.get_stats()['total_documents']}")

if __name__ == "__main__":
    main()
