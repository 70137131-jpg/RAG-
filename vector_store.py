"""
Vector Store using Pinecone with Integrated Inference
Handles embedding and retrieval of documents using Serverless Pinecone
"""

from pinecone import Pinecone
from typing import List, Dict, Optional
from config import VectorStoreConfig
import time

class VectorStore:
    """Vector database for storing and retrieving document embeddings"""

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store with Pinecone
        """
        self.api_key = config.api_key
        self.index_name = config.index_name
        self.embedding_model = config.embedding_model
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

        if not self.api_key:
            print("WARNING: Pinecone API key not found in environment!")
            
        # Initialize Pinecone Client
        self.pc = Pinecone(api_key=self.api_key)
        self.index = self.pc.Index(self.index_name)
        
        print(f"Connected to Pinecone index: {self.index_name}")

    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """Split text into overlapping chunks by words"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

            if current_size >= (chunk_size * 5): # rough char equivalent of words
                chunks.append(' '.join(current_chunk))
                overlap_words = int(len(current_chunk) * (overlap / chunk_size))
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_size = sum(len(w) + 1 for w in current_chunk)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def add_documents(self, documents: List[Dict[str, str]], chunk_documents: bool = False, batch_size: int = 90):
        """
        Add documents to the vector store using Pinecone Inference API
        """
        print(f"Adding {len(documents)} documents to Pinecone...")
        all_texts = []
        all_ids = []
        all_metadatas = []

        for doc in documents:
            doc_id = doc['id']
            doc_text = doc['text']

            if chunk_documents:
                chunks = self.chunk_text(doc_text)
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_ids.append(f"{doc_id}_chunk_{i}")
                    all_metadatas.append({"source_id": doc_id, "text": chunk, "chunk_index": i})
            else:
                all_texts.append(doc_text)
                all_ids.append(doc_id)
                all_metadatas.append({"source_id": doc_id, "text": doc_text})

        # Process in batches
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]

            # Generate embeddings via Pinecone Inference API with retry logic
            print(f"Embedding batch {i//batch_size + 1}...")
            
            # Simple retry loop for 429 Too Many Requests
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    embeddings_response = self.pc.inference.embed(
                        model=self.embedding_model,
                        inputs=batch_texts,
                        parameters={"input_type": "passage", "truncate": "END"}
                    )
                    break # Success
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = 15 * (attempt + 1)
                        print(f"Rate limited (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise e
            
            # Extract actual float arrays
            vectors = [record["values"] for record in embeddings_response]
            
            # Format for upsert: list of dicts or tuples
            upsert_data = zip(batch_ids, vectors, batch_metadatas)
            
            self.index.upsert(vectors=list(upsert_data))
            time.sleep(1.0) # slightly longer simple rate limit pause

        print("Successfully added documents to Pinecone.")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant documents
        """
        # Embed query via Inference API
        query_embedding_res = self.pc.inference.embed(
            model=self.embedding_model,
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"}
        )
        query_vector = query_embedding_res[0]["values"]

        # Search Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        formatted_results = []
        for match in results["matches"]:
            formatted_results.append({
                'id': match["id"],
                'text': match["metadata"].get("text", ""),
                'metadata': match["metadata"],
                'distance': 1.0 - match["score"] # cosine distance
            })

        return formatted_results

    def reset(self):
        """Delete all vectors in the index"""
        self.index.delete(delete_all=True)
        print(f"Deleted all records in '{self.index_name}'")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        stats = self.index.describe_index_stats()
        return {
            "collection_name": self.index_name,
            "total_documents": stats.total_vector_count,
            "embedding_model": self.embedding_model,
            "persist_directory": "pinecone-cloud"
        }
