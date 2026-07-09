import asyncio
import os
from config import RAGConfig
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

async def run_tests():
    load_dotenv()
    config = RAGConfig.from_env()
    vector_store = VectorStore(config.vector_store)
    rag_pipeline = RAGPipeline(vector_store, llm_model=config.llm.model)
    
    questions = [
        "What is the capital of France?",
        "Who were the Normans and when did they invade England?",
        "What is the airspeed velocity of an unladen swallow?",
        "Hi",
        "Tell me about Mercury"
    ]
    
    for q in questions:
        print(f"\\n\\n=== Question: {q} ===")
        res = await rag_pipeline.query_async(q, top_k=3)
        print(f"Answer: {res['answer']}")
        print(f"Metadata (num_contexts): {res['metadata'].get('num_contexts', 0)}")

if __name__ == "__main__":
    asyncio.run(run_tests())
