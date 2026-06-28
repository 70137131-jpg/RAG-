import asyncio
from config import RAGConfig
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

async def main():
    try:
        config = RAGConfig.from_env()
        print("Initializing VectorStore...")
        vector_store = VectorStore(config.vector_store)
        print("Initializing RAGPipeline...")
        rag_pipeline = RAGPipeline(vector_store, llm_model=config.llm.model)
        
        print("Querying pipeline...")
        result = await rag_pipeline.query_async("What is the capital of France?", top_k=2)
        print("Success!")
        print(result["answer"])
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
