from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import time
import os
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from config import RAGConfig
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from upstash_redis.asyncio import Redis

app = FastAPI(title="RAG Pipeline Serverless")

# Configuration
config = RAGConfig.from_env()

# Initialize global clients
try:
    vector_store = VectorStore(config.vector_store)
    rag_pipeline = RAGPipeline(vector_store, llm_model=config.llm.model)
    redis_client = Redis(url=config.redis.url, token=config.redis.token)
except Exception as e:
    print(f"Failed to initialize core services: {e}")
    # Will fail healthchecks

# Setup static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(BASE_DIR, 'static')):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") if os.path.exists(os.path.join(BASE_DIR, 'templates')) else None

# Authentication Dependency
async def verify_token(authorization: str = Header(None)):
    expected_token = config.app.auth_token
    if not authorization or authorization.replace("Bearer ", "") != expected_token:
        # In a real app we would raise 401, but for testing UI we'll just log and let it pass or use cookies.
        # Actually, let's enforce it for the API.
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    session_id: str = "default"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the chat interface"""
    if templates:
        return templates.TemplateResponse(request=request, name="index.html", context={"request": request})
    return HTMLResponse("<h1>RAG Backend Running</h1><p>Static files missing, but API is up.</p>")

@app.post("/api/query")
async def query(req: QueryRequest, auth: None = Depends(verify_token)):
    """Handle chat queries"""
    start_time = time.time()
    
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Query the RAG system
        result = await rag_pipeline.query_async(question, top_k=req.top_k)
        
        response_time = time.time() - start_time
        
        # Save to chat history in Upstash Redis
        history_key = f"chat_history:{req.session_id}"
        message_entry = {
            "question": question,
            "answer": result["answer"],
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": round(response_time * 1000, 2)
        }
        
        # Retrieve existing history, append, and save
        existing_history = await redis_client.get(history_key)
        history = json.loads(existing_history) if existing_history else []
        history.append(message_entry)
        await redis_client.set(history_key, json.dumps(history))
        
        # Increment total metrics
        await redis_client.incr("metrics:total_queries")
        
        response = {
            'success': True,
            'question': question,
            'answer': result['answer'],
            'sources': [
                {
                    'id': doc['id'],
                    'text': doc['text'][:300] + ('...' if len(doc['text']) > 300 else ''),
                    'similarity': round(doc['distance'] * 100, 1),
                }
                for doc in result['metadata']['retrieved_docs']
            ],
            'timestamp': datetime.now().isoformat(),
            'response_time_ms': round(response_time * 1000, 2)
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(session_id: str = "default", auth: None = Depends(verify_token)):
    """Get chat history for current session"""
    history_key = f"chat_history:{session_id}"
    existing_history = await redis_client.get(history_key)
    history = json.loads(existing_history) if existing_history else []
    
    return {
        'success': True,
        'history': history
    }

@app.post("/api/clear")
async def clear_history(session_id: str = "default", auth: None = Depends(verify_token)):
    """Clear chat history for current session"""
    history_key = f"chat_history:{session_id}"
    await redis_client.delete(history_key)
    
    return {
        'success': True,
        'message': 'Chat history cleared'
    }

@app.get("/api/stats")
async def get_stats(auth: None = Depends(verify_token)):
    """Get system statistics"""
    try:
        stats = vector_store.get_stats()
        total_queries = await redis_client.get("metrics:total_queries") or "0"
        
        return {
            'success': True,
            'stats': {
                'total_documents': stats['total_documents'],
                'embedding_model': stats['embedding_model'],
                'llm_model': config.llm.model,
                'total_queries': int(total_queries),
                'collection_name': stats['collection_name'],
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        stats = vector_store.get_stats()
        return {
            'status': 'healthy',
            'documents_indexed': stats['total_documents']
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'unhealthy', 'error': str(e)})

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 5000))
    print(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
