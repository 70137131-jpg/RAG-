from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status, Query as FastAPIQuery
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
import time
import os
import re
import uuid
import json
import hmac
import copy
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from config import RAGConfig
from config_utils import create_data_loader_from_config
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from postgres_chat_log import PostgresChatLogStore
from upstash_redis.asyncio import Redis
import asyncio
import logging

logger = logging.getLogger("rag_app")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SESSION_COOKIE = "rag_session_id"
admin_auth_scheme = HTTPBearer(auto_error=False)

# Setup static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="templates") if os.path.exists(os.path.join(BASE_DIR, 'templates')) else None


def validate_required_services(config: RAGConfig) -> None:
    """Fail before serving traffic if a required external service is missing."""
    missing = []
    llm_provider = config.llm.provider or (
        "gemini" if "gemini" in config.llm.model.lower() else "openrouter"
    )
    if not config.vector_store.api_key:
        missing.append("PINECONE_API_KEY")
    if not config.vector_store.index_name:
        missing.append("PINECONE_INDEX_NAME")
    if llm_provider == "gemini" and not config.llm.api_key:
        missing.append("GOOGLE_API_KEY")
    if llm_provider == "openrouter" and not config.llm.openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    if not config.redis.url:
        missing.append("UPSTASH_REDIS_REST_URL")
    if not config.redis.token:
        missing.append("UPSTASH_REDIS_REST_TOKEN")
    if not config.postgres.dsn:
        missing.append("DATABASE_URL or POSTGRES_DSN")

    if missing:
        raise RuntimeError(
            "Missing required service configuration: "
            + ", ".join(missing)
            + ". Set these environment variables before starting the app."
        )


def create_services(config: RAGConfig):
    validate_required_services(config)
    vector_store = VectorStore(config.vector_store)
    rag_pipeline = RAGPipeline(
        vector_store,
        llm_model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        llm_provider=config.llm.provider,
        openrouter_api_key=config.llm.openrouter_api_key,
        openrouter_base_url=config.llm.openrouter_base_url,
        openrouter_site_url=config.llm.openrouter_site_url,
        openrouter_app_name=config.llm.openrouter_app_name,
    )
    redis_client = Redis(url=config.redis.url, token=config.redis.token)
    chat_log_store = PostgresChatLogStore(
        dsn=config.postgres.dsn,
        retention_days=config.postgres.retention_days,
    )
    return vector_store, rag_pipeline, redis_client, chat_log_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = RAGConfig.from_env()
    vector_store, rag_pipeline, redis_client, chat_log_store = create_services(config)
    await chat_log_store.connect()
    await chat_log_store.initialize()
    await chat_log_store.purge_old_logs()
    app.state.config = config
    app.state.vector_store = vector_store
    app.state.rag_pipeline = rag_pipeline
    app.state.redis_client = redis_client
    app.state.chat_log_store = chat_log_store
    app.state.server_start_time = time.time()
    try:
        yield
    finally:
        await chat_log_store.close()


app = FastAPI(title="RAG Pipeline Serverless", lifespan=lifespan)

# CORS: the UI is served same-origin, so cross-origin access is opt-in only.
# Wildcard origins must never be combined with credentials (cookie theft risk).
from fastapi.middleware.cors import CORSMiddleware
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials="*" not in _cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if os.path.exists(os.path.join(BASE_DIR, 'static')):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def get_config(request: Request) -> RAGConfig:
    return request.app.state.config


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_rag_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.rag_pipeline


def get_redis_client(request: Request) -> Redis:
    return request.app.state.redis_client


def get_chat_log_store(request: Request) -> PostgresChatLogStore:
    return request.app.state.chat_log_store


SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def is_valid_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_PATTERN.fullmatch(session_id))


def set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        secure=os.getenv("COOKIE_SECURE", "").lower() == "true",
        max_age=60 * 60 * 24 * 30,
    )


def get_session_id(request: Request, response: Response) -> str:
    session_id = request.cookies.get(SESSION_COOKIE)
    if not session_id or not is_valid_session_id(session_id):
        session_id = uuid.uuid4().hex
        set_session_cookie(response, session_id)
    return session_id


def get_admin_token() -> str:
    return os.getenv("ADMIN_API_TOKEN") or os.getenv("API_AUTH_TOKEN", "")


def require_admin_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(admin_auth_scheme),
) -> None:
    token = get_admin_token()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API token is not configured",
        )
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(credentials.credentials, token)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

    @field_validator('question')
    @classmethod
    def question_not_too_long(cls, v):
        if len(v) > 5000:
            raise ValueError('Question must be 5000 characters or fewer')
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

    @field_validator('top_k')
    @classmethod
    def top_k_bounded(cls, v):
        if v < 1 or v > 20:
            raise ValueError('top_k must be between 1 and 20')
        return v


class SwitchChatRequest(BaseModel):
    session_id: str

REDIS_HISTORY_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days
REDIS_MAX_HISTORY_ENTRIES = 200  # Cap conversation length

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, response: Response):
    """Render the chat interface"""
    session_id = get_session_id(request, response)
    if templates:
        template_response = templates.TemplateResponse(request=request, name="index.html", context={"request": request})
        # Forward Set-Cookie header properly instead of raw_headers hack
        cookie_value = request.cookies.get(SESSION_COOKIE)
        if not cookie_value:
            template_response.set_cookie(
                SESSION_COOKIE,
                session_id,
                httponly=True,
                samesite="lax",
                secure=os.getenv("COOKIE_SECURE", "").lower() == "true",
                max_age=60 * 60 * 24 * 30,
            )
        return template_response
    return HTMLResponse("<h1>RAG Backend Running</h1><p>Static files missing, but API is up.</p>")

@app.post("/api/query")
async def query(
    req: QueryRequest,
    session_id: str = Depends(get_session_id),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline),
    redis_client: Redis = Depends(get_redis_client),
    chat_log_store: PostgresChatLogStore = Depends(get_chat_log_store),
):
    """Handle chat queries"""
    start_time = time.time()
    
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Query the RAG system
        result = await rag_pipeline.query_async(question, top_k=req.top_k)
        
        response_time = time.time() - start_time
        
        response_time_ms = round(response_time * 1000, 2)
        sources = [
            {
                'id': doc['id'],
                'text': doc['text'][:300] + ('...' if len(doc['text']) > 300 else ''),
                # doc['distance'] = 1 - cosine_similarity, so similarity = (1 - distance) * 100
                'similarity': round((1 - doc['distance']) * 100, 1),
            }
            for doc in result['metadata']['retrieved_docs']
        ]

        # Save to chat history in Upstash Redis
        history_key = f"chat_history:{session_id}"
        message_entry = {
            "question": question,
            "answer": result["answer"],
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": response_time_ms
        }
        
        # Retrieve existing history, append, and save with TTL + size cap
        existing_history = await redis_client.get(history_key)
        history = json.loads(existing_history) if existing_history else []
        history.append(message_entry)
        # Cap history size to prevent unbounded growth
        if len(history) > REDIS_MAX_HISTORY_ENTRIES:
            history = history[-REDIS_MAX_HISTORY_ENTRIES:]
        await redis_client.set(history_key, json.dumps(history), ex=REDIS_HISTORY_TTL_SECONDS)
        
        # Increment total metrics
        await redis_client.incr("metrics:total_queries")
        await redis_client.sadd("metrics:sessions", session_id)  # unique sessions
        await redis_client.incrbyfloat("metrics:total_latency_ms", response_time_ms)  # running sum
        await redis_client.set("metrics:last_latency_ms", response_time_ms)            # most recent

        # Persist a durable PostgreSQL log with seven-day retention.
        await chat_log_store.log_chat(
            session_id=session_id,
            question=question,
            answer=result["answer"],
            sources=sources,
            response_time_ms=response_time_ms,
        )
        
        response = {
            'success': True,
            'question': question,
            'answer': result['answer'],
            'sources': sources,
            'timestamp': datetime.now().isoformat(),
            'response_time_ms': response_time_ms,
            'session_id': session_id,
        }
        return response

    except Exception as e:
        logger.exception("Error processing query")
        return JSONResponse(status_code=500, content={'success': False, 'error': str(e)})

@app.post("/api/index-data")
async def index_data(
    _: None = Depends(require_admin_token),
    config: RAGConfig = Depends(get_config),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Trigger data ingestion from the frontend. Requires admin auth."""
    try:
        # Copy config to avoid mutating the shared singleton
        local_data_config = copy.deepcopy(config.data)
        local_data_config.max_samples = 100
        
        local_config = copy.deepcopy(config)
        local_config.data = local_data_config
        
        loader = create_data_loader_from_config(local_config)
        contexts = loader.get_contexts()
        
        # Run blocking Pinecone upsert off the event loop
        await asyncio.to_thread(vector_store.add_documents, contexts, True)
        
        return {"success": True, "indexed_count": len(contexts)}
    except Exception as e:
        logger.exception("Error indexing data")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(
    response: Response,
    request: Request,
    session_id: str = Depends(get_session_id),
    requested_session_id: Optional[str] = FastAPIQuery(None, alias="session_id"),
    redis_client: Redis = Depends(get_redis_client),
):
    """Get chat history for the current session, or a known saved session."""
    active_session_id = requested_session_id or session_id
    history_key = f"chat_history:{active_session_id}"
    existing_history = await redis_client.get(history_key)
    history = json.loads(existing_history) if existing_history else []
    
    return {
        'success': True,
        'session_id': active_session_id,
        'history': history
    }


@app.post("/api/new-chat")
async def new_chat(request: Request, response: Response):
    """Start a fresh chat session without deleting prior session history."""
    old_session_id = request.cookies.get(SESSION_COOKIE)
    new_session_id = uuid.uuid4().hex
    response.set_cookie(
        SESSION_COOKIE,
        new_session_id,
        httponly=True,
        samesite="lax",
        secure=os.getenv("COOKIE_SECURE", "").lower() == "true",
        max_age=60 * 60 * 24 * 30,
    )

    return {
        'success': True,
        'old_session_id': old_session_id,
        'session_id': new_session_id,
    }


@app.post("/api/switch-chat")
async def switch_chat(req: SwitchChatRequest, response: Response):
    """Make a saved session active again for follow-up messages."""
    session_id = req.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id cannot be empty")

    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        httponly=True,
        samesite="lax",
        secure=os.getenv("COOKIE_SECURE", "").lower() == "true",
        max_age=60 * 60 * 24 * 30,
    )

    return {
        'success': True,
        'session_id': session_id,
    }

@app.post("/api/clear")
async def clear_history(
    session_id: str = Depends(get_session_id),
    redis_client: Redis = Depends(get_redis_client),
):
    """Clear chat history for current session"""
    history_key = f"chat_history:{session_id}"
    await redis_client.delete(history_key)
    
    return {
        'success': True,
        'message': 'Chat history cleared'
    }

@app.get("/api/chat-logs")
async def get_chat_logs(
    _: None = Depends(require_admin_token),
    chat_log_store: PostgresChatLogStore = Depends(get_chat_log_store),
    limit: int = FastAPIQuery(100, ge=1, le=500),
    session_id: Optional[str] = None,
):
    """Get recent PostgreSQL chat logs from the retention window."""
    logs = await chat_log_store.fetch_recent_logs(limit=limit, session_id=session_id)
    return {
        'success': True,
        'retention_days': chat_log_store.retention_days,
        'logs': logs,
    }


@app.get("/api/session-chat-logs")
async def get_session_chat_logs(
    session_id: str = Depends(get_session_id),
    chat_log_store: PostgresChatLogStore = Depends(get_chat_log_store),
    limit: int = FastAPIQuery(25, ge=1, le=100),
):
    """Get durable PostgreSQL logs for the active browser session."""
    logs = await chat_log_store.fetch_recent_logs(limit=limit, session_id=session_id)
    return {
        'success': True,
        'session_id': session_id,
        'retention_days': chat_log_store.retention_days,
        'logs': logs,
    }

@app.get("/api/stats")
async def get_stats(
    request: Request,
    _: None = Depends(require_admin_token),
    config: RAGConfig = Depends(get_config),
    vector_store: VectorStore = Depends(get_vector_store),
    redis_client: Redis = Depends(get_redis_client),
):
    """Get system statistics"""
    try:
        stats = vector_store.get_stats()
        total_queries = int(await redis_client.get("metrics:total_queries") or "0")
        total_sessions = await redis_client.scard("metrics:sessions") or 0
        total_latency_ms = float(await redis_client.get("metrics:total_latency_ms") or "0")
        last_latency_ms = float(await redis_client.get("metrics:last_latency_ms") or "0")
        avg_latency_ms = round(total_latency_ms / total_queries, 2) if total_queries else 0

        return {
            'success': True,
            'stats': {
                'total_documents': stats['total_documents'],
                'embedding_model': stats['embedding_model'],
                'llm_model': config.llm.model,
                'collection_name': stats['collection_name'],
                'dataset_name': config.data.dataset_name,
                'total_queries': total_queries,
                'total_sessions': total_sessions,
                'avg_response_time_ms': avg_latency_ms,
                'last_response_time_ms': round(last_latency_ms, 2),
                'uptime_minutes': round((time.time() - request.app.state.server_start_time) / 60, 1),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint — verifies Pinecone, Redis, and PostgreSQL connectivity."""
    health_status = {'status': 'healthy', 'services': {}}
    try:
        stats = app.state.vector_store.get_stats()
        health_status['documents_indexed'] = stats['total_documents']
        health_status['services']['pinecone'] = 'ok'
    except Exception as e:
        health_status['services']['pinecone'] = f'error: {e}'
        health_status['status'] = 'degraded'

    try:
        await app.state.redis_client.ping()
        health_status['services']['redis'] = 'ok'
    except Exception as e:
        health_status['services']['redis'] = f'error: {e}'
        health_status['status'] = 'degraded'

    try:
        async with app.state.chat_log_store.pool.acquire() as conn:
            await conn.execute('SELECT 1')
        health_status['services']['postgres'] = 'ok'
    except Exception as e:
        health_status['services']['postgres'] = f'error: {e}'
        health_status['status'] = 'degraded'

    status_code = 200 if health_status['status'] == 'healthy' else 503
    return JSONResponse(status_code=status_code, content=health_status)

if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 5000))
    print(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
