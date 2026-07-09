from unittest.mock import Mock

from fastapi.testclient import TestClient


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.sets = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def ping(self):
        return True

    async def delete(self, key):
        self.values.pop(key, None)

    async def incr(self, key):
        self.values[key] = str(int(self.values.get(key, "0")) + 1)

    async def incrbyfloat(self, key, value):
        self.values[key] = str(float(self.values.get(key, "0")) + float(value))

    async def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    async def scard(self, key):
        return len(self.sets.get(key, set()))


class FakeChatLogStore:
    retention_days = 7

    def __init__(self):
        self.logs = []
        self.connected = False
        self.initialized = False
        self.closed = False
        self.purge_count = 0
        self.pool = FakePool()

    async def connect(self):
        self.connected = True

    async def initialize(self):
        self.initialized = True

    async def close(self):
        self.closed = True

    async def purge_old_logs(self):
        self.purge_count += 1
        return 0

    async def log_chat(self, **kwargs):
        self.logs.append(kwargs)

    async def fetch_recent_logs(self, limit=100, session_id=None):
        logs = self.logs
        if session_id is not None:
            logs = [log for log in logs if log["session_id"] == session_id]
        return logs[:limit]


class FakeConnection:
    async def execute(self, query):
        return "SELECT 1"


class FakePoolAcquire:
    async def __aenter__(self):
        return FakeConnection()

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakePool:
    def acquire(self):
        return FakePoolAcquire()


def make_client(monkeypatch):
    import app as app_module
    from config import RAGConfig

    config = RAGConfig.from_env()
    vector_store = Mock()
    vector_store.get_stats.return_value = {
        "total_documents": 10,
        "embedding_model": "multilingual-e5-large",
        "collection_name": "test-index",
    }

    rag_pipeline = Mock()
    rag_pipeline.query_async = Mock()

    async def query_async(question, top_k=3):
        return {
            "answer": "Test answer [Context 1]",
            "metadata": {
                "retrieved_docs": [
                    {"id": "doc1", "text": "Context 1", "distance": 0.1},
                    {"id": "doc2", "text": "Context 2", "distance": 0.2},
                ]
            },
        }

    rag_pipeline.query_async.side_effect = query_async
    redis = FakeRedis()
    chat_log_store = FakeChatLogStore()

    def fake_create_services(_config):
        return vector_store, rag_pipeline, redis, chat_log_store

    monkeypatch.setattr(app_module, "create_services", fake_create_services)
    monkeypatch.setenv("ADMIN_API_TOKEN", "test-admin-token")
    return TestClient(app_module.app), rag_pipeline, redis, chat_log_store


def test_index_sets_session_cookie(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "rag_session_id" in response.cookies


def test_query_endpoint_uses_session_cookie_and_returns_sources(monkeypatch):
    client, _, _, chat_log_store = make_client(monkeypatch)
    with client:
        response = client.post("/api/query", json={"question": "What is AI?", "top_k": 2})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["answer"] == "Test answer [Context 1]"
    assert data["sources"][0]["similarity"] == 90.0
    assert chat_log_store.logs[0]["question"] == "What is AI?"
    assert chat_log_store.logs[0]["answer"] == "Test answer [Context 1]"


def test_query_empty_question(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        response = client.post("/api/query", json={"question": ""})

    assert response.status_code == 422


def test_history_and_clear_are_session_scoped(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        client.post("/api/query", json={"question": "First question"})
        history_response = client.get("/api/history")
        clear_response = client.post("/api/clear")
        empty_history_response = client.get("/api/history")

    assert history_response.status_code == 200
    history = history_response.json()["history"]
    assert len(history) == 1
    # Restored conversations must keep their retrieved contexts (RAG sources)
    assert history[0]["sources"][0]["id"] == "doc1"
    assert history[0]["sources"][0]["similarity"] == 90.0
    assert clear_response.json()["success"] is True
    assert empty_history_response.json()["history"] == []


def test_new_chat_preserves_old_session_history(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        client.post("/api/query", json={"question": "First question"})
        old_session_id = client.cookies.get("rag_session_id")

        new_response = client.post("/api/new-chat")
        new_session_id = new_response.json()["session_id"]
        new_history_response = client.get("/api/history")
        old_history_response = client.get(f"/api/history?session_id={old_session_id}")

    assert new_response.status_code == 200
    assert new_session_id != old_session_id
    assert new_history_response.json()["history"] == []
    assert old_history_response.json()["history"][0]["question"] == "First question"


def test_switch_chat_makes_saved_session_active(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        client.post("/api/query", json={"question": "First question"})
        old_session_id = client.cookies.get("rag_session_id")
        client.post("/api/new-chat")

        switch_response = client.post(
            "/api/switch-chat",
            json={"session_id": old_session_id},
        )
        history_response = client.get("/api/history")

    assert switch_response.status_code == 200
    assert history_response.json()["session_id"] == old_session_id
    assert history_response.json()["history"][0]["question"] == "First question"


def test_switch_chat_rejects_malformed_session_id(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        response = client.post(
            "/api/switch-chat",
            json={"session_id": "not-a-valid-session-id"},
        )

    assert response.status_code == 400
    assert "Invalid session_id" in response.json()["detail"]


def test_history_rejects_malformed_session_id_query_param(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        response = client.get("/api/history?session_id=<script>alert(1)</script>")

    assert response.status_code == 400
    assert "Invalid session_id" in response.json()["detail"]


def test_chat_logs_endpoint_requires_admin_token(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        response = client.get("/api/chat-logs")

    assert response.status_code == 401


def test_chat_logs_endpoint_returns_postgres_logs(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        client.post("/api/query", json={"question": "Logged question"})
        response = client.get(
            "/api/chat-logs",
            headers={"Authorization": "Bearer test-admin-token"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["retention_days"] == 7
    assert data["logs"][0]["question"] == "Logged question"


def test_session_chat_logs_endpoint_returns_current_session_logs(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        client.post("/api/query", json={"question": "Current session log"})
        current_session_id = client.cookies.get("rag_session_id")
        client.post("/api/new-chat")
        client.post("/api/query", json={"question": "New session log"})

        current_logs_response = client.get("/api/session-chat-logs")
        old_logs_response = client.get(
            "/api/chat-logs",
            params={"session_id": current_session_id},
            headers={"Authorization": "Bearer test-admin-token"},
        )

    assert current_logs_response.status_code == 200
    assert current_logs_response.json()["logs"][0]["question"] == "New session log"
    assert old_logs_response.json()["logs"][0]["question"] == "Current session log"


def test_stats_requires_admin_token(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        client.post("/api/query", json={"question": "Stats question"})
        response = client.get("/api/stats")

    assert response.status_code == 401


def test_stats_requires_configured_admin_token(monkeypatch):
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    with make_client(monkeypatch)[0] as client:
        monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
        monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
        response = client.get(
            "/api/stats",
            headers={"Authorization": "Bearer test-admin-token"},
        )

    assert response.status_code == 503


def test_stats_endpoint_with_admin_token(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        client.post("/api/query", json={"question": "Stats question"})
        response = client.get(
            "/api/stats",
            headers={"Authorization": "Bearer test-admin-token"},
        )

    assert response.status_code == 200
    stats = response.json()["stats"]
    assert stats["total_documents"] == 10
    assert stats["total_queries"] == 1
    assert stats["total_sessions"] == 1


def test_health_endpoint(monkeypatch):
    with make_client(monkeypatch)[0] as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "services": {
            "pinecone": "ok",
            "redis": "ok",
            "postgres": "ok",
        },
        "documents_indexed": 10,
    }


def test_startup_validation_reports_missing_services():
    from app import validate_required_services
    from config import RAGConfig

    config = RAGConfig.from_env()
    config.vector_store.api_key = ""
    config.vector_store.index_name = "test-index"
    config.llm.provider = "gemini"
    config.llm.model = "gemini-1.5-flash"
    config.llm.api_key = ""
    config.redis.url = "https://redis.example"
    config.redis.token = "redis-token"
    config.postgres.dsn = ""

    try:
        validate_required_services(config)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("validate_required_services should fail")

    assert "PINECONE_API_KEY" in message
    assert "GOOGLE_API_KEY" in message
    assert "DATABASE_URL or POSTGRES_DSN" in message


def test_startup_validation_accepts_openrouter_config():
    from app import validate_required_services
    from config import RAGConfig

    config = RAGConfig.from_env()
    config.vector_store.api_key = "pinecone-key"
    config.vector_store.index_name = "test-index"
    config.llm.provider = "openrouter"
    config.llm.model = "openai/gpt-oss-20b:free"
    config.llm.api_key = ""
    config.llm.openrouter_api_key = "openrouter-key"
    config.redis.url = "https://redis.example"
    config.redis.token = "redis-token"
    config.postgres.dsn = "postgresql://user:pass@localhost/db"

    validate_required_services(config)
