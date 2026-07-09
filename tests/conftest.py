import pytest


ISOLATED_ENV_VARS = (
    "ADMIN_API_TOKEN",
    "API_AUTH_TOKEN",
    "CHAT_LOG_RETENTION_DAYS",
    "DATABASE_URL",
    "GOOGLE_API_KEY",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "OPENROUTER_API_KEY",
    "OPENROUTER_APP_NAME",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_SITE_URL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "POSTGRES_DSN",
    "UPSTASH_REDIS_REST_TOKEN",
    "UPSTASH_REDIS_REST_URL",
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    """Keep unit tests independent from local .env files and shell state."""
    for name in ISOLATED_ENV_VARS:
        monkeypatch.setenv(name, "")


@pytest.fixture(autouse=True)
def block_unexpected_requests(monkeypatch):
    """Fail fast if a unit test accidentally reaches the network via requests."""
    import requests

    def blocked_request(self, method, url, **kwargs):
        raise AssertionError(f"Unexpected outbound HTTP request during tests: {method} {url}")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked_request)
