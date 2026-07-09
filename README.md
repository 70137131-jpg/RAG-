# RAG_SQUAD

A highly scalable, serverless, and production-ready Retrieval-Augmented Generation (RAG) chatbot using FastAPI, Pinecone, Upstash Redis, and Gemini.

## PostgreSQL chat logs

The app stores every chat request in PostgreSQL and automatically keeps only the last 7 days of logs.

Required environment variables:

```env
DATABASE_URL=postgresql://rag:rag_password@localhost:5432/rag_chat_logs
CHAT_LOG_RETENTION_DAYS=7
ADMIN_API_TOKEN=change-this-token
```

To use OpenRouter instead of Gemini:

```env
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-oss-20b:free
OPENROUTER_API_KEY=your-new-openrouter-key
```

Do not commit real API keys. If a key was shared publicly, revoke it and create a new one.

With Docker Compose, PostgreSQL is started automatically and the web container receives:

```env
DATABASE_URL=postgresql://rag:rag_password@postgres:5432/rag_chat_logs
```

Start the stack:

```bash
docker compose up --build
```

Read recent logs with an admin token:

```bash
curl -H "Authorization: Bearer $ADMIN_API_TOKEN" "http://localhost:8000/api/chat-logs?limit=100"
```

## Deploy on Render

This repo includes a `render.yaml` Blueprint for a Render Web Service.

1. Push the repo to GitHub.
2. In Render, choose **New > Blueprint** and connect this repository.
3. Render will use:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port $PORT
```

4. Add the required secret environment variables in Render:

```env
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
DATABASE_URL=
ADMIN_API_TOKEN=
LLM_PROVIDER=gemini
LLM_MODEL=
GOOGLE_API_KEY=
```

For OpenRouter instead of Gemini, set:

```env
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-oss-20b:free
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_NAME=RAG Pipeline
```

The deployed app health check is available at `/health`.
