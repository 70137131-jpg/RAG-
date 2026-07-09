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
