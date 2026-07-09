"""
PostgreSQL-backed chat log storage.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    import asyncpg
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    asyncpg = None


class PostgresChatLogStore:
    """Durable chat logging with a retention window."""

    def __init__(self, dsn: str, retention_days: int = 7):
        self.dsn = dsn
        self.retention_days = retention_days
        self.pool = None

    async def connect(self) -> None:
        if asyncpg is None:
            raise ImportError("asyncpg is required for PostgreSQL chat logging.")
        self.pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=5)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()

    async def initialize(self) -> None:
        await self._execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                response_time_ms DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at
                ON chat_logs (created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_chat_logs_session_created_at
                ON chat_logs (session_id, created_at DESC);
            """
        )

    async def log_chat(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        response_time_ms: float,
    ) -> None:
        await self._execute(
            """
            INSERT INTO chat_logs (
                session_id,
                question,
                answer,
                sources,
                response_time_ms
            )
            VALUES ($1, $2, $3, $4::jsonb, $5)
            """,
            session_id,
            question,
            answer,
            json.dumps(sources),
            response_time_ms,
        )

    async def purge_old_logs(self) -> int:
        cutoff = self._cutoff()
        result = await self._execute(
            "DELETE FROM chat_logs WHERE created_at < $1",
            cutoff,
        )
        return int(result.rsplit(" ", 1)[-1])

    async def fetch_recent_logs(
        self,
        *,
        limit: int = 100,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cutoff = self._cutoff()
        if session_id:
            rows = await self._fetch(
                """
                SELECT id, session_id, question, answer, sources, response_time_ms, created_at
                FROM chat_logs
                WHERE created_at >= $1 AND session_id = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                cutoff,
                session_id,
                limit,
            )
        else:
            rows = await self._fetch(
                """
                SELECT id, session_id, question, answer, sources, response_time_ms, created_at
                FROM chat_logs
                WHERE created_at >= $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                cutoff,
                limit,
            )

        return [self._serialize_row(row) for row in rows]

    def _cutoff(self) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=self.retention_days)

    async def _execute(self, query: str, *args: Any) -> str:
        self._ensure_connected()
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def _fetch(self, query: str, *args: Any):
        self._ensure_connected()
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    def _ensure_connected(self) -> None:
        if self.pool is None:
            raise RuntimeError("PostgreSQL chat log store is not connected.")

    def _serialize_row(self, row) -> Dict[str, Any]:
        sources = row["sources"]
        if isinstance(sources, str):
            sources = json.loads(sources)
        created_at = row["created_at"]
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()

        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "question": row["question"],
            "answer": row["answer"],
            "sources": sources,
            "response_time_ms": row["response_time_ms"],
            "created_at": created_at,
        }
