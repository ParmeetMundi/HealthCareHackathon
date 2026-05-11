"""
Task result cache — stores completed task results in PostgreSQL.

When the Prompt Opinion platform sends a follow-up message referencing a
previous task ID, the a2a-sdk rejects it because the task is in a terminal
state ("completed").  This cache allows the middleware to:

1. Store the result text of every completed task.
2. On incoming messages that reference a previous task ID, retrieve the
   cached result, strip the taskId (so the SDK treats it as a new task),
   and inject the previous result as context into the question.

Uses PostgreSQL for persistence — survives restarts and works across
multiple workers.
"""

import logging
import os

logger = logging.getLogger(__name__)

# How long (in seconds) to keep task results. Default: 30 minutes.
_TTL_SECONDS = int(os.getenv("TASK_CACHE_TTL_SECONDS", "1800"))


class TaskResultCache:
    """PostgreSQL-backed cache for completed task results."""

    def __init__(self, ttl_seconds: int = _TTL_SECONDS):
        self._ttl = ttl_seconds

    def put(self, task_id: str, result_text: str) -> None:
        """Store a task result with the current timestamp."""
        if not task_id or not result_text:
            return
        from shared.db import get_conn, put_conn

        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO task_cache (task_id, result_text, created_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (task_id) DO UPDATE
                        SET result_text = EXCLUDED.result_text,
                            created_at  = NOW();
                    """,
                    (task_id, result_text),
                )
            conn.commit()
            logger.info(
                "task_cache_stored task_id=%s result_len=%d ttl=%ds",
                task_id, len(result_text), self._ttl,
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            put_conn(conn)

    def get(self, task_id: str) -> str | None:
        """Retrieve a cached result, or None if missing/expired."""
        if not task_id:
            return None
        from shared.db import get_conn, put_conn

        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT result_text,
                           EXTRACT(EPOCH FROM (NOW() - created_at)) AS age_seconds
                    FROM task_cache
                    WHERE task_id = %s;
                    """,
                    (task_id,),
                )
                row = cur.fetchone()

            if row is None:
                return None

            result_text, age_seconds = row
            if age_seconds > self._ttl:
                # Expired — delete it
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM task_cache WHERE task_id = %s;", (task_id,))
                conn.commit()
                logger.info("task_cache_expired task_id=%s age=%ds", task_id, int(age_seconds))
                return None

            logger.info(
                "task_cache_hit task_id=%s result_len=%d age_seconds=%d",
                task_id, len(result_text), int(age_seconds),
            )
            return result_text
        except Exception:
            conn.rollback()
            raise
        finally:
            put_conn(conn)

    def cleanup_expired(self) -> int:
        """Delete all expired entries. Returns number of rows deleted."""
        from shared.db import get_conn, put_conn

        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM task_cache WHERE EXTRACT(EPOCH FROM (NOW() - created_at)) > %s;",
                    (self._ttl,),
                )
                deleted = cur.rowcount
            conn.commit()
            if deleted:
                logger.info("task_cache_cleanup deleted=%d", deleted)
            return deleted
        except Exception:
            conn.rollback()
            raise
        finally:
            put_conn(conn)


# Module-level singleton
task_cache = TaskResultCache()
