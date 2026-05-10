"""
Task result cache — stores completed task results for a configurable TTL.

When the Prompt Opinion platform sends a follow-up message referencing a
previous task ID, the a2a-sdk rejects it because the task is in a terminal
state ("completed").  This cache allows the middleware to:

1. Store the result text of every completed task.
2. On incoming messages that reference a previous task ID, retrieve the
   cached result, strip the taskId (so the SDK treats it as a new task),
   and inject the previous result as context into the question.

The cache uses an in-memory dict with timestamps. Expired entries are
lazily evicted on each access. For production deployments with multiple
workers, replace with Redis or another shared store.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# How long (in seconds) to keep task results. Default: 30 minutes.
_TTL_SECONDS = int(os.getenv("TASK_CACHE_TTL_SECONDS", "1800"))

# Maximum number of cached entries to prevent unbounded memory growth.
_MAX_ENTRIES = int(os.getenv("TASK_CACHE_MAX_ENTRIES", "500"))


class TaskResultCache:
    """Thread-safe in-memory cache for completed task results."""

    def __init__(self, ttl_seconds: int = _TTL_SECONDS, max_entries: int = _MAX_ENTRIES):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._lock = threading.Lock()
        # {task_id: (result_text, timestamp)}
        self._store: dict[str, tuple[str, float]] = {}

    def put(self, task_id: str, result_text: str) -> None:
        """Store a task result with the current timestamp."""
        if not task_id or not result_text:
            return
        with self._lock:
            self._evict_expired()
            # If at capacity, evict the oldest entry
            if len(self._store) >= self._max_entries:
                oldest_key = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest_key]
                logger.info("task_cache_evicted_oldest task_id=%s", oldest_key)
            self._store[task_id] = (result_text, time.time())
            logger.info(
                "task_cache_stored task_id=%s result_len=%d ttl=%ds cache_size=%d",
                task_id, len(result_text), self._ttl, len(self._store),
            )

    def get(self, task_id: str) -> str | None:
        """Retrieve a cached result, or None if missing/expired."""
        if not task_id:
            return None
        with self._lock:
            entry = self._store.get(task_id)
            if entry is None:
                return None
            result_text, ts = entry
            if time.time() - ts > self._ttl:
                del self._store[task_id]
                logger.info("task_cache_expired task_id=%s", task_id)
                return None
            logger.info(
                "task_cache_hit task_id=%s result_len=%d age_seconds=%d",
                task_id, len(result_text), int(time.time() - ts),
            )
            return result_text

    def _evict_expired(self) -> None:
        """Remove all expired entries. Must be called under lock."""
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]
        if expired:
            logger.info("task_cache_evicted_expired count=%d", len(expired))

    def size(self) -> int:
        """Return the current number of cached entries."""
        with self._lock:
            return len(self._store)


# Module-level singleton
task_cache = TaskResultCache()
