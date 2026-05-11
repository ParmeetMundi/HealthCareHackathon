"""
Database — PostgreSQL connection pool and schema initialisation.

Provides a module-level connection pool and an init_db() function that
creates the required tables and extensions (pgvector) on first startup.

The DATABASE_URL is read from the environment (.env file).
"""
import logging
import os

from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

_pool: pool.ThreadedConnectionPool | None = None


def get_pool() -> pool.ThreadedConnectionPool:
    """Return (and lazily create) the module-level connection pool."""
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set in the environment.")
        _pool = pool.ThreadedConnectionPool(minconn=2, maxconn=10, dsn=DATABASE_URL)
        logger.info("postgres_pool_created dsn=%s", DATABASE_URL.split("@")[-1])
    return _pool


def get_conn():
    """Get a connection from the pool."""
    return get_pool().getconn()


def put_conn(conn):
    """Return a connection to the pool."""
    get_pool().putconn(conn)


def init_db() -> None:
    """
    Create pgvector extension and all required tables if they don't exist.

    Safe to call multiple times — uses IF NOT EXISTS.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # RAG documents table — stores chunked embeddings
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id            TEXT PRIMARY KEY,
                    collection    TEXT NOT NULL,
                    document      TEXT NOT NULL,
                    metadata      JSONB DEFAULT '{}',
                    embedding     vector(384),
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_collection
                ON rag_documents (collection);
            """)

            # Task result cache — replaces in-memory TaskResultCache
            cur.execute("""
                CREATE TABLE IF NOT EXISTS task_cache (
                    task_id       TEXT PRIMARY KEY,
                    result_text   TEXT NOT NULL,
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Patient chat history — stores conversation turns per patient
            cur.execute("""
                CREATE TABLE IF NOT EXISTS patient_chat_history (
                    id            SERIAL PRIMARY KEY,
                    patient_id    TEXT NOT NULL,
                    role          TEXT NOT NULL,
                    message       TEXT NOT NULL,
                    metadata      JSONB DEFAULT '{}',
                    created_at    TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_patient_id
                ON patient_chat_history (patient_id, created_at DESC);
            """)

        conn.commit()
        logger.info("database_schema_initialised tables=rag_documents,task_cache,patient_chat_history")
    except Exception:
        conn.rollback()
        raise
    finally:
        put_conn(conn)
