"""
RAG store — PostgreSQL + pgvector backend with sentence-transformers embeddings.

Replaces ChromaDB with PostgreSQL for vector storage. Uses pgvector for
similarity search and sentence-transformers for embedding generation.
Database connection is managed via shared.db.
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Lazy-loaded embedding model ────────────────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("sentence_transformer_loaded model=all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = _get_model()
    embeddings = model.encode(texts)
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return embed_texts([text])[0]


# ── Database operations ────────────────────────────────────────────────────────

def upsert_documents(
    collection: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> int:
    """
    Embed and upsert documents into the rag_documents table.

    Returns the number of rows upserted.
    """
    from shared.db import get_conn, put_conn

    embeddings = embed_texts(documents)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for doc_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
                cur.execute(
                    """
                    INSERT INTO rag_documents (id, collection, document, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE
                        SET document  = EXCLUDED.document,
                            metadata  = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding;
                    """,
                    (doc_id, collection, doc, json.dumps(meta), str(emb)),
                )
        conn.commit()
        logger.info("rag_upserted collection=%s count=%d", collection, len(ids))
        return len(ids)
    except Exception:
        conn.rollback()
        raise
    finally:
        put_conn(conn)


def query_documents(
    collection: str,
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Embed the query and return the top-N most similar documents from the
    given collection using pgvector cosine distance.

    Returns a list of dicts: [{document, metadata, distance}, ...]
    """
    from shared.db import get_conn, put_conn

    query_embedding = embed_text(query)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document, metadata, (embedding <=> %s::vector) AS distance
                FROM rag_documents
                WHERE collection = %s
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (str(query_embedding), collection, n_results),
            )
            rows = cur.fetchall()
        results = []
        for doc, meta, dist in rows:
            results.append({
                "document": doc,
                "metadata": meta if isinstance(meta, dict) else json.loads(meta),
                "distance": float(dist),
            })
        logger.info("rag_queried collection=%s results=%d", collection, len(results))
        return results
    finally:
        put_conn(conn)


def collection_count(collection: str) -> int:
    """Return the number of documents in a collection."""
    from shared.db import get_conn, put_conn

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM rag_documents WHERE collection = %s;",
                (collection,),
            )
            return cur.fetchone()[0]
    finally:
        put_conn(conn)
