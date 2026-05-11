"""
RAG tools — store and retrieve documents from PostgreSQL + pgvector.

These tools are added to every CrewAI agent's tools list so all agents
have access to long-term memory via their dedicated collections.
"""
import json
import logging
import uuid

from crewai.tools import tool

from rag.store import upsert_documents, query_documents, collection_count

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into token-approximate chunks with overlap."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


@tool("Store information in long-term memory")
def rag_store(collection_name: str, document: str, metadata: str) -> str:
    """Chunks document into 512-token segments with 50-token overlap,
    generates embeddings, upserts into the named PostgreSQL collection.
    metadata must be a JSON string with key-value pairs.
    Returns confirmation message.

    Args:
        collection_name: Name of the collection to store in.
        document: The text content to store.
        metadata: A JSON string of metadata key-value pairs.
    """
    try:
        meta = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        meta = {"raw_metadata": metadata}

    # Ensure all metadata values are str, int, float, or bool
    clean_meta = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean_meta[k] = v
        else:
            clean_meta[k] = str(v)

    chunks = _chunk_text(document)

    ids = []
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{uuid.uuid4().hex[:12]}_{i}"
        chunk_meta = {**clean_meta, "chunk_index": i, "total_chunks": len(chunks)}
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(chunk_meta)

    count = upsert_documents(collection_name, ids, documents, metadatas)
    logger.info(
        "rag_stored collection=%s chunks=%d",
        collection_name, count,
    )
    return f"Successfully stored {count} chunk(s) in collection '{collection_name}'."


@tool("Retrieve relevant information from long-term memory")
def rag_retrieve(collection_name: str, query: str, n_results: int = 5) -> str:
    """Embeds query and returns top n_results most similar chunks from
    the named PostgreSQL collection as a formatted numbered list.

    Args:
        collection_name: Name of the collection to search.
        query: The search query text.
        n_results: Number of results to return (default 5).
    """
    total = collection_count(collection_name)
    if total == 0:
        return f"No documents found in collection '{collection_name}'."

    actual_n = min(n_results, total)
    results = query_documents(collection_name, query, actual_n)

    if not results:
        return f"No relevant results found in collection '{collection_name}' for query: {query}"

    lines = []
    for i, row in enumerate(results, 1):
        similarity = max(0.0, 1.0 - row["distance"])
        lines.append(f"{i}. [similarity={similarity:.3f}] {row['document']}")

    logger.info(
        "rag_retrieved collection=%s query_len=%d results=%d",
        collection_name, len(query), len(results),
    )
    return "\n".join(lines)
