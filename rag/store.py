"""
RAG store — ChromaDB persistent client with sentence-transformers embeddings.

Manages a single persistent ChromaDB client. Collections are created or
retrieved on demand via get_collection(name).
"""
import os
import logging

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """Custom ChromaDB embedding function using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("sentence_transformer_loaded model=%s", self._model_name)

    def __call__(self, input: Documents) -> Embeddings:
        self._load_model()
        embeddings = self._model.encode(input)
        return embeddings.tolist()


_client: chromadb.ClientAPI | None = None
_embedding_fn: SentenceTransformerEmbeddingFunction | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_PERSIST_DIR)
        logger.info("chromadb_client_created persist_dir=%s", _PERSIST_DIR)
    return _client


def _get_embedding_fn() -> SentenceTransformerEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbeddingFunction()
    return _embedding_fn


def get_collection(name: str) -> chromadb.Collection:
    """Create or retrieve a named ChromaDB collection with sentence-transformer embeddings."""
    client = _get_client()
    ef = _get_embedding_fn()
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ef,
    )
    logger.info("chromadb_collection name=%s count=%d", name, collection.count())
    return collection
