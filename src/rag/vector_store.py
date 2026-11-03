"""
ChromaDB Vector Store Manager Module

Manages persistent ChromaDB vector storage for complaint documents using
the local SentenceTransformer embedding model (all-MiniLM-L6-v2).
"""

import sys
from pathlib import Path

# Resolve project root and ensure it is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from config.settings import VECTOR_DB_DIR, MODELS_DIR, EMBEDDING_MODEL_NAME
from src.clustering.generate_embeddings import load_embedding_model
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir

logger = get_logger(__name__)

_cached_chroma_client = None


# ---------------------------------------------------------------------------
# Custom Embedding Function for ChromaDB using Local SentenceTransformer
# ---------------------------------------------------------------------------

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

class LocalSentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible embedding function wrapping our local SentenceTransformer."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = load_embedding_model(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # Encode inputs and convert to standard Python float lists
        embeddings = self.model.encode(
            input,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(float).tolist()

    @staticmethod
    def name() -> str:
        return "all-MiniLM-L6-v2-local"


# ---------------------------------------------------------------------------
# ChromaDB Client & Collection Management
# ---------------------------------------------------------------------------

def get_chroma_client(db_dir: Path = VECTOR_DB_DIR):
    """Get or create persistent ChromaDB client."""
    global _cached_chroma_client
    if _cached_chroma_client is not None:
        return _cached_chroma_client

    import chromadb
    from chromadb.config import Settings

    ensure_dir(db_dir)
    logger.info("Initializing persistent ChromaDB client at: %s", db_dir)
    _cached_chroma_client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )
    return _cached_chroma_client


def get_or_create_collection(
    collection_name: str = "customer_complaints_v1",
    db_dir: Path = VECTOR_DB_DIR,
):
    """Retrieve or create ChromaDB collection with local embedding function."""
    client = get_chroma_client(db_dir)
    embedding_fn = LocalSentenceTransformerEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_documents_in_batches(
    collection,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    embeddings: Optional[np.ndarray] = None,
    batch_size: int = 500,
):
    """Add documents to ChromaDB in memory-bounded batches (default: 500)."""
    total = len(documents)
    logger.info("Indexing %d documents into ChromaDB collection '%s'...", total, collection.name)

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        b_docs = documents[i:end]
        b_meta = metadatas[i:end]
        b_ids = ids[i:end]

        kwargs: Dict[str, Any] = {
            "documents": b_docs,
            "metadatas": b_meta,
            "ids": b_ids,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings[i:end].astype(float).tolist()

        collection.upsert(**kwargs)
        logger.info("Indexed batch %d-%d / %d documents", i, end, total)

    logger.info("ChromaDB collection now contains %d records.", collection.count())


def search_vector_store(
    query_text: str,
    n_results: int = 5,
    where_filter: Optional[Dict[str, Any]] = None,
    collection_name: str = "customer_complaints_v1",
) -> List[Dict[str, Any]]:
    """Execute cosine similarity search over complaint documents.

    Parameters
    ----------
    query_text : str
        User's natural language question.
    n_results : int
        Number of top documents to retrieve.
    where_filter : dict, optional
        Metadata filter (e.g. {"company": "Cedar Ridge Bank"}).

    Returns
    -------
    List[Dict[str, Any]]
        List of result items with id, document, metadata, and distance.
    """
    collection = get_or_create_collection(collection_name)
    kwargs: Dict[str, Any] = {
        "query_texts": [query_text],
        "n_results": n_results,
    }
    if where_filter:
        kwargs["where"] = where_filter

    results = collection.query(**kwargs)

    parsed_results = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        ids = results["ids"][0]
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
        dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
            parsed_results.append({
                "complaint_id": doc_id,
                "document": doc_text,
                "metadata": meta,
                "distance": float(dist),
                "similarity_score": round(float(1.0 - dist), 3),
            })

    return parsed_results
