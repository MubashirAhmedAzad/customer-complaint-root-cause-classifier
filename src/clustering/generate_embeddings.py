"""
Semantic Embeddings Generator Module

Generates sentence embeddings for customer complaint narratives using
sentence-transformers (all-MiniLM-L6-v2) in memory-efficient batches.

Saves embeddings as .npy files so downstream clustering, RAG vector stores,
and the Streamlit dashboard do not recompute embeddings on startup.
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
import torch
from typing import List, Optional, Tuple

from config.settings import (
    PROJECT_ROOT,
    MODELS_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_MODEL_NAME,
    BATCH_SIZE,
    RANDOM_SEED,
)
from src.data.load_data import load_complaints
from src.data.clean_data import clean_complaints
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, timer_decorator

logger = get_logger(__name__)

_cached_embedding_model = None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Load and cache the SentenceTransformer model."""
    global _cached_embedding_model
    if _cached_embedding_model is not None:
        return _cached_embedding_model

    from sentence_transformers import SentenceTransformer

    local_model_path = MODELS_DIR / "embedding_model"
    target_model = str(local_model_path) if local_model_path.exists() else model_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading embedding model from '%s' on device '%s'...", target_model, device)
    _cached_embedding_model = SentenceTransformer(target_model, device=device)
    logger.info("Embedding model loaded successfully.")
    return _cached_embedding_model


# ---------------------------------------------------------------------------
# Embeddings Generation
# ---------------------------------------------------------------------------

@timer_decorator
def generate_embeddings(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Generate dense embeddings for a list of texts in memory-bounded batches.

    Parameters
    ----------
    texts : list[str]
        List of complaint narratives.
    model_name : str
        Hugging Face / SentenceTransformer model identifier.
    batch_size : int
        Batch size (default: 512, tuned for 8 GB RAM).
    show_progress : bool
        Whether to display a progress bar.
    normalize : bool
        Whether to L2-normalize vectors (recommended for cosine similarity).

    Returns
    -------
    np.ndarray
        Array of shape (len(texts), embedding_dim), float32.
    """
    model = load_embedding_model(model_name)
    logger.info(
        "Encoding %d narratives (batch_size=%d, normalize=%s)...",
        len(texts),
        batch_size,
        normalize,
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    logger.info("Generated embeddings array with shape: %s", embeddings.shape)
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Deduplicated Batch Embedding Pipeline
# ---------------------------------------------------------------------------

def generate_and_save_embeddings(
    df: Optional[pd.DataFrame] = None,
    output_dir: Path = EMBEDDINGS_DIR,
    sample_size: Optional[int] = None,
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generate embeddings with unique-narrative deduplication to save compute.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Complaints DataFrame. If None, loaded and cleaned automatically.
    output_dir : Path
        Directory where complaint_embeddings.npy and metadata are saved.
    sample_size : int, optional
        Optional sample size for quick prototyping (e.g. 10000 or 25000).
    model_name : str
        Model identifier.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame]
        (embeddings array, metadata DataFrame aligned with rows)
    """
    if df is None:
        logger.info("Loading complaints dataset for embedding generation...")
        df = load_complaints()
        df = clean_complaints(df)

    if sample_size and sample_size < len(df):
        logger.info("Sampling %d records for embedding generation...", sample_size)
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

    ensure_dir(output_dir)

    # Deduplicate unique narratives to avoid redundant GPU/CPU passes
    unique_narratives = df["complaint_narrative"].drop_duplicates().tolist()
    logger.info(
        "Found %d unique narratives across %d complaints (saves %.1f%% compute)",
        len(unique_narratives),
        len(df),
        (1 - len(unique_narratives) / len(df)) * 100,
    )

    # Encode unique texts
    unique_embeddings = generate_embeddings(
        unique_narratives,
        model_name=model_name,
        batch_size=batch_size,
    )

    # Map embeddings back to all records
    text_to_idx = {text: idx for idx, text in enumerate(unique_narratives)}
    mapped_indices = [text_to_idx[text] for text in df["complaint_narrative"]]
    all_embeddings = unique_embeddings[mapped_indices]

    # Save to disk
    emb_file = output_dir / "complaint_embeddings.npy"
    meta_file = output_dir / "complaints_metadata.csv"

    np.save(emb_file, all_embeddings)
    df.to_csv(meta_file, index=False)

    logger.info("Saved embeddings to: %s (%.1f MB)", emb_file, all_embeddings.nbytes / 1e6)
    logger.info("Saved metadata to:   %s", meta_file)

    return all_embeddings, df


# ---------------------------------------------------------------------------
# Loading Cached Embeddings
# ---------------------------------------------------------------------------

def load_saved_embeddings(
    output_dir: Path = EMBEDDINGS_DIR,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load previously computed embeddings and matching metadata from disk."""
    emb_file = output_dir / "complaint_embeddings.npy"
    meta_file = output_dir / "complaints_metadata.csv"

    if not emb_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"Embeddings not found in {output_dir}. Run generate_embeddings.py first."
        )

    logger.info("Loading precomputed embeddings from %s...", emb_file)
    embeddings = np.load(emb_file)
    metadata_df = pd.read_csv(meta_file)
    logger.info("Loaded embeddings shape: %s", embeddings.shape)
    return embeddings, metadata_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate embeddings for a high-quality 20,000 representative sample
    # or the full dataset depending on runtime preferences
    sample = 20000
    print(f"Generating embeddings for {sample:,} complaints...")
    embeddings, meta_df = generate_and_save_embeddings(sample_size=sample)
    print(f"[SUCCESS] Embeddings generated! Shape: {embeddings.shape}")

