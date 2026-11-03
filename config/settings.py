"""
config/settings.py
~~~~~~~~~~~~~~~~~~
Central configuration for the Customer Complaint Root-Cause Classifier project.

All directory paths, model parameters, and environment-driven settings are
defined here so that every other module can simply ``from config.settings
import *`` (or import individual names) to stay in sync.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# 1.  Load environment variables from .env (at module import time)
# ---------------------------------------------------------------------------
# Resolve .env relative to the project root (one level up from this file).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)

# ---------------------------------------------------------------------------
# 2.  Directory layout
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
VECTOR_DB_DIR: Path = PROJECT_ROOT / "vector_db"

# ---------------------------------------------------------------------------
# 3.  Key file paths
# ---------------------------------------------------------------------------
RAW_CSV_PATH: Path = RAW_DATA_DIR / "customer_complaints.csv"
CLEANED_CSV_PATH: Path = PROCESSED_DATA_DIR / "cleaned_complaints.csv"

# ---------------------------------------------------------------------------
# 4.  Experiment / training hyper-parameters
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
TFIDF_MAX_FEATURES: int = 20_000
BATCH_SIZE: int = 512  # batch size for embedding generation

# ---------------------------------------------------------------------------
# 5.  Model names & API configurations (loaded from .env with sensible defaults)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL_NAME: str = os.getenv(
    "LLM_MODEL", "gemini-2.0-flash"
)
GEMINI_API_KEY: str = (
    os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
).strip()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
