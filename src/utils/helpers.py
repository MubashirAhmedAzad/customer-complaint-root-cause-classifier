"""
src/utils/helpers.py
~~~~~~~~~~~~~~~~~~~~
Common utility functions used across the Customer Complaint Root-Cause
Classifier project.
"""

import functools
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict

from dotenv import dotenv_values, load_dotenv

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1.  Timer decorator
# ---------------------------------------------------------------------------
def timer_decorator(func: Callable) -> Callable:
    """Decorator that logs the wall-clock execution time of *func*.

    Example::

        @timer_decorator
        def train_model():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(
            "%s executed in %.2f seconds (%.2f min)",
            func.__qualname__,
            elapsed,
            elapsed / 60,
        )
        return result

    return wrapper


# ---------------------------------------------------------------------------
# 2.  Memory usage
# ---------------------------------------------------------------------------
def get_memory_usage() -> float:
    """Return current process RSS memory usage in **MB**.

    Uses :mod:`psutil` when available.  If *psutil* is not installed, a
    warning is logged and ``0.0`` is returned.

    Returns
    -------
    float
        Memory usage in megabytes.
    """
    try:
        import psutil  # type: ignore[import-untyped]

        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return round(mem_mb, 2)
    except ImportError:
        logger.warning(
            "psutil is not installed – memory usage unavailable. "
            "Install it with: pip install psutil"
        )
        return 0.0


# ---------------------------------------------------------------------------
# 3.  Narrative hash (for GroupShuffleSplit duplicate grouping)
# ---------------------------------------------------------------------------
def generate_narrative_hash(text: str) -> str:
    """Return an MD5 hex-digest of *text* after stripping and lower-casing.

    This deterministic hash is used as a group key in
    :class:`~sklearn.model_selection.GroupShuffleSplit` so that duplicate
    or near-identical complaint narratives always land in the same fold.

    Parameters
    ----------
    text : str
        Raw complaint narrative.

    Returns
    -------
    str
        32-character hexadecimal MD5 digest.
    """
    normalised = str(text).strip().lower()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 4.  Directory helper
# ---------------------------------------------------------------------------
def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if it does not already exist.

    Parameters
    ----------
    path : Path or str
        Directory path to create.

    Returns
    -------
    Path
        The same path, guaranteed to exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 5.  Load .env configuration
# ---------------------------------------------------------------------------
def load_config() -> Dict[str, str | None]:
    """Load the ``.env`` file and return its contents as a dictionary.

    The function first calls :func:`~dotenv.load_dotenv` to inject the
    variables into :data:`os.environ`, then reads the file with
    :func:`~dotenv.dotenv_values` so that the caller receives a clean
    mapping of *only* the project-specific keys.

    Returns
    -------
    dict
        Key-value pairs defined in the ``.env`` file.
    """
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    config = dotenv_values(dotenv_path=env_path)
    return dict(config)
