"""
src/utils/logger.py
~~~~~~~~~~~~~~~~~~~
Reusable logging setup for the Customer Complaint Root-Cause Classifier.

Usage::

    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured :class:`logging.Logger`.

    * Format : ``%(asctime)s | %(name)s | %(levelname)s | %(message)s``
    * Date   : ``%Y-%m-%d %H:%M:%S``
    * Handler: :class:`logging.StreamHandler` (stdout)

    The function is safe to call multiple times with the same *name*;
    duplicate handlers are avoided by checking whether the logger already
    has handlers attached.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    level : int, optional
        Logging level, by default ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding duplicate handlers on repeated calls
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
