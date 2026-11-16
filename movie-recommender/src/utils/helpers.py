"""Small shared utilities: logging setup, config loading, fuzzy title matching, timing."""

from __future__ import annotations

import functools
import logging
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML pipeline config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured module-level logger (idempotent — safe to call repeatedly)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def timeit(func):
    """Decorator that logs how long a pipeline stage took."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} finished in {elapsed:.3f}s")
        return result

    return wrapper


def fuzzy_find_title(query: str, all_titles: list[str], threshold: float = 0.6) -> str | None:
    """
    Resolve a user-typed title to the closest real title in the catalog.
    Handles typos / case differences (e.g. "avatr" -> "Avatar").
    Returns None if nothing is close enough.
    """
    if query in all_titles:
        return query
    matches = get_close_matches(query, all_titles, n=1, cutoff=threshold)
    return matches[0] if matches else None
