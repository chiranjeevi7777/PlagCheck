"""
Structured logging configuration for PlagCheck AI.

Import `get_logger` and call `get_logger(__name__)` in every module.
The root logger is configured once at import time.
"""

from __future__ import annotations

import logging
import sys


def _configure_root_logger() -> None:
    """Configure the root logger exactly once."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger under the 'PlagCheck' namespace."""
    return logging.getLogger(f"PlagCheck.{name}")


# Convenience alias used by legacy code via `from app.core.logging import logger`
logger = get_logger("core")
