"""Logging configuration shared by the CLI and library entry points."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"
_DATEFMT = "%H:%M:%S"


def configure(level: str = "INFO") -> None:
    """Install a single stderr handler; safe to call more than once."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
