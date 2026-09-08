"""Structured-ish logging shared by every entry point."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format=_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
        # Spark and py4j are far too chatty at INFO.
        logging.getLogger("py4j").setLevel(logging.WARNING)
        logging.getLogger("pyspark").setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
