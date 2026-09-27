"""Project-wide logging setup. Use ``get_logger(__name__)`` in every module."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger; the root handler is configured once on first call."""
    global _configured
    if not _configured:
        logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", stream=sys.stdout)
        # rasterio/GDAL are very chatty at INFO level
        logging.getLogger("rasterio").setLevel(logging.WARNING)
        _configured = True
    return logging.getLogger(name)
