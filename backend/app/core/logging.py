"""Structured JSON logging for NewsLens-AI.

All log output is JSON-formatted for easy ingestion into log aggregators
(Loki, CloudWatch, Datadog, etc.) in production.

Usage:
    from app.core.logging import setup_logging, get_logger
    setup_logging("INFO")          # call once at startup
    logger = get_logger(__name__)
    logger.info("Processing page", extra={"page_id": 42, "duration_ms": 150})
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger.json import JsonFormatter


class _NewsLensFormatter(JsonFormatter):
    """Custom JSON formatter that adds a 'service' field to every log record."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["service"] = "newslens-ai"
        log_record["level"] = record.levelname
        # Remove redundant field added by pythonjsonlogger
        log_record.pop("color_message", None)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with JSON output to stdout.

    Call once at application startup (FastAPI lifespan or Celery worker init).

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _NewsLensFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Quiet noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "multipart", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Assumes setup_logging() has been called.

    Args:
        name: Logger name (typically __name__ of the calling module).
    """
    return logging.getLogger(name)
