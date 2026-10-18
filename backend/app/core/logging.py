"""Centralized logging configuration for Electronics Recommendation Agent.

Provides structured logging with timestamps, levels, logger names, and messages,
along with credential masking to prevent secret leakage in log files or console.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

# Pattern for masking MongoDB credentials and bearer tokens
MONGO_URI_AUTH_PATTERN = re.compile(r"(mongodb(?:\+srv)?://)([^:@\s]+):([^@\s]+)@")
API_KEY_PATTERN = re.compile(r"(api[-_]?key|authorization|bearer\s+)([^\s,;]+)", re.IGNORECASE)


class SensitiveDataFilter(logging.Filter):
    """Logging filter that scrubs sensitive credentials from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.mask_secrets(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self.mask_secrets(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self.mask_secrets(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        return True

    @staticmethod
    def mask_secrets(text: str) -> str:
        # Mask mongodb://user:pass@
        text = MONGO_URI_AUTH_PATTERN.sub(r"\1\2:***@", text)
        return text


def setup_logging(log_level: str = "INFO") -> None:
    """Configure centralized application logging.

    Ensures single handler attachment and sets reasonable levels
    for third-party libraries.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Prevent duplicate handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        console_handler.addFilter(SensitiveDataFilter())

        root_logger.addHandler(console_handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
            handler.addFilter(SensitiveDataFilter())

    # Quiet overly chatty third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance with the specified name."""
    return logging.getLogger(name)
