"""Structured logging module for AARYX.

Provides clean, unified log formatting across all application modules,
prevents duplicate handlers, and filters out accidental secret leaks.
"""

import logging
import re
import sys
from typing import TextIO

# Regular expressions for masking common secret patterns
_SENSITIVE_PATTERNS = [
    re.compile(r"(bearer\s+)[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"(api[_-]?key[\"'\s:=]+)[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"(password[\"'\s:=]+)[^\s,\"']+", re.IGNORECASE),
    re.compile(r"(secret[\"'\s:=]+)[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"(token[_-]?[\"'\s:=]+)[a-zA-Z0-9_\-\.]+", re.IGNORECASE),
]

# Credentials inside URLs. Single definition, shared with app.providers.base.redact_secrets.
URL_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # scheme://user:pass@host  or  scheme://token@host
    (re.compile(r"(://)[^\s/?#@]+@"), r"\1[REDACTED]@"),
    # secret-bearing query parameters
    (
        re.compile(
            r"(?i)([?&;](?:api[_-]?key|key|token|access[_-]?token|auth|authorization|"
            r"signature|sig|secret|password|passwd|pwd|x-api-key)=)[^&\s#\"']+"
        ),
        r"\1[REDACTED]",
    ),
)

_SENSITIVE_KEY_NAMES = {
    "password",
    "pwd",
    "token",
    "tok",
    "secret",
    "api_key",
    "apikey",
    "auth",
    "authorization",
}


def redact_sensitive_text(text: str) -> str:
    """Scrub sensitive credentials (bearer tokens, API keys, passwords, secrets) from text."""
    for url_pattern, url_replacement in URL_SECRET_PATTERNS:
        text = url_pattern.sub(url_replacement, text)
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub(r"\g<1>[REDACTED]", text)
    return text


def _scrub_arg(arg: object) -> object:
    """Redact a log argument. Numbers/None keep their type so %d / %f formatting still works;
    every other object (str, httpx.URL, exceptions, ...) is rendered and redacted."""
    if arg is None or isinstance(arg, (bool, int, float)):
        return arg
    return redact_sensitive_text(arg if isinstance(arg, str) else str(arg))


class SensitiveDataFilter(logging.Filter):
    """Logging filter that scrubs sensitive credentials from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_text(record.msg)

        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(_scrub_arg(arg) for arg in record.args)
            elif isinstance(record.args, dict):
                record.args = {
                    k: (
                        "[REDACTED]"
                        if str(k).lower() in _SENSITIVE_KEY_NAMES
                        else (redact_sensitive_text(v) if isinstance(v, str) else v)
                    )
                    for k, v in record.args.items()
                }

        if getattr(record, "exc_text", None) and isinstance(record.exc_text, str):
            record.exc_text = redact_sensitive_text(record.exc_text)

        return True


class StructuredFormatter(logging.Formatter):
    """Human-readable structured log formatter with comprehensive secret redaction."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format the record and scrub sensitive text from message and tracebacks."""
        formatted = super().format(record)
        return redact_sensitive_text(formatted)


def protect_third_party_loggers() -> None:
    """Attach the redaction filter to HTTP client loggers that live outside 'aaryx.*'.

    httpx logs every request at INFO including the full URL, which can carry credentials
    (``https://user:pass@host`` or ``?api_key=...``). A logger-level filter runs before any
    handler, so it protects whatever logging configuration the host process uses.
    Idempotent.
    """
    for name in ("httpx", "httpcore"):
        target = logging.getLogger(name)
        if not any(isinstance(f, SensitiveDataFilter) for f in target.filters):
            target.addFilter(SensitiveDataFilter())


protect_third_party_loggers()


def setup_logging(
    log_level: str | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Initialize structured logging for the application.

    Configures the root 'aaryx' logger with a stream handler and sensitive data filter.
    Idempotent: prevents duplicate handlers if called multiple times.

    Args:
        log_level: Log level string (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
                   If None, defaults to 'INFO'.
        stream: Optional output stream (defaults to sys.stdout). Useful for tests.

    Returns:
        The configured 'aaryx' base logger.
    """
    effective_level = (log_level or "INFO").upper()
    numeric_level = getattr(logging, effective_level, logging.INFO)

    root_logger = logging.getLogger("aaryx")
    root_logger.setLevel(numeric_level)

    # Ensure sensitive filter is attached to logger so all handlers benefit
    if not any(isinstance(f, SensitiveDataFilter) for f in root_logger.filters):
        root_logger.addFilter(SensitiveDataFilter())

    # Avoid duplicate handlers: if an existing StreamHandler is already attached,
    # reconfigure it instead of adding redundant handlers.
    target_stream = stream or sys.stdout
    existing_handler: logging.Handler | None = None

    for handler in root_logger.handlers:
        if type(handler) is logging.StreamHandler or (
            isinstance(handler, logging.StreamHandler)
            and not type(handler).__name__.startswith("LogCapture")
        ):
            existing_handler = handler
            break

    if existing_handler is not None:
        existing_handler.setLevel(numeric_level)
        existing_handler.setFormatter(StructuredFormatter())
        if stream is not None:
            existing_handler.setStream(stream)
    else:
        new_handler = logging.StreamHandler(target_stream)
        new_handler.setLevel(numeric_level)
        new_handler.setFormatter(StructuredFormatter())
        new_handler.addFilter(SensitiveDataFilter())
        root_logger.addHandler(new_handler)

    # Do not propagate to Python root logger to prevent duplicate standard logging
    root_logger.propagate = False
    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger scoped under the 'aaryx' namespace.

    Example:
        logger = get_logger("core.config")
        # Returns logger named 'aaryx.core.config'
    """
    if not name or name == "aaryx":
        return logging.getLogger("aaryx")
    if name.startswith("aaryx."):
        return logging.getLogger(name)
    return logging.getLogger(f"aaryx.{name}")
