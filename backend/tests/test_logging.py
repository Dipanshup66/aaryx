"""Unit tests for AARYX structured logging (app.core.logging)."""

import io
import logging

from app.core.logging import (
    get_logger,
    setup_logging,
)


def test_setup_logging_initialization():
    """Verify logging initializes with specified level."""
    logger = setup_logging(log_level="DEBUG")
    assert logger.level == logging.DEBUG

    logger = setup_logging(log_level="WARNING")
    assert logger.level == logging.WARNING


def test_avoid_duplicate_handlers():
    """Verify multiple calls to setup_logging do not attach duplicate stream handlers."""
    setup_logging("INFO")
    stream_handlers = [
        h for h in logging.getLogger("aaryx").handlers if type(h) is logging.StreamHandler
    ]
    assert len(stream_handlers) == 1

    # Call setup_logging repeatedly
    setup_logging("DEBUG")
    setup_logging("ERROR")

    final_stream_handlers = [
        h for h in logging.getLogger("aaryx").handlers if type(h) is logging.StreamHandler
    ]
    assert len(final_stream_handlers) == 1


def test_get_logger_namespacing():
    """Verify logger names are properly prefixed under 'aaryx' namespace."""
    log1 = get_logger()
    assert log1.name == "aaryx"

    log2 = get_logger("brain.orchestrator")
    assert log2.name == "aaryx.brain.orchestrator"

    log3 = get_logger("aaryx.providers.ollama")
    assert log3.name == "aaryx.providers.ollama"


def test_structured_log_output():
    """Verify log output follows the structured format: timestamp | level | name | message."""
    stream = io.StringIO()
    setup_logging(log_level="INFO", stream=stream)

    test_logger = get_logger("test_module")
    test_logger.info("Initializing component")

    output = stream.getvalue()
    assert "INFO" in output
    assert "aaryx.test_module" in output
    assert "Initializing component" in output
    # Verify separator
    assert " | " in output


def test_sensitive_data_redaction():
    """Verify sensitive tokens and keys are redacted from log outputs."""
    stream = io.StringIO()
    setup_logging(log_level="INFO", stream=stream)

    test_logger = get_logger("security_test")
    test_logger.info("Connecting to cloud with api_key=sk-live-998877665544332211")
    test_logger.warning("Auth header received: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
    test_logger.error("Failed for user with password=MySecretPassword123!")

    output = stream.getvalue()
    # Sensitive values must NOT appear in logs
    assert "sk-live-998877665544332211" not in output
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in output
    assert "MySecretPassword123!" not in output

    # Must contain [REDACTED]
    assert "[REDACTED]" in output


def test_exception_traceback_secret_redaction():
    """Verify sensitive tokens inside exception messages and tracebacks are scrubbed."""
    stream = io.StringIO()
    setup_logging(log_level="INFO", stream=stream)

    test_logger = get_logger("traceback_test")
    try:
        raise RuntimeError("Provider failed: api_key=sk-live-secret-in-exception-12345")
    except RuntimeError:
        test_logger.exception("Model invocation error occurred")

    output = stream.getvalue()
    # Sensitive key must NOT leak into traceback logs
    assert "sk-live-secret-in-exception-12345" not in output
    assert "api_key=[REDACTED]" in output
    # Traceback structure and context must be preserved
    assert "Traceback (most recent call last):" in output
    assert "RuntimeError" in output
    assert "Model invocation error occurred" in output


def test_dict_logging_arguments_redaction():
    """Verify secrets passed in dictionary logging arguments are redacted."""
    stream = io.StringIO()
    setup_logging(log_level="INFO", stream=stream)

    test_logger = get_logger("dict_args_test")
    test_logger.info(
        "User %(user)s logged in with password=%(pwd)s and token=%(tok)s",
        {"user": "admin_user", "pwd": "SuperSecretPassword!", "tok": "token_abc123xyz"},
    )

    output = stream.getvalue()
    assert "SuperSecretPassword!" not in output
    assert "token_abc123xyz" not in output
    assert "admin_user" in output
    assert "[REDACTED]" in output


def test_log_level_filtering():
    """Verify messages below threshold level are not emitted."""
    stream = io.StringIO()
    setup_logging(log_level="WARNING", stream=stream)

    test_logger = get_logger("filter_test")
    test_logger.debug("Debug detail that should be ignored")
    test_logger.info("Info message that should be ignored")
    test_logger.warning("Warning message that should appear")

    output = stream.getvalue()
    assert "Debug detail that should be ignored" not in output
    assert "Info message that should be ignored" not in output
    assert "Warning message that should appear" in output
