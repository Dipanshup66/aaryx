"""Unit tests for AARYX configuration management (app.core.config)."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_default_settings():
    """Verify default configuration loads properly independent of .env file."""
    settings = Settings(_env_file=None)
    assert settings.app_name == "AARYX"
    assert settings.app_env == "development"
    assert settings.ai_mode == "local"
    assert settings.ai_provider == "ollama"
    assert settings.ai_fallback_enabled is False
    assert settings.ai_fallback_provider is None
    assert settings.ai_max_retries == 1
    assert settings.ai_allow_cloud_fallback is False
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_model == "qwen3.5:4b"
    assert settings.ollama_timeout == 180.0
    assert settings.cloud_base_url is None
    assert settings.cloud_api_key is None
    assert settings.cloud_model is None
    assert settings.memory_enabled is True
    assert settings.memory_database_url == "sqlite+aiosqlite:///./data/aaryx.db"
    assert settings.memory_max_history_messages == 20
    assert settings.log_level == "INFO"
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000


def test_env_loading_overrides(monkeypatch):
    """Verify environment variables override configuration defaults."""
    monkeypatch.setenv("APP_NAME", "AARYX_CUSTOM")
    monkeypatch.setenv("APP_ENV", "TEST")
    monkeypatch.setenv("AI_MODE", "LOCAL")
    monkeypatch.setenv("AI_PROVIDER", "OLLAMA")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.1.100:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)
    assert settings.app_name == "AARYX_CUSTOM"
    assert settings.app_env == "test"
    assert settings.ai_mode == "local"
    assert settings.ai_provider == "ollama"
    # Trailing slash should be stripped by validator
    assert settings.ollama_base_url == "http://192.168.1.100:11434"
    assert settings.ollama_model == "qwen3:4b"
    assert settings.log_level == "DEBUG"


def test_invalid_app_env():
    """Verify validation error when an unsupported APP_ENV is configured."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(app_env="invalid_environment", _env_file=None)
    assert "app_env" in str(exc_info.value)


def test_invalid_ai_mode():
    """Verify validation error when an unsupported AI_MODE is configured."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(ai_mode="invalid_quantum_mode", _env_file=None)
    assert "ai_mode" in str(exc_info.value)


def test_invalid_ai_provider():
    """Verify validation error when an unsupported AI_PROVIDER is configured."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(ai_provider="unsupported_quantum_ai", _env_file=None)
    assert "ai_provider" in str(exc_info.value)
    assert "Unsupported AI_PROVIDER" in str(exc_info.value)


def test_empty_ai_provider():
    """Verify validation error when AI_PROVIDER is blank or whitespace."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(ai_provider="   ", _env_file=None)
    assert "ai_provider" in str(exc_info.value)
    assert "AI_PROVIDER cannot be empty" in str(exc_info.value)


def test_invalid_ollama_base_url():
    """Verify validation error when OLLAMA_BASE_URL has an invalid scheme."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(ollama_base_url="ftp://localhost:11434", _env_file=None)
    assert "Invalid OLLAMA_BASE_URL" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        Settings(ollama_base_url="not-a-valid-url", _env_file=None)
    assert "Invalid OLLAMA_BASE_URL" in str(exc_info.value)


def test_empty_ollama_model():
    """Verify validation error when OLLAMA_MODEL is empty or whitespace."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(ollama_model="   ", _env_file=None)
    assert "OLLAMA_MODEL cannot be empty" in str(exc_info.value)


def test_empty_optional_cloud_configs():
    """Verify empty string values for optional cloud settings are converted to None."""
    settings = Settings(
        cloud_base_url="   ",
        cloud_api_key="",
        cloud_model="  ",
        _env_file=None,
    )
    assert settings.cloud_base_url is None
    assert settings.cloud_api_key is None
    assert settings.cloud_model is None


def test_valid_cloud_configs():
    """Verify valid non-empty cloud settings are preserved properly."""
    settings = Settings(
        cloud_base_url="https://api.openai.com/v1",
        cloud_api_key="sk-test-key-12345",
        cloud_model="gpt-4o-mini",
        _env_file=None,
    )
    assert settings.cloud_base_url == "https://api.openai.com/v1"
    assert settings.cloud_model == "gpt-4o-mini"
    assert settings.cloud_api_key is not None
    assert settings.cloud_api_key.get_secret_value() == "sk-test-key-12345"


def test_invalid_log_level():
    """Verify validation error when LOG_LEVEL is not a valid logging level."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(log_level="VERBOSE_TRACE", _env_file=None)
    assert "Invalid LOG_LEVEL" in str(exc_info.value)


def test_cloud_api_key_masked():
    """Verify that sensitive cloud API keys are masked in repr and str."""
    raw_secret = "sk-super-secret-production-key-12345"
    settings = Settings(cloud_api_key=raw_secret, _env_file=None)

    # SecretStr must not display raw secret in repr or str
    assert raw_secret not in str(settings.cloud_api_key)
    assert raw_secret not in repr(settings.cloud_api_key)
    assert "**********" in repr(settings.cloud_api_key)
    # The actual secret is securely accessible via get_secret_value()
    assert settings.cloud_api_key.get_secret_value() == raw_secret


def test_valid_cloud_providers():
    """Verify supported cloud providers are accepted and normalized."""
    for provider in ("openai", "OpenAI", "anthropic", "ANTHROPIC", "gemini", "Gemini"):
        settings = Settings(ai_provider=provider, _env_file=None)
        assert settings.ai_provider == provider.strip().lower()


def test_cloud_specific_api_keys_masked():
    """Verify OpenAI, Anthropic, and Gemini API keys are masked in repr and str."""
    settings = Settings(
        openai_api_key="sk-openai-secret",
        anthropic_api_key="sk-ant-secret",
        gemini_api_key="AIzaSyGeminiSecret",
        _env_file=None,
    )
    for key_attr, secret_val in (
        ("openai_api_key", "sk-openai-secret"),
        ("anthropic_api_key", "sk-ant-secret"),
        ("gemini_api_key", "AIzaSyGeminiSecret"),
    ):
        secret_obj = getattr(settings, key_attr)
        assert secret_val not in str(secret_obj)
        assert secret_val not in repr(secret_obj)
        assert "**********" in repr(secret_obj)
        assert secret_obj.get_secret_value() == secret_val


def test_fallback_enabled_without_provider():
    """Verify error when fallback is enabled without specifying fallback provider."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            ai_fallback_enabled=True,
            ai_fallback_provider=None,
            _env_file=None,
        )
    assert "AI_FALLBACK_PROVIDER must be specified" in str(exc_info.value)


def test_fallback_identical_to_primary():
    """Verify error when fallback provider is identical to primary."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            ai_provider="ollama",
            ai_fallback_enabled=True,
            ai_fallback_provider="ollama",
            _env_file=None,
        )
    assert "cannot be identical" in str(exc_info.value)


def test_local_to_cloud_fallback_without_opt_in():
    """Verify error when fallback to cloud is attempted without explicit opt-in authorization."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            ai_provider="ollama",
            ai_fallback_enabled=True,
            ai_fallback_provider="openai",
            ai_allow_cloud_fallback=False,
            openai_api_key="sk-test",
            _env_file=None,
        )
    assert "requires explicit opt-in" in str(exc_info.value)


def test_local_to_cloud_fallback_with_opt_in_success():
    """Verify valid local-to-cloud fallback configuration with opt-in and credentials."""
    settings = Settings(
        ai_provider="ollama",
        ai_fallback_enabled=True,
        ai_fallback_provider="openai",
        ai_allow_cloud_fallback=True,
        openai_api_key="sk-test",
        _env_file=None,
    )
    assert settings.ai_fallback_enabled is True
    assert settings.ai_fallback_provider == "openai"
    assert settings.ai_allow_cloud_fallback is True


def test_memory_configuration():
    """Verify memory configuration options and normalization."""
    # Test normalization from sqlite:// to sqlite+aiosqlite://
    settings = Settings(
        memory_database_url="sqlite:///custom/path.db",
        memory_max_history_messages=50,
        _env_file=None,
    )
    assert settings.memory_database_url == "sqlite+aiosqlite:///custom/path.db"
    assert settings.memory_max_history_messages == 50

    # Test invalid database url scheme
    with pytest.raises(ValidationError) as exc_info:
        Settings(memory_database_url="postgresql://user:pass@localhost/db", _env_file=None)
    assert "memory_database_url" in str(exc_info.value).lower()

    # Test empty database url
    with pytest.raises(ValidationError) as exc_info2:
        Settings(memory_database_url="   ", _env_file=None)
    assert "memory_database_url" in str(exc_info2.value).lower()


def test_get_settings_caching():
    """Verify get_settings returns a cached instance and handles cache clearing."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
