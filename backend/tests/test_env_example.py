"""`.env.example` is the user's reference for configuration, so it must not drift from the code."""

from pathlib import Path

import pytest

from app.core.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def _parse_env_example() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        assert key not in values, f"{key} is defined twice in .env.example"
        values[key.strip()] = value.strip()
    return values


@pytest.fixture
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure the developer's own shell variables cannot change what the file parses to."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name, raising=False)


def test_every_setting_is_documented_in_env_example() -> None:
    documented = set(_parse_env_example())
    expected = {name.upper() for name in Settings.model_fields}

    assert expected - documented == set(), "settings missing from .env.example"
    assert documented - expected == set(), "stale keys in .env.example"


def test_env_example_loads_without_validation_errors(clean_environment: None) -> None:
    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.ai_provider == "ollama"
    assert settings.api_host == "127.0.0.1"
    assert settings.memory_suggestions_enabled is True


def test_env_example_matches_the_code_defaults(clean_environment: None) -> None:
    from_file = Settings(_env_file=ENV_EXAMPLE)
    defaults = Settings(_env_file=None)

    for name in Settings.model_fields:
        a, b = getattr(from_file, name), getattr(defaults, name)
        if hasattr(a, "get_secret_value"):  # secrets are covered by the no-real-secrets test
            continue
        if a in ("", None) and b in ("", None):
            continue
        assert a == b, f"{name}: .env.example says {a!r} but the code default is {b!r}"


def test_env_example_contains_no_real_secrets() -> None:
    values = _parse_env_example()

    for key, value in values.items():
        if key.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
            assert value == "", f"{key} must be empty in .env.example"
