"""Shared test fixtures."""

import os
from typing import Any

import pytest

from app.core.security import LAUNCH_HOST_ENV
from app.memory.extractor import InferredMemoryExtractor
from app.providers.ollama import OllamaProvider

# The API settings every test assumes. They are the code defaults, written into the process
# environment so that neither an exported variable nor a developer's backend/.env can change
# what the module-level `app` in app.main is built with (process variables win over .env).
NEUTRAL_API_ENVIRONMENT = {
    "API_HOST": "127.0.0.1",
    "API_PORT": "8000",
    "API_TOKEN": "",
    "API_CORS_ORIGINS": "tauri://localhost,http://tauri.localhost",
    "API_ALLOWED_HOSTS": "localhost,127.0.0.1,[::1]",
}


def pytest_configure(config: pytest.Config) -> None:
    """Runs before test modules are imported, so `app.main.app` is built from neutral values."""
    os.environ.update(NEUTRAL_API_ENVIRONMENT)
    os.environ.pop(LAUNCH_HOST_ENV, None)


@pytest.fixture(autouse=True)
def _forget_launch_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m app` records its bind address in the environment; never let a test leak it."""
    monkeypatch.setenv(LAUNCH_HOST_ENV, "")  # registers the undo, restoring "not set"
    monkeypatch.delenv(LAUNCH_HOST_ENV)


@pytest.fixture(autouse=True)
def _never_call_real_ollama_for_memory_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat turns now trigger memory extraction in the background.

    A MemoryService built without an injected extractor would talk to a real Ollama on
    localhost (if the developer has one running) and send test conversations to it. Block
    exactly that case; extractors given a scripted/mock provider are untouched.
    """
    original = InferredMemoryExtractor.extract

    async def guarded_extract(self: InferredMemoryExtractor, *args: Any, **kwargs: Any) -> Any:
        if type(self.provider) is OllamaProvider:
            return []
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(InferredMemoryExtractor, "extract", guarded_extract)
