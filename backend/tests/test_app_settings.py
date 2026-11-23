"""An app must run with the settings it was created with, not with whatever the cached
environment settings say: startup services, provider routing, logging and the health route.

Every test plants a "trap": `get_settings()` is made to return settings that would enable memory
and write to a trap database. If the app ever reads the cached settings instead of its own,
the trap fires (a trap database appears, or a trap value shows up).
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import app.main as main_module
from app.api.deps import get_provider
from app.core.config import Settings
from app.main import create_app, lifespan
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
)


def _db_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.as_posix()}"


@pytest.fixture
def trap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make the cached settings enable memory and point at a database that must stay unused."""
    trap_db = tmp_path / "trap" / "cached.db"
    trap_settings = Settings(
        _env_file=None,
        memory_enabled=True,
        memory_database_url=_db_url(trap_db),
        memory_max_context_items=1,
        memory_suggestions_enabled=True,
        app_name="TrapApp",
        app_env="staging",
        ollama_model="trap-model",
        log_level="CRITICAL",
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: trap_settings)
    return trap_db


async def test_memory_disabled_app_creates_no_database_and_no_service(
    tmp_path: Path, trap: Path
) -> None:
    own_db = tmp_path / "own.db"
    app = create_app(
        Settings(_env_file=None, memory_enabled=False, memory_database_url=_db_url(own_db))
    )

    async with lifespan(app):
        assert app.state.memory_service is None
        assert app.state.memory_engine is None

    assert not own_db.exists(), "memory is disabled: no database may be created"
    assert not trap.exists(), "the app read the cached settings instead of its own"
    assert not trap.parent.exists()


async def test_memory_enabled_app_uses_its_own_database_path(tmp_path: Path, trap: Path) -> None:
    own_db = tmp_path / "own.db"
    app = create_app(Settings(_env_file=None, memory_database_url=_db_url(own_db)))

    async with lifespan(app):
        assert app.state.memory_service is not None

    assert own_db.exists()
    assert not trap.exists()


async def test_memory_service_gets_the_apps_memory_settings(tmp_path: Path, trap: Path) -> None:
    settings = Settings(
        _env_file=None,
        memory_database_url=_db_url(tmp_path / "own.db"),
        memory_suggestions_enabled=False,
        memory_max_context_items=5,
        memory_max_history_messages=7,
    )
    app = create_app(settings)

    async with lifespan(app):
        service = app.state.memory_service
        assert service.settings is settings
        assert service.settings.memory_suggestions_enabled is False
        assert service.settings.memory_max_context_items == 5
        assert service.max_history_messages == 7
        assert service.extractor.settings is settings


async def test_provider_router_uses_the_apps_settings(tmp_path: Path, trap: Path) -> None:
    app = create_app(
        Settings(
            _env_file=None,
            memory_enabled=False,
            ollama_model="my-own-model",
            ai_max_retries=3,
        )
    )

    async with lifespan(app):
        router = app.state.provider
        primary = await router.get_provider(router.primary_provider)
        assert primary.default_model == "my-own-model"
        assert router.settings.ai_max_retries == 3


async def test_logging_is_set_up_from_the_apps_settings(
    tmp_path: Path, trap: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    levels: list[str] = []
    monkeypatch.setattr(main_module, "setup_logging", levels.append)
    app = create_app(Settings(_env_file=None, memory_enabled=False, log_level="DEBUG"))

    async with lifespan(app):
        pass

    assert levels == ["DEBUG"]


async def test_two_apps_in_one_process_do_not_share_settings(tmp_path: Path, trap: Path) -> None:
    quiet = create_app(Settings(_env_file=None, memory_enabled=False))
    full = create_app(Settings(_env_file=None, memory_database_url=_db_url(tmp_path / "full.db")))

    async with lifespan(quiet):
        async with lifespan(full):
            assert quiet.state.memory_service is None
            assert full.state.memory_service is not None
        assert quiet.state.memory_service is None

    assert not trap.exists()


async def test_app_not_built_by_create_app_falls_back_to_cached_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback_db = tmp_path / "fallback.db"
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: Settings(_env_file=None, memory_database_url=_db_url(fallback_db)),
    )
    plain = FastAPI()  # no `state.settings`

    async with lifespan(plain):
        assert plain.state.memory_service is not None

    assert fallback_db.exists()


def test_create_app_stores_its_settings_on_the_app(trap: Path) -> None:
    settings = Settings(_env_file=None, memory_enabled=False)

    assert create_app(settings).state.settings is settings


# ------------------------------------------------------------------------ health route


class _Healthy(LLMProvider):
    async def generate(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="x", model="m", provider="p")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(delta="x", model="m", provider="p")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="p")

    async def aclose(self) -> None:
        pass


async def test_health_reports_the_apps_name_and_environment(trap: Path) -> None:
    app = create_app(
        Settings(_env_file=None, memory_enabled=False, app_name="MyAARYX", app_env="production")
    )
    app.dependency_overrides[get_provider] = lambda: _Healthy()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        body = (await client.get("/health")).json()

    assert body["app"] == "MyAARYX" and body["environment"] == "production"
    assert body["status"] == "ok"
