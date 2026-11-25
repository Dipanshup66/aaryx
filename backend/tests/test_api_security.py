"""Local API protection: Host guard (DNS rebinding), optional API token, CORS, bind rule."""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from app import __main__ as launcher
from app.brain.orchestrator import BrainOrchestrator
from app.core.config import Settings
from app.core.security import (
    LAUNCH_HOST_ENV,
    UnsafeBindError,
    check_launch_bind,
    is_loopback_host,
    parse_host_header,
    validate_bind,
)
from app.main import create_app, lifespan
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
)

TOKEN = "t0ken-" + "x" * 30


class FakeProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        return ChatResponse(content="hi there", model="m", provider="fake")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self.calls += 1
        yield ChatStreamChunk(delta="hi ", model="m", provider="fake")
        yield ChatStreamChunk(delta="there", model="m", provider="fake", finish_reason="stop")

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="m")]

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="fake")

    async def aclose(self) -> None:
        pass


@asynccontextmanager
async def api(**settings_kwargs):
    """A fresh app with the given settings, a fake model, and no database."""
    app = create_app(Settings(_env_file=None, **settings_kwargs))
    provider = FakeProvider()
    app.state.provider = provider
    app.state.orchestrator = BrainOrchestrator(provider=provider)
    app.state.memory_service = None
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://localhost") as client:
        client.provider = provider  # type: ignore[attr-defined]
        yield client


def bearer(token: str = TOKEN) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


# ===================================================================== Host header guard

ALLOWED_HOSTS = [
    "localhost",
    "localhost:8000",
    "LOCALHOST:8000",
    "127.0.0.1",
    "127.0.0.1:8000",
    "[::1]",
    "[::1]:8000",
]


@pytest.mark.parametrize("host", ALLOWED_HOSTS)
async def test_normal_local_hosts_are_accepted(host: str) -> None:
    async with api() as client:
        res = await client.get("/api/models", headers={"host": host})

    assert res.status_code == 200, res.text


BLOCKED_HOSTS = [
    "evil.com",
    "evil.com:8000",
    "localhost.evil.com",
    "127.0.0.1.evil.com",
    "localhost:80@evil.com",
    "localhost:99999",
    "localhost:",
    "localhost/evil",
    "[::1",
    "0.0.0.0",
    "192.168.1.5:8000",
    "",
    " ",
]


@pytest.mark.parametrize("host", BLOCKED_HOSTS, ids=repr)
async def test_other_host_names_are_refused_before_anything_runs(host: str) -> None:
    async with api() as client:
        res = await client.get("/api/models", headers={"host": host})
        health = await client.get("/health", headers={"host": host})

    assert res.status_code == 400 and res.json()["error"] == "Invalid host"
    assert health.status_code == 400
    assert client.provider.calls == 0  # type: ignore[attr-defined]


async def test_dns_rebinding_cannot_reach_the_chat_api() -> None:
    async with api() as client:
        res = await client.post(
            "/api/chat", json={"message": "hello"}, headers={"host": "attacker.example:8000"}
        )

    assert res.status_code == 400
    assert client.provider.calls == 0  # type: ignore[attr-defined]


async def test_extra_allowed_hosts_can_be_configured() -> None:
    async with api(api_allowed_hosts="my-laptop,localhost") as client:
        ok = await client.get("/api/models", headers={"host": "my-laptop:8000"})
        bad = await client.get("/api/models", headers={"host": "127.0.0.1"})

    assert ok.status_code == 200
    assert bad.status_code == 400  # 127.0.0.1 was not listed this time


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("localhost", "localhost"),
        ("LocalHost:8000", "localhost"),
        ("[::1]:8000", "[::1]"),
        ("127.0.0.1:1", "127.0.0.1"),
        ("a@b", None),
        ("host:0", None),
        ("host:65536", None),
        ("two words", None),
        ("", None),
    ],
)
def test_host_header_parsing(value: str, expected: str | None) -> None:
    assert parse_host_header(value) == expected


# ======================================================================== API token


async def test_without_a_configured_token_nothing_is_required() -> None:
    async with api() as client:
        res = await client.get("/api/models")

    assert res.status_code == 200


async def test_valid_token_in_either_header_is_accepted() -> None:
    async with api(api_token=TOKEN) as client:
        a = await client.get("/api/models", headers=bearer())
        b = await client.get("/api/models", headers={"authorization": f"bearer {TOKEN}"})
        c = await client.get("/api/models", headers={"x-api-key": TOKEN})
        d = await client.get(
            "/api/models", headers={"authorization": "Bearer wrong", "x-api-key": TOKEN}
        )

    assert [r.status_code for r in (a, b, c, d)] == [200, 200, 200, 200]


BAD_CREDENTIALS = [
    {},
    bearer("wrong-token-wrong-token-wrong"),
    bearer(TOKEN + "x"),
    bearer(TOKEN[:-1]),
    bearer(TOKEN[:8]),
    bearer(""),
    {"authorization": "Bearer"},
    {"authorization": TOKEN},
    {"authorization": f"Basic {TOKEN}"},
    {"authorization": f"Token {TOKEN}"},
    {"x-api-key": ""},
    {"x-api-key": TOKEN.upper()},
    {"cookie": f"token={TOKEN}"},
]


@pytest.mark.parametrize("headers", BAD_CREDENTIALS, ids=lambda h: json.dumps(h)[:40])
async def test_missing_or_wrong_credentials_get_401(headers: dict[str, str]) -> None:
    async with api(api_token=TOKEN) as client:
        res = await client.get("/api/models", headers=headers)

    assert res.status_code == 401
    assert res.headers["www-authenticate"] == "Bearer"
    assert res.json() == {"error": "Unauthorized", "detail": "A valid API token is required."}
    assert TOKEN not in res.text


async def test_token_in_the_url_is_not_accepted() -> None:
    async with api(api_token=TOKEN) as client:
        res = await client.get("/api/models", params={"token": TOKEN, "api_key": TOKEN})

    assert res.status_code == 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/chat"),
        ("POST", "/api/chat/stream"),
        ("GET", "/api/models"),
        ("GET", "/api/memories"),
        ("POST", "/api/memories"),
        ("PATCH", "/api/memories/abc"),
        ("DELETE", "/api/memories/abc"),
        ("POST", "/api/memories/abc/approve"),
        ("POST", "/api/memories/abc/reject"),
        ("GET", "/api/conversations"),
        ("DELETE", "/api/conversations/abc"),
        ("GET", "/api/does-not-exist"),
        ("GET", "/health/../api/memories"),
        ("GET", "/HEALTH"),
    ],
)
async def test_every_api_route_requires_the_token(method: str, path: str) -> None:
    async with api(api_token=TOKEN) as client:
        res = await client.request(method, path, json={"message": "hello"})

    assert res.status_code == 401, (method, path, res.status_code)
    assert client.provider.calls == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize("path", ["/health", "/docs", "/openapi.json", "/redoc"])
async def test_liveness_and_docs_stay_open(path: str) -> None:
    async with api(api_token=TOKEN) as client:
        res = await client.get(path)

    assert res.status_code == 200, path


async def test_chat_and_streaming_work_normally_with_the_token() -> None:
    async with api(api_token=TOKEN) as client:
        chat = await client.post("/api/chat", json={"message": "hello"}, headers=bearer())
        stream = await client.post("/api/chat/stream", json={"message": "hello"}, headers=bearer())

    assert chat.status_code == 200 and chat.json()["reply"] == "hi there"
    assert stream.status_code == 200
    assert "data: [DONE]" in stream.text and '"delta":"hi "' in stream.text


async def test_host_guard_runs_before_the_token_check() -> None:
    async with api(api_token=TOKEN) as client:
        res = await client.get("/api/models", headers={**bearer(), "host": "evil.com"})

    assert res.status_code == 400


# ============================================================================== CORS

APP_ORIGIN = "tauri://localhost"
PREFLIGHT = {
    "origin": APP_ORIGIN,
    "access-control-request-method": "POST",
    "access-control-request-headers": "content-type,authorization",
}


async def test_the_desktop_app_origin_can_preflight_and_call() -> None:
    async with api() as client:
        pre = await client.options("/api/chat", headers=PREFLIGHT)
        real = await client.get("/api/models", headers={"origin": APP_ORIGIN})

    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == APP_ORIGIN
    assert "authorization" in pre.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in pre.headers
    assert real.headers["access-control-allow-origin"] == APP_ORIGIN


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.com",
        "http://localhost:3000",
        "null",
        "tauri://localhost.evil.com",
        "http://tauri.localhost.evil.com",
    ],
)
async def test_other_origins_get_no_cors_permission(origin: str) -> None:
    async with api() as client:
        pre = await client.options("/api/chat", headers={**PREFLIGHT, "origin": origin})
        real = await client.get("/api/models", headers={"origin": origin})

    assert "access-control-allow-origin" not in pre.headers
    assert "access-control-allow-origin" not in real.headers


async def test_preflight_works_without_a_token_even_when_a_token_is_required() -> None:
    async with api(api_token=TOKEN) as client:
        pre = await client.options("/api/chat", headers=PREFLIGHT)
        denied = await client.get("/api/models", headers={"origin": APP_ORIGIN})

    assert pre.status_code == 200
    assert denied.status_code == 401
    assert denied.headers["access-control-allow-origin"] == APP_ORIGIN  # the app can read the error


async def test_extra_dev_origin_can_be_added() -> None:
    async with api(api_cors_origins="tauri://localhost,http://localhost:1420") as client:
        res = await client.get("/api/models", headers={"origin": "http://localhost:1420"})

    assert res.headers["access-control-allow-origin"] == "http://localhost:1420"


async def test_empty_origin_list_turns_cors_off() -> None:
    async with api(api_cors_origins="") as client:
        res = await client.get("/api/models", headers={"origin": APP_ORIGIN})

    assert res.status_code == 200 and "access-control-allow-origin" not in res.headers


# ============================================================================ bind rule


@pytest.mark.parametrize(
    "host", ["localhost", "LOCALHOST", "127.0.0.1", "127.1.2.3", "::1", "[::1]", " 127.0.0.1 "]
)
def test_loopback_addresses(host: str) -> None:
    assert is_loopback_host(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "::",
        "192.168.1.5",
        "10.0.0.1",
        "example.com",
        "",
        " ",
        "localhost.evil.com",
        "127.0.0.1.evil.com",
    ],
)
def test_non_loopback_addresses(host: str) -> None:
    assert is_loopback_host(host) is False


def test_validate_bind_rules() -> None:
    validate_bind("127.0.0.1", has_api_token=False)
    validate_bind("localhost", has_api_token=False)
    validate_bind("0.0.0.0", has_api_token=True)
    with pytest.raises(UnsafeBindError, match="API_TOKEN"):
        validate_bind("0.0.0.0", has_api_token=False)
    with pytest.raises(UnsafeBindError):
        validate_bind("192.168.1.5", has_api_token=False)


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(launcher.uvicorn, "run", lambda app, **kw: calls.append({"app": app, **kw}))
    return calls


def _use_settings(monkeypatch: pytest.MonkeyPatch, **kwargs) -> None:
    monkeypatch.setattr(launcher, "get_settings", lambda: Settings(_env_file=None, **kwargs))


def test_launcher_defaults_to_loopback(monkeypatch, runs) -> None:
    _use_settings(monkeypatch)

    assert launcher.main([]) == 0
    assert runs == [{"app": "app.main:app", "host": "127.0.0.1", "port": 8000, "reload": False}]


def test_launcher_uses_configured_port_and_reload_flag(monkeypatch, runs) -> None:
    _use_settings(monkeypatch, api_port=9100)

    launcher.main(["--reload"])

    assert runs[0]["port"] == 9100 and runs[0]["reload"] is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5"])
def test_launcher_refuses_to_expose_the_api_without_a_token(
    monkeypatch, runs, capsys, host
) -> None:
    _use_settings(monkeypatch, api_host=host)

    assert launcher.main([]) == 2
    assert runs == []
    assert "API_TOKEN" in capsys.readouterr().err


def test_launcher_allows_exposure_with_a_token_and_warns(monkeypatch, runs, capsys) -> None:
    _use_settings(monkeypatch, api_host="0.0.0.0", api_token=TOKEN)

    assert launcher.main([]) == 0
    assert runs[0]["host"] == "0.0.0.0"
    err = capsys.readouterr().err
    assert "warning" in err and TOKEN not in err


def test_launcher_rejects_unknown_flags(monkeypatch, runs) -> None:
    _use_settings(monkeypatch)

    with pytest.raises(SystemExit) as info:
        launcher.main(["--host", "0.0.0.0"])  # the host must come from the settings, not the CLI

    assert info.value.code == 2 and runs == []


# ============================================================================ settings


@pytest.mark.parametrize("token", ["", "   ", None])
def test_empty_token_means_not_set(token) -> None:
    assert Settings(_env_file=None, api_token=token).api_token is None


@pytest.mark.parametrize("token", ["short", "a" * 15, "  short with pad  "])
def test_short_tokens_are_refused(token: str) -> None:
    with pytest.raises(ValueError, match="at least 16"):
        Settings(_env_file=None, api_token=token)


def test_a_rejected_token_is_not_echoed_in_the_error() -> None:
    almost = "almost-secret-1"  # 15 characters: one too short, but it may be a real secret

    with pytest.raises(ValueError) as info:
        Settings(_env_file=None, api_token=almost)

    assert "at least 16" in str(info.value) and almost not in str(info.value)


def test_a_rejected_provider_key_value_is_not_echoed_either() -> None:
    with pytest.raises(ValueError) as info:
        Settings(_env_file=None, ai_mode="sk-real-looking-secret-value")  # type: ignore[arg-type]

    assert "sk-real-looking-secret-value" not in str(info.value)


def test_token_is_never_shown_in_repr_or_dumps() -> None:
    settings = Settings(_env_file=None, api_token=TOKEN)

    assert TOKEN not in repr(settings) and TOKEN not in str(settings)
    assert TOKEN not in json.dumps(settings.model_dump(mode="json"))


def test_token_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("API_TOKEN", TOKEN)

    assert Settings(_env_file=None).api_token.get_secret_value() == TOKEN  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "origins",
    [
        "*",
        "https://*.evil.com",
        "http://a.com/path",
        "a.com",
        "http://",
        "javascript:alert(1)",
        "http://a.com, *",
    ],
)
def test_bad_cors_origins_are_refused(origins: str) -> None:
    with pytest.raises(ValueError, match="API_CORS_ORIGINS"):
        Settings(_env_file=None, api_cors_origins=origins)


def test_cors_origins_parse_with_spaces_and_trailing_commas() -> None:
    s = Settings(_env_file=None, api_cors_origins=" tauri://localhost , http://localhost:1420 ,")

    assert s.cors_origins == ["tauri://localhost", "http://localhost:1420"]


@pytest.mark.parametrize(
    "hosts", ["", " , ", "*", "http://localhost", "localhost:8000", "a/b", "*.evil.com"]
)
def test_bad_allowed_hosts_are_refused(hosts: str) -> None:
    with pytest.raises(ValueError, match="API_ALLOWED_HOSTS"):
        Settings(_env_file=None, api_allowed_hosts=hosts)


def test_allowed_hosts_are_lowercased() -> None:
    assert Settings(_env_file=None, api_allowed_hosts="MyHost,[::1]").allowed_hosts == [
        "myhost",
        "[::1]",
    ]


@pytest.mark.parametrize("port", [0, -1, 65536, 99999])
def test_bad_ports_are_refused(port: int) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, api_port=port)


# ============================================================== reload workers (launch bind)


def test_check_launch_bind_does_nothing_when_not_started_by_the_launcher() -> None:
    assert LAUNCH_HOST_ENV not in os.environ

    check_launch_bind(has_api_token=False)  # e.g. tests, or uvicorn started by hand


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_launch_needs_no_token(monkeypatch, host: str) -> None:
    monkeypatch.setenv(LAUNCH_HOST_ENV, host)

    check_launch_bind(has_api_token=False)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5"])
def test_external_launch_without_token_is_refused_with_token_it_is_allowed(
    monkeypatch, host: str
) -> None:
    monkeypatch.setenv(LAUNCH_HOST_ENV, host)

    with pytest.raises(UnsafeBindError, match="API_TOKEN"):
        check_launch_bind(has_api_token=False)
    check_launch_bind(has_api_token=True)


def test_empty_launch_host_is_treated_as_not_set(monkeypatch) -> None:
    monkeypatch.setenv(LAUNCH_HOST_ENV, "")

    check_launch_bind(has_api_token=False)


def test_launcher_records_the_address_it_bound(monkeypatch, runs) -> None:
    _use_settings(monkeypatch, api_host="0.0.0.0", api_token=TOKEN)

    assert launcher.main([]) == 0
    assert os.environ[LAUNCH_HOST_ENV] == "0.0.0.0"


def test_launcher_records_loopback_by_default(monkeypatch, runs) -> None:
    _use_settings(monkeypatch)

    launcher.main([])

    assert os.environ[LAUNCH_HOST_ENV] == "127.0.0.1"


def test_a_refused_launch_records_nothing_and_starts_nothing(monkeypatch, runs) -> None:
    _use_settings(monkeypatch, api_host="0.0.0.0")

    assert launcher.main([]) == 2
    assert LAUNCH_HOST_ENV not in os.environ and runs == []


async def test_worker_refuses_to_start_on_an_external_socket_without_a_token(monkeypatch) -> None:
    monkeypatch.setenv(LAUNCH_HOST_ENV, "0.0.0.0")
    app = create_app(Settings(_env_file=None, memory_enabled=False))  # token was removed

    with pytest.raises(UnsafeBindError, match="API_TOKEN"):
        async with lifespan(app):
            pytest.fail("the worker must not start serving")

    assert not hasattr(app.state, "provider"), "nothing may be started before the check"


async def test_worker_starts_on_an_external_socket_when_the_token_is_still_set(monkeypatch) -> None:
    monkeypatch.setenv(LAUNCH_HOST_ENV, "0.0.0.0")
    app = create_app(Settings(_env_file=None, memory_enabled=False, api_token=TOKEN))

    async with lifespan(app):
        assert app.state.provider is not None


async def test_worker_uses_the_launch_address_not_the_current_api_host(monkeypatch) -> None:
    # The launcher's socket is the real exposure. API_HOST changed in .env afterwards is irrelevant.
    monkeypatch.setenv(LAUNCH_HOST_ENV, "0.0.0.0")
    app = create_app(Settings(_env_file=None, memory_enabled=False, api_host="127.0.0.1"))

    with pytest.raises(UnsafeBindError):
        async with lifespan(app):
            pass

    monkeypatch.setenv(LAUNCH_HOST_ENV, "127.0.0.1")
    exposed_in_settings_only = create_app(
        Settings(_env_file=None, memory_enabled=False, api_host="0.0.0.0")
    )
    async with lifespan(exposed_in_settings_only):  # the socket stays on loopback
        pass


async def test_reload_sequence_token_removed_after_launch(monkeypatch, runs) -> None:
    """Launch exposed with a token, then restart the worker after the token left .env."""
    _use_settings(monkeypatch, api_host="0.0.0.0", api_token=TOKEN)
    assert launcher.main(["--reload"]) == 0 and runs[0]["reload"] is True

    first_worker = create_app(Settings(_env_file=None, memory_enabled=False, api_token=TOKEN))
    async with lifespan(first_worker):
        pass

    second_worker = create_app(Settings(_env_file=None, memory_enabled=False))
    with pytest.raises(UnsafeBindError):
        async with lifespan(second_worker):
            pass
