"""Protection for the local API: loopback bind rule, Host header guard, optional API token.

AARYX stores personal data, so the API must not be reachable by accident:

* ``validate_bind`` refuses to listen beyond this computer unless an API token is configured.
* ``HostGuardMiddleware`` answers only to the configured Host names. A web page cannot make the
  browser talk to ``127.0.0.1`` under an attacker's domain (DNS rebinding) because that request
  would carry the attacker's Host header.
* ``ApiTokenMiddleware`` requires a shared secret on every API call when one is configured.

Both middlewares are plain ASGI classes so they never buffer streaming (SSE) responses.
"""

import hmac
import ipaddress
import json
import os
import re
from collections.abc import Iterable

from starlette.types import ASGIApp, Receive, Scope, Send

# Reachable without a token: liveness and the interactive API page. They expose the shape of
# the API and provider status, never conversations or memories.
OPEN_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})

_HOST_HEADER_RE = re.compile(r"^(?P<host>\[[0-9a-f:.]+\]|[a-z0-9._-]+)(?::(?P<port>\d{1,5}))?$")


# `python -m app` records the address it bound here. A reload worker inherits the socket opened
# by the launcher but re-reads the settings, so it must check this value, not its own settings.
LAUNCH_HOST_ENV = "AARYX_LAUNCH_HOST"


class UnsafeBindError(ValueError):
    """Raised when the server would listen beyond loopback without an API token."""


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP addresses (127.0.0.0/8, ::1)."""
    candidate = host.strip().strip("[]").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def validate_bind(host: str, has_api_token: bool) -> None:
    """Refuse a non-loopback bind address unless an API token protects the API."""
    if is_loopback_host(host) or has_api_token:
        return
    raise UnsafeBindError(
        f"Refusing to listen on '{host}': that address is reachable from other computers and "
        "AARYX holds personal data. Set API_TOKEN (at least 16 characters) in backend/.env, or "
        "use API_HOST=127.0.0.1."
    )


def check_launch_bind(has_api_token: bool) -> None:
    """Refuse to serve on a socket the launcher bound beyond loopback when no token is set.

    With ``python -m app --reload`` the launcher validates the bind once and keeps the socket
    open, while every new worker re-reads ``.env``. If ``API_TOKEN`` was removed meanwhile, the
    worker would serve the external socket without authentication. Each worker therefore
    repeats the check against the address recorded at launch (not ``API_HOST``, which may have
    changed since). Does nothing when the process was not started by ``python -m app``.
    """
    launch_host = os.environ.get(LAUNCH_HOST_ENV)
    if launch_host:
        validate_bind(launch_host, has_api_token)


def parse_host_header(value: str) -> str | None:
    """Return the lowercase host part of a Host header, or None if it is malformed."""
    match = _HOST_HEADER_RE.match(value.strip().lower())
    if match is None:
        return None
    port = match.group("port")
    if port is not None and not 0 < int(port) <= 65535:
        return None
    return match.group("host")


async def _send_json(
    send: Send,
    status: int,
    payload: dict[str, str],
    extra_headers: Iterable[tuple[bytes, bytes]] = (),
) -> None:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *extra_headers,
    ]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class HostGuardMiddleware:
    """Reject requests whose Host header is not one of the allowed host names."""

    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str]) -> None:
        self.app = app
        self.allowed = frozenset(h.lower() for h in allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        raw = dict(scope["headers"]).get(b"host", b"").decode("latin-1")
        host = parse_host_header(raw)
        if host is not None and host in self.allowed:
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await _send_json(
            send,
            400,
            {
                "error": "Invalid host",
                "detail": "This API only answers to its configured host names (API_ALLOWED_HOSTS).",
            },
        )


class ApiTokenMiddleware:
    """Require ``Authorization: Bearer <token>`` (or ``X-API-Key``) on every non-open path."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self._token = token.encode("utf-8")

    def _presented_tokens(self, scope: Scope) -> list[bytes]:
        headers = dict(scope["headers"])
        presented: list[bytes] = []
        authorization = headers.get(b"authorization", b"")
        scheme, _, credentials = authorization.partition(b" ")
        if scheme.lower() == b"bearer" and credentials.strip():
            presented.append(credentials.strip())
        api_key = headers.get(b"x-api-key", b"").strip()
        if api_key:
            presented.append(api_key)
        return presented

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Preflight requests carry no credentials by design, and open paths need none.
        if scope["method"] == "OPTIONS" or scope["path"] in OPEN_PATHS:
            await self.app(scope, receive, send)
            return

        if any(
            hmac.compare_digest(candidate, self._token)
            for candidate in self._presented_tokens(scope)
        ):
            await self.app(scope, receive, send)
            return

        await _send_json(
            send,
            401,
            {"error": "Unauthorized", "detail": "A valid API token is required."},
            extra_headers=[(b"www-authenticate", b"Bearer")],
        )


__all__ = [
    "LAUNCH_HOST_ENV",
    "OPEN_PATHS",
    "ApiTokenMiddleware",
    "HostGuardMiddleware",
    "UnsafeBindError",
    "check_launch_bind",
    "is_loopback_host",
    "parse_host_header",
    "validate_bind",
]
