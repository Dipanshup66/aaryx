"""Start the AARYX backend: ``python -m app``.

Unlike calling ``uvicorn`` directly, this honours ``API_HOST`` / ``API_PORT`` from the settings
and refuses to listen beyond this computer unless ``API_TOKEN`` is set.
"""

import argparse
import sys
from collections.abc import Sequence

import uvicorn

from app.core.config import get_settings
from app.core.security import UnsafeBindError, is_loopback_host, validate_bind


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the bind address from the settings, then run the server."""
    parser = argparse.ArgumentParser(prog="python -m app", description="Run the AARYX backend.")
    parser.add_argument(
        "--reload", action="store_true", help="restart on code changes (development)"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    try:
        validate_bind(settings.api_host, settings.api_token is not None)
    except UnsafeBindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not is_loopback_host(settings.api_host):
        print(
            f"warning: listening on {settings.api_host}, reachable from other computers. "
            "API_TOKEN is required on every request; add the host names clients use to "
            "API_ALLOWED_HOSTS.",
            file=sys.stderr,
        )

    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
