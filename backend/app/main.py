"""AARYX backend entry point and FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes_chat import router as chat_router
from app.api.routes_conversations import router as conversations_router
from app.api.routes_health import router as health_router
from app.api.routes_memories import router as memories_router
from app.api.routes_models import router as models_router
from app.brain.orchestrator import BrainOrchestrator
from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.core.security import ApiTokenMiddleware, HostGuardMiddleware, check_launch_bind
from app.memory.database import create_database_engine, get_session_factory, init_db
from app.memory.exceptions import (
    ConversationNotFoundError,
    ConversationValidationError,
    MemoryNotFoundError,
    MemoryValidationError,
)
from app.memory.service import MemoryService
from app.providers.base import (
    ProviderConnectionError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    describe_error,
    error_payload,
)
from app.providers.router import ProviderRouter

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup, provider initialization, memory setup, and graceful shutdown.

    Uses the settings the app was created with (``create_app``); only an app built some other
    way falls back to the cached environment settings.
    """
    settings = getattr(app.state, "settings", None) or get_settings()
    # Before anything starts: a reload worker must not serve an external socket without a token.
    check_launch_bind(settings.api_token is not None)
    setup_logging(settings.log_level)
    logger.info(
        "Initializing %s in %s mode (provider=%s, fallback_enabled=%s, memory_enabled=%s)",
        settings.app_name,
        settings.ai_mode,
        settings.ai_provider,
        settings.ai_fallback_enabled,
        settings.memory_enabled,
    )

    # Initialize ProviderRouter with configured primary and fallback
    router = ProviderRouter(settings=settings)
    await router.get_provider(router.primary_provider)
    orchestrator = BrainOrchestrator(
        provider=router,
        max_history_messages=settings.memory_max_history_messages,
    )

    app.state.provider = router
    app.state.router = router
    app.state.orchestrator = orchestrator

    # Initialize Persistent Conversation Memory if enabled
    memory_engine = None
    memory_service = None
    if settings.memory_enabled:
        logger.info(
            "Initializing persistent conversation memory at %s",
            settings.memory_database_url,
        )
        memory_engine = create_database_engine(settings.memory_database_url)
        await init_db(memory_engine)
        session_factory = get_session_factory(memory_engine)
        memory_service = MemoryService(
            session_factory=session_factory,
            max_history_messages=settings.memory_max_history_messages,
            settings=settings,
        )
        logger.info("Persistent conversation memory successfully initialized")

    app.state.memory_engine = memory_engine
    app.state.memory_service = memory_service

    yield

    logger.info("Shutting down %s...", settings.app_name)
    await router.aclose()
    if memory_engine is not None:
        logger.info("Disposing memory database engine...")
        await memory_engine.dispose()
    logger.info("Shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the central FastAPI application."""
    settings = settings or get_settings()
    app = FastAPI(
        title=f"{settings.app_name} API",
        version=__version__,
        description="AARYX Personal AI Companion API",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # Middleware order: the last one added is the outermost. A request passes through the
    # Host check first, then CORS (which answers browser preflights), then the API token.
    if settings.api_token is not None:
        app.add_middleware(ApiTokenMiddleware, token=settings.api_token.get_secret_value())
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
            max_age=600,
        )
    app.add_middleware(HostGuardMiddleware, allowed_hosts=settings.allowed_hosts)

    # Register Provider Exception Handlers
    @app.exception_handler(ProviderConnectionError)
    async def provider_connection_error_handler(
        request: Request, exc: ProviderConnectionError
    ) -> JSONResponse:
        logger.error("Provider connection error: %s", describe_error(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_payload(exc),
        )

    @app.exception_handler(ProviderTimeoutError)
    async def provider_timeout_error_handler(
        request: Request, exc: ProviderTimeoutError
    ) -> JSONResponse:
        logger.error("Provider timeout: %s", describe_error(exc))
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=error_payload(exc),
        )

    @app.exception_handler(ProviderResponseError)
    async def provider_response_error_handler(
        request: Request, exc: ProviderResponseError
    ) -> JSONResponse:
        logger.error("Provider response error: %s", describe_error(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=error_payload(exc),
        )

    @app.exception_handler(ProviderError)
    async def provider_generic_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
        logger.error("Provider generic error: %s", describe_error(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_payload(exc),
        )

    # Register Memory Exception Handlers
    @app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found_handler(
        request: Request, exc: ConversationNotFoundError
    ) -> JSONResponse:
        logger.warning("Conversation not found: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "Not Found",
                "detail": str(exc),
            },
        )

    @app.exception_handler(ConversationValidationError)
    async def conversation_validation_error_handler(
        request: Request, exc: ConversationValidationError
    ) -> JSONResponse:
        logger.warning("Conversation validation error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Validation Error",
                "detail": str(exc),
            },
        )

    @app.exception_handler(MemoryNotFoundError)
    async def memory_not_found_handler(request: Request, exc: MemoryNotFoundError) -> JSONResponse:
        logger.warning("Memory not found: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "Not Found",
                "detail": str(exc),
            },
        )

    @app.exception_handler(MemoryValidationError)
    async def memory_validation_error_handler(
        request: Request, exc: MemoryValidationError
    ) -> JSONResponse:
        logger.warning("Memory validation error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Validation Error",
                "detail": str(exc),
            },
        )

    # Register Routers
    app.include_router(health_router)
    app.include_router(chat_router, prefix="/api")
    app.include_router(conversations_router, prefix="/api")
    app.include_router(memories_router, prefix="/api")
    app.include_router(models_router, prefix="/api")

    return app


app = create_app()
