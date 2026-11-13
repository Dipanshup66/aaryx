"""API route modules and dependencies."""

from app.api.deps import get_orchestrator, get_provider
from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.api.routes_models import router as models_router

__all__ = [
    "chat_router",
    "get_orchestrator",
    "get_provider",
    "health_router",
    "models_router",
]
