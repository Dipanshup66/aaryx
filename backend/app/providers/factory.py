"""Provider factory for runtime LLM provider instantiation and selection.

Allows dynamic selection of AI providers (e.g., local Ollama, future cloud providers)
via application configuration without coupling orchestration or API endpoints
to concrete provider implementations.
"""

from collections.abc import Callable

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.providers.anthropic import AnthropicProvider
from app.providers.base import LLMProvider, UnsupportedProviderError
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider

logger = get_logger("providers.factory")

# Type alias for provider builder callables
ProviderBuilder = Callable[[Settings], LLMProvider]

# Registry mapping lowercase provider names to their builder functions
_PROVIDER_REGISTRY: dict[str, ProviderBuilder] = {}


def register_provider(name: str, builder: ProviderBuilder) -> None:
    """Register a provider builder for a given provider identifier.

    Args:
        name: Case-insensitive unique provider name (e.g., 'ollama').
        builder: Factory callable accepting Settings and returning an LLMProvider.

    Raises:
        ValueError: If provider name is empty or whitespace.
    """
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Provider name cannot be empty")
    _PROVIDER_REGISTRY[normalized] = builder
    logger.debug("Registered LLM provider builder for '%s'", normalized)


def get_supported_providers() -> list[str]:
    """Return a sorted list of all currently registered provider names."""
    return sorted(_PROVIDER_REGISTRY.keys())


def _build_ollama_provider(settings: Settings) -> LLMProvider:
    """Instantiate the local Ollama provider from configuration settings."""
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        default_model=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )


def _build_openai_provider(settings: Settings) -> LLMProvider:
    """Instantiate the OpenAI provider from configuration settings."""
    return OpenAIProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        default_model=settings.openai_model,
        timeout=settings.openai_timeout,
    )


def _build_anthropic_provider(settings: Settings) -> LLMProvider:
    """Instantiate the Anthropic provider from configuration settings."""
    return AnthropicProvider(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
        default_model=settings.anthropic_model,
        timeout=settings.anthropic_timeout,
    )


def _build_gemini_provider(settings: Settings) -> LLMProvider:
    """Instantiate the Google Gemini provider from configuration settings."""
    return GeminiProvider(
        api_key=settings.gemini_api_key,
        base_url=settings.gemini_base_url,
        default_model=settings.gemini_model,
        timeout=settings.gemini_timeout,
    )


def reset_provider_registry() -> None:
    """Reset the registry to default built-in providers (useful for test isolation)."""
    _PROVIDER_REGISTRY.clear()
    register_provider("ollama", _build_ollama_provider)
    register_provider("openai", _build_openai_provider)
    register_provider("anthropic", _build_anthropic_provider)
    register_provider("gemini", _build_gemini_provider)


# Register built-in default providers
reset_provider_registry()


def create_provider(
    settings: Settings | None = None,
    provider_name: str | None = None,
) -> LLMProvider:
    """Instantiate and return an LLMProvider based on runtime configuration.

    Args:
        settings: Application settings. If None, resolved via get_settings().
        provider_name: Explicit provider override. If None, resolved from settings.ai_provider.

    Returns:
        An instance of LLMProvider ready for inference.

    Raises:
        UnsupportedProviderError: If the requested provider is not registered.
    """
    resolved_settings = settings or get_settings()
    raw_name = provider_name or resolved_settings.ai_provider
    target_name = raw_name.strip().lower()

    builder = _PROVIDER_REGISTRY.get(target_name)
    if builder is None:
        available = ", ".join(get_supported_providers()) or "none"
        logger.error(
            "Unsupported AI provider '%s' requested. Available providers: %s",
            target_name,
            available,
        )
        raise UnsupportedProviderError(
            f"Unsupported AI provider: '{target_name}'. Available providers: {available}",
            provider=target_name,
        )

    logger.info("Initializing LLM provider '%s'", target_name)
    return builder(resolved_settings)
