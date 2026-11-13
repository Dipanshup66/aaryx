"""Unit tests for LLM Provider Factory (backend/app/providers/factory.py)."""

from collections.abc import AsyncIterator

import pytest

from app.core.config import Settings
from app.main import app, lifespan
from app.providers.anthropic import AnthropicProvider
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderConfigurationError,
    ProviderHealth,
    UnsupportedProviderError,
)
from app.providers.factory import (
    create_provider,
    get_supported_providers,
    register_provider,
    reset_provider_registry,
)
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider


class DummyCustomProvider(LLMProvider):
    """Test provider implementation for verifying factory registration."""

    def __init__(self, tag: str = "custom") -> None:
        self.tag = tag
        self.closed = False

    async def generate(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content=f"Hello from {self.tag}", model="dummy-v1")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(delta="dummy")

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="dummy-v1")]

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider=self.tag)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure provider registry is clean before and after each test."""
    reset_provider_registry()
    yield
    reset_provider_registry()


@pytest.mark.asyncio
async def test_default_provider_selection_returns_ollama():
    """Verify default configuration without overrides instantiates OllamaProvider."""
    settings = Settings(_env_file=None)
    assert settings.ai_provider == "ollama"

    provider = create_provider(settings=settings)
    try:
        assert isinstance(provider, OllamaProvider)
        assert isinstance(provider, LLMProvider)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_explicit_provider_selection():
    """Verify explicit AI_PROVIDER='ollama' returns OllamaProvider."""
    settings = Settings(ai_provider="ollama", _env_file=None)
    provider = create_provider(settings=settings)
    try:
        assert isinstance(provider, OllamaProvider)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["ollama", "Ollama", "OLLAMA", "  ollama  "])
async def test_provider_selection_case_insensitive(provider_name: str):
    """Verify provider selection is case-insensitive and trims whitespace."""
    provider = create_provider(provider_name=provider_name)
    try:
        assert isinstance(provider, OllamaProvider)
    finally:
        await provider.aclose()


def test_unsupported_provider_raises_error():
    """Verify requesting an unregistered provider raises UnsupportedProviderError."""
    with pytest.raises(UnsupportedProviderError) as exc_info:
        create_provider(provider_name="unsupported_cloud_provider")

    assert "Unsupported AI provider: 'unsupported_cloud_provider'" in str(exc_info.value)
    assert exc_info.value.provider == "unsupported_cloud_provider"


@pytest.mark.asyncio
async def test_factory_returns_llm_provider_contract():
    """Verify factory returns an object implementing the full LLMProvider interface."""
    settings = Settings(_env_file=None)
    provider = create_provider(settings=settings)
    try:
        assert isinstance(provider, LLMProvider)
        assert callable(getattr(provider, "generate", None))
        assert callable(getattr(provider, "stream", None))
        assert callable(getattr(provider, "list_models", None))
        assert callable(getattr(provider, "health_check", None))
        assert callable(getattr(provider, "aclose", None))
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_provider_configuration_passed():
    """Verify provider configuration options (URL, model, timeout) are respected."""
    custom_settings = Settings(
        ollama_base_url="http://192.168.1.50:11434",
        ollama_model="qwen3:4b",
        ollama_timeout=45.0,
        _env_file=None,
    )
    provider = create_provider(settings=custom_settings)
    try:
        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://192.168.1.50:11434"
        assert provider.default_model == "qwen3:4b"
        assert provider.timeout == 45.0
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_fastapi_startup_uses_factory():
    """Verify FastAPI application lifespan initializes provider and orchestrator via factory."""
    async with lifespan(app):
        assert hasattr(app.state, "provider")
        assert hasattr(app.state, "orchestrator")
        assert isinstance(app.state.provider, LLMProvider)
        assert app.state.provider.primary_provider == "ollama"
        assert app.state.orchestrator.provider is app.state.provider
        assert not app.state.provider._client.is_closed

    # Verify shutdown closed the provider
    assert app.state.provider._client.is_closed


@pytest.mark.asyncio
async def test_custom_provider_registration_extension_point():
    """Verify new providers can be registered dynamically without modifying factory internals."""
    register_provider("dummy", lambda s: DummyCustomProvider(tag="custom-dummy"))

    assert "dummy" in get_supported_providers()

    provider = create_provider(provider_name="dummy")
    try:
        assert isinstance(provider, DummyCustomProvider)
        health = await provider.health_check()
        assert health.healthy is True
        assert health.provider == "custom-dummy"

        resp = await provider.generate(
            ChatRequest(messages=[ChatMessage(role="user", content="Hi")])
        )
        assert resp.content == "Hello from custom-dummy"
    finally:
        await provider.aclose()
        assert provider.closed is True


def test_empty_provider_registration_rejected():
    """Verify registering an empty or whitespace provider name raises ValueError."""
    with pytest.raises(ValueError, match="Provider name cannot be empty"):
        register_provider("   ", lambda s: DummyCustomProvider())


def test_supported_providers_listing():
    """Verify get_supported_providers returns sorted list of registered keys."""
    providers = get_supported_providers()
    assert set(providers) == {"anthropic", "gemini", "ollama", "openai"}
    assert isinstance(providers, list)
    assert providers == sorted(providers)


@pytest.mark.asyncio
async def test_cloud_providers_instantiation_with_keys():
    """Verify factory instantiates OpenAI, Anthropic, and Gemini when credentials are provided."""
    settings = Settings(
        openai_api_key="sk-openai-key",
        anthropic_api_key="sk-ant-key",
        gemini_api_key="AIzaSyGeminiKey",
        _env_file=None,
    )

    openai_p = create_provider(settings=settings, provider_name="openai")
    try:
        assert isinstance(openai_p, OpenAIProvider)
        assert openai_p.default_model == settings.openai_model
    finally:
        await openai_p.aclose()

    anthropic_p = create_provider(settings=settings, provider_name="anthropic")
    try:
        assert isinstance(anthropic_p, AnthropicProvider)
        assert anthropic_p.default_model == settings.anthropic_model
    finally:
        await anthropic_p.aclose()

    gemini_p = create_provider(settings=settings, provider_name="gemini")
    try:
        assert isinstance(gemini_p, GeminiProvider)
        assert gemini_p.default_model == settings.gemini_model
    finally:
        await gemini_p.aclose()


def test_cloud_provider_missing_key_raises_configuration_error():
    """Verify creating cloud providers without API keys raises ProviderConfigurationError."""
    settings = Settings(
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        _env_file=None,
    )

    with pytest.raises(ProviderConfigurationError) as exc_info:
        create_provider(settings=settings, provider_name="openai")
    assert "OpenAI API key is missing" in str(exc_info.value)
    assert exc_info.value.provider == "openai"

    with pytest.raises(ProviderConfigurationError) as exc_info:
        create_provider(settings=settings, provider_name="anthropic")
    assert "Anthropic API key is missing" in str(exc_info.value)
    assert exc_info.value.provider == "anthropic"

    with pytest.raises(ProviderConfigurationError) as exc_info:
        create_provider(settings=settings, provider_name="gemini")
    assert "Gemini API key is missing" in str(exc_info.value)
    assert exc_info.value.provider == "gemini"
