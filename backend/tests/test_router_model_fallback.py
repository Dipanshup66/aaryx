"""Regression tests for model-identifier handling during primary -> fallback routing.

Covers review finding #2: a per-request `model` override chosen for the primary provider
(e.g. an Ollama tag) must not be blindly forwarded to an incompatible cloud provider.
"""

from collections.abc import AsyncIterator

import pytest

from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderConnectionError,
    ProviderHealth,
)
from app.providers.factory import register_provider, reset_provider_registry
from app.providers.router import ProviderRouter, is_model_compatible


class RecordingProvider(LLMProvider):
    """Records the model override it receives; optionally fails with a connection error."""

    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.seen_models: list[str | None] = []

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.seen_models.append(request.model)
        if self.fail:
            raise ProviderConnectionError(f"{self.name} down", provider=self.name)
        return ChatResponse(content="ok", model=request.model or f"{self.name}-default")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self.seen_models.append(request.model)
        if self.fail:
            raise ProviderConnectionError(f"{self.name} down", provider=self.name)
        yield ChatStreamChunk(delta="ok", model=request.model or f"{self.name}-default")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider=self.name)

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _registry():
    reset_provider_registry()
    yield
    reset_provider_registry()


def _router(
    primary: str, fallback: str
) -> tuple[ProviderRouter, RecordingProvider, RecordingProvider]:
    p = RecordingProvider(primary, fail=True)
    f = RecordingProvider(fallback)
    register_provider(primary, lambda s: p)
    register_provider(fallback, lambda s: f)
    router = ProviderRouter(
        primary_provider=primary,
        fallback_provider=fallback,
        fallback_enabled=True,
        max_retries=0,
        allow_cloud_fallback=True,
    )
    return router, p, f


def _req(model: str | None) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content="hi")], model=model)


@pytest.mark.parametrize("fallback", ["openai", "anthropic", "gemini"])
async def test_generate_ollama_model_not_sent_to_cloud_fallback(fallback: str) -> None:
    router, primary, fb = _router("ollama", fallback)
    resp = await router.generate(_req("qwen3.5:4b"))
    assert primary.seen_models == ["qwen3.5:4b"]  # primary still honours the override
    assert fb.seen_models == [None]  # fallback uses its own configured default
    assert resp.model == f"{fallback}-default"


@pytest.mark.parametrize("fallback", ["openai", "anthropic", "gemini"])
async def test_stream_ollama_model_not_sent_to_cloud_fallback(fallback: str) -> None:
    router, primary, fb = _router("ollama", fallback)
    chunks = [c async for c in router.stream(_req("qwen3.5:4b"))]
    assert primary.seen_models == ["qwen3.5:4b"]
    assert fb.seen_models == [None]
    assert chunks[0].model == f"{fallback}-default"


async def test_ollama_tag_that_looks_like_openai_name_is_not_forwarded() -> None:
    """`gpt-oss:20b` is an Ollama tag (contains ':'), not an OpenAI model."""
    router, _, fb = _router("ollama", "openai")
    await router.generate(_req("gpt-oss:20b"))
    assert fb.seen_models == [None]


async def test_unsupported_unknown_identifier_is_not_forwarded_to_cloud() -> None:
    router, _, fb = _router("ollama", "anthropic")
    await router.generate(_req("my-custom-finetune"))
    assert fb.seen_models == [None]


async def test_compatible_model_override_is_preserved() -> None:
    """Anthropic primary -> OpenAI fallback: a gpt-* override is compatible and kept."""
    router, _, fb = _router("anthropic", "openai")
    await router.generate(_req("gpt-4o"))
    assert fb.seen_models == ["gpt-4o"]


async def test_cloud_model_not_sent_to_ollama_fallback_but_local_tag_is() -> None:
    router, _, fb = _router("openai", "ollama")
    await router.generate(_req("gpt-4o"))
    assert fb.seen_models == [None]

    router2, _, fb2 = _router("openai", "ollama")
    await router2.generate(_req("llama3.2:3b"))
    assert fb2.seen_models == ["llama3.2:3b"]


async def test_no_override_stays_no_override() -> None:
    router, _, fb = _router("ollama", "openai")
    await router.generate(_req(None))
    assert fb.seen_models == [None]


async def test_original_request_object_is_not_mutated() -> None:
    router, _, _ = _router("ollama", "openai")
    request = _req("qwen3.5:4b")
    await router.generate(request)
    assert request.model == "qwen3.5:4b"


def test_is_model_compatible_matrix() -> None:
    assert is_model_compatible("openai", "gpt-4o-mini")
    assert is_model_compatible("openai", "o3-mini")
    assert is_model_compatible("openai", "ft:gpt-4o-mini:acme::abc123")
    assert not is_model_compatible("openai", "qwen3.5:4b")
    assert not is_model_compatible("openai", "claude-3-5-sonnet-20241022")
    assert not is_model_compatible("openai", "gpt-oss:20b")
    assert is_model_compatible("anthropic", "claude-3-5-sonnet-20241022")
    assert not is_model_compatible("anthropic", "gpt-4o")
    assert is_model_compatible("gemini", "gemini-1.5-flash")
    assert is_model_compatible("gemini", "models/gemini-1.5-pro")
    assert not is_model_compatible("gemini", "qwen3.5:4b")
    assert is_model_compatible("ollama", "qwen3.5:4b")
    assert is_model_compatible("ollama", "llama3")
    assert is_model_compatible("ollama", "gpt-oss:20b")
    assert not is_model_compatible("ollama", "gpt-4o")
    assert not is_model_compatible("ollama", "claude-3-5-sonnet-20241022")
    assert not is_model_compatible("openai", "")
