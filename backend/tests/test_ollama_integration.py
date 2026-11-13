"""Integration tests connecting to live local Ollama instance.

These tests require an active Ollama daemon at 127.0.0.1:11434 with qwen3.5:4b.
They are marked with 'integration' and can be filtered via `-m "not integration"`.
"""

import httpx
import pytest

from app.core.config import get_settings
from app.providers.base import ChatMessage, ChatRequest
from app.providers.ollama import OllamaProvider


async def is_ollama_live(base_url: str) -> bool:
    """Quick probe to determine if local Ollama daemon is reachable."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(f"{base_url}/api/tags")
            return res.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_ollama_generate():
    """Verify live generation against local Ollama daemon."""
    settings = get_settings()
    if not await is_ollama_live(settings.ollama_base_url):
        pytest.skip(f"Ollama daemon unreachable at {settings.ollama_base_url}")

    async with OllamaProvider() as provider:
        request = ChatRequest(
            messages=[
                ChatMessage(
                    role="user", content="Respond with the single word 'HELLO' and nothing else."
                )
            ],
            model=settings.ollama_model,
            temperature=0.1,
        )
        response = await provider.generate(request)

        assert response.content is not None
        assert len(response.content.strip()) > 0
        assert settings.ollama_model in response.model
        assert response.finish_reason is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_ollama_stream():
    """Verify live token streaming against local Ollama daemon."""
    settings = get_settings()
    if not await is_ollama_live(settings.ollama_base_url):
        pytest.skip(f"Ollama daemon unreachable at {settings.ollama_base_url}")

    async with OllamaProvider() as provider:
        request = ChatRequest(
            messages=[ChatMessage(role="user", content="Count from 1 to 3 separated by spaces.")],
            model=settings.ollama_model,
            temperature=0.1,
        )

        tokens = []
        async for chunk in provider.stream(request):
            if chunk.delta:
                tokens.append(chunk.delta)

        full_reply = "".join(tokens).strip()
        assert len(full_reply) > 0
