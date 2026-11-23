"""Shared test fixtures."""

from typing import Any

import pytest

from app.memory.extractor import InferredMemoryExtractor
from app.providers.ollama import OllamaProvider


@pytest.fixture(autouse=True)
def _never_call_real_ollama_for_memory_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat turns now trigger memory extraction in the background.

    A MemoryService built without an injected extractor would talk to a real Ollama on
    localhost (if the developer has one running) and send test conversations to it. Block
    exactly that case; extractors given a scripted/mock provider are untouched.
    """
    original = InferredMemoryExtractor.extract

    async def guarded_extract(self: InferredMemoryExtractor, *args: Any, **kwargs: Any) -> Any:
        if type(self.provider) is OllamaProvider:
            return []
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(InferredMemoryExtractor, "extract", guarded_extract)
