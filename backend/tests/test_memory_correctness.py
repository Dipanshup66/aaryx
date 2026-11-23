"""Regression tests for memory correctness (review findings #5, #6, #7 and #9).

#5  explicit-directive prefixes need word boundaries
#6  contradiction detection must use whole tokens and meaningful topic overlap
#7  active-memory view must stay fresh inside one extraction and cover more than 500 rows
#9  confidence thresholds must be coherent between settings, extractor and MemoryCreate
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.memory.database import create_database_engine, get_session_factory, init_db
from app.memory.extractor import (
    ExplicitMemoryParser,
    InferredMemoryExtractor,
    detect_contradiction,
)
from app.memory.models import Memory
from app.memory.repository import MemoryRepository
from app.memory.schemas import MemoryCreate
from app.memory.service import MemoryService
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
)

# ============================================================================ helpers


class ScriptedExtractionProvider(LLMProvider):
    """Returns a fixed JSON extraction payload."""

    def __init__(self, items: list[dict]) -> None:
        self.payload = json.dumps(items)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content=self.payload, model="m", provider="ollama")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(delta=self.payload, model="m", provider="ollama")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="ollama")

    async def aclose(self) -> None:
        pass


def _item(content: str, confidence: float = 0.9, importance: float = 3.0, category: str = "fact"):
    return {
        "content": content,
        "category": category,
        "confidence": confidence,
        "importance": importance,
    }


def _mem(content: str, category: str = "fact", origin: str = "inferred") -> Memory:
    return Memory(content=content, category=category, origin=origin, status="active")


def _cand(content: str, category: str = "fact") -> MemoryCreate:
    return MemoryCreate(
        content=content,
        category=category,  # type: ignore[arg-type]
        origin="inferred",
        confidence=0.9,
        importance=3.0,
        status="candidate",
    )


@pytest_asyncio.fixture
async def temp_db(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncEngine, async_sessionmaker[AsyncSession]]]:
    engine = create_database_engine(f"sqlite+aiosqlite:///{tmp_path / 'm.db'}")
    await init_db(engine)
    try:
        yield engine, get_session_factory(engine)
    finally:
        await engine.dispose()


def _service(factory: async_sessionmaker[AsyncSession], items: list[dict]) -> MemoryService:
    service = MemoryService(session_factory=factory)
    service.extractor = InferredMemoryExtractor(provider=ScriptedExtractionProvider(items))
    return service


# ============================================================ #5 directive word boundaries


@pytest.mark.parametrize(
    "text",
    [
        "Notebook batteries are dead",
        "Remembering that day was hard",
        "Noted, thanks for the update",
        "Notes from the meeting were long",
        "Notable results this quarter were good",
        "Rememberthis is not a directive",
        "Always rememberthat is not a directive either",
        "Don't forgetful people annoy me",
        "Keep in minded people are rare",
        "Never forgetting names is my superpower",
        "Please notebook sales are up",
    ],
)
def test_directive_prefix_requires_word_boundary(text: str) -> None:
    matched, remainder = ExplicitMemoryParser.match_prefix(text)
    assert matched is False, remainder
    assert ExplicitMemoryParser.parse(text) is None


def test_that_keyword_requires_word_boundary() -> None:
    parsed = ExplicitMemoryParser.parse("Remember thatcher was a prime minister")
    assert parsed is not None
    assert parsed.content == "Thatcher was a prime minister."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Remember that I like tea", "I like tea."),
        ("Remember I like tea", "I like tea."),
        ("remember: I like tea", "I like tea."),
        ("REMEMBER THAT I like tea", "I like tea."),
        ("Please remember that I like tea", "I like tea."),
        ("Note that standup is at ten", "Standup is at ten."),
        ("Note: standup is at ten", "Standup is at ten."),
        ("Don't forget that I like tea", "I like tea."),
        ("Do not forget I like tea", "I like tea."),
        ("Always remember that I like tea", "I like tea."),
        ("Keep in mind that I like tea", "I like tea."),
        ("Make sure to remember that I like tea", "I like tea."),
        ("Never forget that I like tea", "I like tea."),
        ("Make a note that I like tea", "I like tea."),
        ("Take note: I like tea", "I like tea."),
        ("Remember\nthat I like tea", "I like tea."),
    ],
)
def test_real_directives_still_parse(text: str, expected: str) -> None:
    parsed = ExplicitMemoryParser.parse(text)
    assert parsed is not None, text
    assert parsed.content == expected
    assert parsed.origin == "explicit"


# ================================================================ #6 contradiction detection


@pytest.mark.parametrize(
    ("candidate", "existing", "why"),
    [
        (
            "User is frustrated with Python documentation",
            "User prefers Python for scripting.",
            "'rust' inside 'frustrated'",
        ),
        (
            "User trusts the Python community",
            "User prefers Python for scripting.",
            "'rust' inside 'trusts'",
        ),
        ("User has a good keyboard", "User prefers Python for scripting.", "'go' inside 'good'"),
        (
            "User went to Google headquarters",
            "User prefers Python for scripting.",
            "'go' inside 'google'",
        ),
        (
            "User is going to the gym daily",
            "User prefers Python for scripting.",
            "'go' inside 'going'",
        ),
        (
            "User switched to a new laptop last week",
            "User prefers dark mode UI.",
            "only 'user' shared",
        ),
        (
            "User moved to Berlin last year",
            "User prefers tabs for indentation.",
            "only 'user' shared",
        ),
        ("User no longer enjoys running", "User prefers dark mode UI.", "only 'user' shared"),
        (
            "User uses Python and Rust",
            "User prefers Python for scripting.",
            "additive, no negation",
        ),
    ],
)
def test_no_false_contradictions(candidate: str, existing: str, why: str) -> None:
    conflict, reason = detect_contradiction(_cand(candidate), [_mem(existing)])
    assert conflict is None, f"{why}: flagged as {reason}"


@pytest.mark.parametrize(
    ("candidate", "existing"),
    [
        ("User prefers light mode UI", "User prefers dark mode UI."),
        ("User now prefers spaces for indentation", "User prefers tabs for indentation."),
        ("User switched from Python to Rust", "User prefers Python for scripting."),
        ("User is now vegan", "User eats meat regularly."),
        ("User no longer uses Docker for deployment", "User uses Docker for deployment."),
    ],
)
def test_real_contradictions_still_detected(candidate: str, existing: str) -> None:
    conflict, reason = detect_contradiction(_cand(candidate), [_mem(existing)])
    assert conflict is not None
    assert reason


def test_negation_with_same_category_topic_overlap_detected() -> None:
    conflict, _ = detect_contradiction(
        _cand("User stopped using Postgres", category="preference"),
        [_mem("User uses Postgres daily.", category="preference")],
    )
    assert conflict is not None


def test_non_active_memories_are_ignored() -> None:
    old = _mem("User prefers dark mode UI.")
    old.status = "superseded"
    conflict, _ = detect_contradiction(_cand("User prefers light mode UI"), [old])
    assert conflict is None


# ================================================================ #7 stale lists / 500-row cap


async def test_duplicate_suggestions_in_one_extraction_insert_once(temp_db) -> None:
    _, factory = temp_db
    service = _service(
        factory,
        [_item("User prefers Neovim."), _item("  USER PREFERS NEOVIM!  ", confidence=0.95)],
    )

    stored = await service.extract_and_store_memories("I keep using Neovim for everything.")

    pending = await service.list_memories(status="candidate")
    assert pending.total == 1
    assert len(stored) == 1 and stored[0].status == "candidate"
    assert (await service.list_memories(status="active")).total == 0


async def test_inferred_suggestions_never_supersede_an_active_memory(temp_db) -> None:
    _, factory = temp_db
    service = _service(factory, [])
    old = await service.create_memory(
        MemoryCreate(
            content="User prefers dark mode UI.",
            category="preference",
            origin="explicit",
            confidence=1.0,
            importance=2.0,
            status="active",
        )
    )
    service.extractor = InferredMemoryExtractor(
        provider=ScriptedExtractionProvider(
            [
                _item("User prefers light mode UI.", confidence=0.99, importance=3.0),
                _item("User prefers light theme in the evening.", confidence=0.99, importance=3.0),
            ]
        )
    )

    stored = await service.extract_and_store_memories("Light mode is easier on my eyes lately.")

    assert stored and all(m.status == "candidate" for m in stored)
    untouched = await service.get_memory(old.id)
    assert untouched is not None and untouched.status == "active"
    assert untouched.superseded_by_id is None
    assert stored[0].metadata_json.get("conflicts_with") == old.id
    assert [m.id for m in (await service.list_memories(status="active")).memories] == [old.id]


async def test_pending_suggestions_do_not_conflict_with_each_other_in_a_batch(temp_db) -> None:
    _, factory = temp_db
    service = _service(
        factory,
        [
            _item("User prefers light mode UI.", importance=3.0),
            _item("User prefers dark mode UI again.", importance=3.0),
        ],
    )

    stored = await service.extract_and_store_memories("Switching themes a lot this week.")

    assert len(stored) == 2 and all(m.status == "candidate" for m in stored)
    assert (await service.list_memories(status="active")).total == 0
    assert (await service.list_memories(status="candidate")).total == 2


async def test_dedup_and_contradiction_cover_more_than_500_active_memories(temp_db) -> None:
    _, factory = temp_db
    async with factory() as session, session.begin():
        repo = MemoryRepository(session)
        # Oldest row first: it falls outside a newest-first LIMIT 500 window.
        await repo.create_memory(
            content="User uses Neovim daily.",
            category="preference",
            origin="inferred",
            confidence=0.9,
            importance=3.0,
            status="active",
        )
        await repo.create_memory(
            content="User prefers dark mode UI.",
            category="preference",
            origin="explicit",
            confidence=1.0,
            importance=4.0,
            status="active",
        )
        for i in range(520):
            await repo.create_memory(
                content=f"Filler record number {i} about gardening plot {i}.",
                category="fact",
                origin="inferred",
                confidence=0.9,
                importance=1.0,
                status="active",
            )

    service = MemoryService(session_factory=factory)
    before = (await service.list_memories(status="active", limit=1)).total
    assert before == 522

    # exact duplicate of the oldest row
    await service.extract_and_store_memories("Remember that User uses Neovim daily.")
    after_dup = (await service.list_memories(status="active", limit=1)).total
    assert after_dup == before, "duplicate of a row beyond the 500 cap was inserted"

    # contradiction with an old explicit memory beyond the cap is still detected
    service.extractor = InferredMemoryExtractor(
        provider=ScriptedExtractionProvider([_item("User prefers light mode UI.")])
    )
    stored = await service.extract_and_store_memories("Light mode looks nicer today.")
    assert stored[0].status == "candidate"
    assert stored[0].metadata_json.get("conflict_detected") is True


# ================================================================ #9 confidence thresholds


@pytest.mark.parametrize("candidate_min", [0.0, 0.60, 0.75, 0.90, 1.0])
def test_settings_accept_candidate_threshold(candidate_min: float) -> None:
    s = Settings(memory_candidate_min_confidence=candidate_min, _env_file=None)
    assert s.memory_candidate_min_confidence == candidate_min


def test_auto_store_setting_no_longer_exists() -> None:
    """Inferred memories are never auto-activated, so there is no auto-store threshold."""
    assert "memory_auto_store_min_confidence" not in Settings.model_fields


def test_memory_suggestions_default_on_and_can_be_disabled() -> None:
    assert Settings(_env_file=None).memory_suggestions_enabled is True
    assert (
        Settings(memory_suggestions_enabled=False, _env_file=None).memory_suggestions_enabled
        is False
    )


async def test_extractor_only_ever_produces_candidates() -> None:
    """Whatever the confidence, an inferred memory is a pending suggestion, never active."""
    settings = Settings.model_construct(
        memory_candidate_min_confidence=0.60,
        memory_extraction_model="m",
        ollama_base_url="http://localhost:11434",
        ollama_timeout=5.0,
    )
    provider = ScriptedExtractionProvider(
        [_item("User likes Go.", confidence=0.72), _item("User likes Rust.", confidence=1.0)]
    )
    extractor = InferredMemoryExtractor(settings=settings, provider=provider)

    results = await extractor.extract("I really like Go and Rust a lot these days.")

    assert len(results) == 2
    assert {r.status for r in results} == {"candidate"}
    assert {r.origin for r in results} == {"inferred"}


async def test_extractor_discards_below_candidate_minimum() -> None:
    settings = Settings.model_construct(
        memory_candidate_min_confidence=0.90,
        memory_extraction_model="m",
        ollama_base_url="http://localhost:11434",
        ollama_timeout=5.0,
    )
    provider = ScriptedExtractionProvider(
        [_item("User likes Go.", confidence=0.85), _item("User likes Rust.", confidence=0.95)]
    )
    extractor = InferredMemoryExtractor(settings=settings, provider=provider)

    results = await extractor.extract("I really like Go and Rust a lot these days.")

    assert [r.content for r in results] == ["User likes Rust."]  # 0.85 < candidate minimum
    assert results[0].status == "candidate"
