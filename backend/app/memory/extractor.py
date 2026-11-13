"""Selective memory extraction and deduplication engine for AARYX.

Implements explicit directive parsing, local-only inferred fact extraction,
conservative contradiction detection, and normalized exact deduplication.
"""

import json
import re

from app.brain.orchestrator import sanitize_reply_content
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.memory.exceptions import MemoryValidationError
from app.memory.models import Memory
from app.memory.schemas import (
    MemoryCategory,
    MemoryCreate,
    normalize_memory_content,
    validate_memory_content_security,
)
from app.providers.anthropic import AnthropicProvider
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    LLMProvider,
    ProviderError,
)
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider

logger = get_logger("memory.extractor")

# Explicit directive prefix patterns
_EXPLICIT_PREFIX_PATTERNS = [
    re.compile(r"^(?:please\s+)?always\s+remember\s*(?:that)?[:,\s]*", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?remember\s*(?:that)?[:,\s]*", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?note\s*(?:that)?[:,\s]*", re.IGNORECASE),
    re.compile(
        r"^(?:please\s+)?(?:don'?t|do\s+not)\s+forget\s*(?:that)?[:,\s]*",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:please\s+)?keep\s+in\s+mind\s*(?:that)?[:,\s]*", re.IGNORECASE),
    re.compile(
        r"^(?:please\s+)?make\s+sure\s+to\s+remember\s*(?:that)?[:,\s]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:never\s+forget|make\s+a\s+note|take\s+note)\s*(?:that)?[:,\s]*",
        re.IGNORECASE,
    ),
]

# Words that indicate trivial or conversational non-facts
_AMBIGUOUS_FACTS = {"it", "this", "that", "yes", "no", "ok", "okay", "sure", "fine"}

# Conversational greetings / fillers that should bypass inferred extraction
_GREETING_PATTERNS = [
    re.compile(
        r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))\b[!.?]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:thanks|thank\s+you|ok|okay|bye|goodbye)[!.?]?$", re.IGNORECASE),
    re.compile(r"^(?:how\s+are\s+you|what'?s\s+up)[!.?]?$", re.IGNORECASE),
]

_INFERRED_EXTRACTION_SYSTEM_PROMPT = """You are AARYX's Long-Term Memory Extraction Engine.
Your task is to analyze user conversation input and identify durable, long-term personal facts, user preferences, project details, or standing instructions.

STRICT RULES:
1. Extract ONLY facts about the USER or their projects.
2. DO NOT extract transient states (e.g., "I am eating lunch", "I am tired", "I am feeling sick today").
3. DO NOT extract greetings, questions, or conversational filler ("Hello", "How are you?").
4. DO NOT extract assistant-generated claims or advice.
5. DO NOT extract passwords, secrets, financial, or sensitive identifying data.
6. Rate confidence between 0.0 and 1.0 based on factual certainty.
7. Rate importance between 1.0 (minor detail) and 5.0 (core standing instruction).
8. Output ONLY a valid JSON array of objects. If no durable facts are present, output [].

Allowed categories: "preference", "fact", "project", "biography", "instruction".

JSON output format:
[
  {
    "content": "Concise factual statement about the user in third person (e.g., 'User prefers Python over JavaScript')",
    "category": "preference",
    "confidence": 0.85,
    "importance": 4.0
  }
]
"""

# Common opposing terms for conservative contradiction detection
_OPPOSING_PAIRS = [
    ({"dark mode", "dark theme"}, {"light mode", "light theme"}),
    ({"tabs", "tab indentation"}, {"spaces", "space indentation"}),
    ({"python"}, {"javascript", "typescript", "rust", "go"}),
    ({"vegetarian", "vegan"}, {"meat", "non-vegetarian", "carnivore"}),
]

_NEGATION_MARKERS = [
    "no longer",
    "not anymore",
    "switched to",
    "moved to",
    "instead of",
    "rather than",
    "now prefer",
    "stopped using",
]


class ExplicitMemoryParser:
    """Parses explicit user commands (e.g., 'Remember that...') into validated memory candidates."""

    @classmethod
    def match_prefix(cls, text: str) -> tuple[bool, str]:
        """Check if text starts with an explicit memory directive and extract the remainder.

        Returns (matched, fact_remainder).
        """
        trimmed = text.strip()
        for pattern in _EXPLICIT_PREFIX_PATTERNS:
            match = pattern.match(trimmed)
            if match:
                remainder = trimmed[match.end() :].strip()
                # Clean leading colon, dash, comma, or quotes
                remainder = re.sub(r"^[:\-,—\"'\s]+", "", remainder).strip()
                # Clean trailing quotes
                remainder = re.sub(r"[\"'\s]+$", "", remainder).strip()
                return True, remainder
        return False, ""

    @classmethod
    def categorize(cls, fact: str, original_text: str = "") -> MemoryCategory:
        """Heuristically assign a category to an explicit fact statement."""
        lower = fact.lower()
        orig_lower = original_text.lower()
        if any(
            w in lower
            for w in (
                "prefer",
                "preference",
                "like",
                "love",
                "favorite",
                "hate",
                "dislike",
                "vegan",
                "vegetarian",
            )
        ):
            return "preference"
        if any(
            w in orig_lower
            for w in ("always", "never", "do not", "must", "format", "concise", "rule")
        ) or any(w in lower for w in ("must", "always", "never")):
            return "instruction"
        if any(
            w in lower
            for w in (
                "project",
                "building",
                "repo",
                "repository",
                "codebase",
                "architecture",
                "aaryx",
                "fastapi",
            )
        ):
            return "project"
        if any(
            w in lower
            for w in (
                "i am",
                "i'm",
                "i live",
                "my name",
                "my role",
                "my job",
                "engineer",
                "developer",
            )
        ):
            return "biography"
        return "fact"

    @classmethod
    def parse(cls, text: str) -> MemoryCreate | None:
        """Parse explicit directives from user text, applying security and privacy validation.

        Returns a validated MemoryCreate object, or None if no valid explicit directive was present.
        Raises MemoryValidationError if prompt injection or sensitive secrets are detected.
        """
        matched, remainder = cls.match_prefix(text)
        if not matched:
            return None

        # Check for empty, too short, or ambiguous statements
        if not remainder or len(remainder) < 4 or remainder.lower() in _AMBIGUOUS_FACTS:
            logger.debug("Rejected empty or ambiguous explicit command remainder: '%s'", remainder)
            return None

        # Format fact cleanly: capitalize first letter and ensure ending punctuation
        formatted_fact = remainder[0].upper() + remainder[1:]
        if not formatted_fact.endswith((".", "!", "?")):
            formatted_fact += "."

        # Security and Privacy Validation
        try:
            validated_content = validate_memory_content_security(formatted_fact)
        except ValueError as exc:
            logger.warning("Rejected explicit memory directive due to security/privacy check")
            raise MemoryValidationError(str(exc)) from exc

        category = cls.categorize(validated_content, original_text=text)
        importance = 5.0 if category == "instruction" else 4.0

        return MemoryCreate(
            content=validated_content,
            category=category,
            origin="explicit",
            confidence=1.0,
            importance=importance,
            status="active",
        )


class InferredMemoryExtractor:
    """Extracts durable long-term memories using the local Ollama provider exclusively.

    Guarantees isolation from cloud providers even when cloud fallback is enabled.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the extractor with application settings and force local provider.

        Args:
            settings: Central configuration. If None, loaded via get_settings().
            provider: Optional LLMProvider. Must be OllamaProvider if supplied;
                      cloud providers are strictly rejected.
        """
        self.settings = settings or get_settings()

        # Enforce that extraction only uses local Ollama
        if provider is not None:
            if isinstance(provider, (OpenAIProvider, AnthropicProvider, GeminiProvider)):
                logger.warning(
                    "Cloud provider '%s' rejected for memory extraction. Forcing local OllamaProvider.",
                    type(provider).__name__,
                )
                self.provider = OllamaProvider(
                    base_url=self.settings.ollama_base_url,
                    default_model=self.settings.memory_extraction_model,
                    timeout=self.settings.ollama_timeout,
                )
            else:
                self.provider = provider
        else:
            self.provider = OllamaProvider(
                base_url=self.settings.ollama_base_url,
                default_model=self.settings.memory_extraction_model,
                timeout=self.settings.ollama_timeout,
            )

        self.model = self.settings.memory_extraction_model
        self.min_confidence = self.settings.memory_candidate_min_confidence
        self.auto_store_confidence = self.settings.memory_auto_store_min_confidence

    def is_conversational_filler(self, text: str) -> bool:
        """Determine if user input is trivial greeting or filler that contains no durable facts."""
        trimmed = text.strip()
        if len(trimmed) < 4:
            return True
        for pattern in _GREETING_PATTERNS:
            if pattern.match(trimmed):
                return True
        return False

    async def extract(
        self,
        user_prompt: str,
        assistant_reply: str | None = None,
    ) -> list[MemoryCreate]:
        """Extract durable memory candidates from user input using the local model.

        Returns a list of validated MemoryCreate objects (active or candidate).
        Never exposes raw model reasoning or leaks sensitive data.
        """
        if self.is_conversational_filler(user_prompt):
            logger.debug("Skipping extraction on conversational greeting/filler")
            return []

        turn_context = f"User: {user_prompt.strip()}"
        if assistant_reply:
            turn_context += f"\nAssistant: {sanitize_reply_content(assistant_reply).strip()}"

        chat_request = ChatRequest(
            messages=[
                ChatMessage(role="system", content=_INFERRED_EXTRACTION_SYSTEM_PROMPT),
                ChatMessage(role="user", content=turn_context),
            ],
            model=self.model,
            temperature=0.1,
        )

        try:
            logger.info("Running local memory extraction with model '%s'", self.model)
            response = await self.provider.generate(chat_request)
        except (ProviderError, TimeoutError, OSError, ValueError) as exc:
            logger.warning("Local memory extraction failed: %s", exc)
            return []

        # Sanitize reply content to strip any reasoning or <think> tags
        clean_content = sanitize_reply_content(response.content)

        return self._parse_json_response(clean_content)

    def _parse_json_response(self, raw_text: str) -> list[MemoryCreate]:
        """Safely parse JSON response from LLM output into validated MemoryCreate instances."""
        if not raw_text or not raw_text.strip():
            return []

        # Find JSON array in the text (handles markdown fences or surrounding commentary)
        json_match = re.search(r"\[[\s\S]*\]", raw_text)
        if not json_match:
            logger.debug("No JSON array pattern found in extraction output")
            return []

        json_str = json_match.group(0)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("Malformed JSON in memory extraction output: %s", exc)
            return []

        if not isinstance(parsed, list):
            logger.debug("Extraction output is not a JSON list")
            return []

        results: list[MemoryCreate] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            content = item.get("content")
            if not content or not isinstance(content, str) or not content.strip():
                continue

            # Check privacy & security validation on content
            try:
                validated_content = validate_memory_content_security(content.strip())
            except ValueError:
                logger.warning("Rejected inferred memory candidate failing security/privacy checks")
                continue

            # Validate category
            raw_cat = str(item.get("category", "fact")).strip().lower()
            allowed_categories = {"preference", "fact", "project", "biography", "instruction"}
            category: MemoryCategory = raw_cat if raw_cat in allowed_categories else "fact"  # type: ignore[assignment]

            # Validate confidence and importance
            try:
                confidence = float(item.get("confidence", 0.70))
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.70

            try:
                importance = float(item.get("importance", 3.0))
                importance = max(1.0, min(5.0, importance))
            except (ValueError, TypeError):
                importance = 3.0

            # Rule 3: Discard if confidence < 0.60
            if confidence < self.min_confidence:
                logger.debug(
                    "Discarding low-confidence candidate (confidence=%.2f < %.2f)",
                    confidence,
                    self.min_confidence,
                )
                continue

            # Lifecycle status assignment
            status = "active" if confidence >= self.auto_store_confidence else "candidate"

            try:
                candidate = MemoryCreate(
                    content=validated_content,
                    category=category,
                    origin="inferred",
                    confidence=confidence,
                    importance=importance,
                    status=status,
                )
                results.append(candidate)
            except ValueError as exc:
                logger.debug("Failed validation for inferred candidate: %s", exc)
                continue

        logger.info("Successfully extracted %d memory candidates", len(results))
        return results


def find_exact_duplicate(
    candidate_content: str,
    active_memories: list[Memory],
) -> Memory | None:
    """Find an existing active memory whose normalized content matches the candidate exactly.

    Returns the matching Memory instance or None.
    """
    normalized_candidate = normalize_memory_content(candidate_content)
    if not normalized_candidate:
        return None

    for mem in active_memories:
        if mem.status == "active" and normalize_memory_content(mem.content) == normalized_candidate:
            return mem
    return None


def detect_contradiction(
    candidate: MemoryCreate,
    existing_memories: list[Memory],
) -> tuple[Memory | None, str | None]:
    """Conservatively check if incoming memory candidate conflicts with an existing active memory.

    Returns (conflicting_memory, reason) if a contradiction is detected, otherwise (None, None).
    """
    cand_norm = normalize_memory_content(candidate.content)
    cand_lower = candidate.content.lower()

    # Check for direct negation or transition markers in candidate
    has_negation = any(m in cand_lower for m in _NEGATION_MARKERS)

    for mem in existing_memories:
        if mem.status != "active":
            continue

        mem_norm = normalize_memory_content(mem.content)

        # 1. Opposing value pairs (e.g. dark mode vs light mode, tabs vs spaces)
        for group_a, group_b in _OPPOSING_PAIRS:
            in_a = any(term in cand_norm for term in group_a)
            in_b = any(term in cand_norm for term in group_b)
            mem_in_a = any(term in mem_norm for term in group_a)
            mem_in_b = any(term in mem_norm for term in group_b)

            if (in_a and mem_in_b) or (in_b and mem_in_a):
                return mem, "opposing_value_detected"

        # 2. Negation/transition cues combined with significant token overlap
        if has_negation:
            cand_tokens = {w for w in cand_norm.split() if len(w) > 3}
            mem_tokens = {w for w in mem_norm.split() if len(w) > 3}
            overlap = cand_tokens.intersection(mem_tokens)
            if len(overlap) >= 1:
                return mem, "negation_and_topic_overlap"

        # 3. Direct category-scoped opposition with negation
        if mem.category == candidate.category and has_negation:
            cand_tokens = {w for w in cand_norm.split() if len(w) > 3}
            mem_tokens = {w for w in mem_norm.split() if len(w) > 3}
            overlap = cand_tokens.intersection(mem_tokens)
            if len(overlap) >= 2:
                return mem, "category_topic_contradiction"

    return None, None
