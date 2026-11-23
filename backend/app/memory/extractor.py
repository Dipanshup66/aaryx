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
    describe_error,
)
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider

logger = get_logger("memory.extractor")

# Explicit directive prefix patterns.
# Every keyword must end at a real word boundary so ordinary words that merely START with a
# keyword ("Notebook", "Remembering", "Noted", "Notes") are never mistaken for a directive.
# `_END` also rejects a following apostrophe or hyphen ("note-taking", "remember's").
_END = r"(?![\w'’-])"
_THAT = rf"(?:that{_END})?"
_TAIL = r"[:,\s]*"
_PLEASE_HI = r"(?:(?:kripya|please|plz)\s+)?"
_KI = rf"(?:ki{_END})?"
_HINGLISH_PREFIX_PATTERNS = [
    # "yaad rakho / yaad rakhna / yaad rakhiye / yaad rakh lo ki ..."
    re.compile(
        rf"^{_PLEASE_HI}yaad\s+rakh(?:o|na|iye|iyega|\s+lo|\s+lena){_END}\s*{_KI}{_TAIL}",
        re.IGNORECASE,
    ),
    # "dhyan rakho / dhyan mein rakhna ki ..."
    re.compile(
        rf"^{_PLEASE_HI}dhyan\s+(?:mein\s+)?rakh(?:o|na|iye|iyega){_END}\s*{_KI}{_TAIL}",
        re.IGNORECASE,
    ),
    # "mat bhoolna / bhoolna mat / mat bhoolo ki ..."
    re.compile(
        rf"^{_PLEASE_HI}(?:mat\s+bhool(?:na|o|iye|iyega)|bhool(?:na|o)\s+mat){_END}\s*{_KI}{_TAIL}",
        re.IGNORECASE,
    ),
    # "note kar lo / note kar lena / note kar do ki ..."
    re.compile(
        rf"^{_PLEASE_HI}note\s+kar\s+(?:lo|lena|le|do|dena){_END}\s*{_KI}{_TAIL}",
        re.IGNORECASE,
    ),
]

_EXPLICIT_PREFIX_PATTERNS = [
    *_HINGLISH_PREFIX_PATTERNS,
    re.compile(rf"^(?:please\s+)?always\s+remember{_END}\s*{_THAT}{_TAIL}", re.IGNORECASE),
    re.compile(rf"^(?:please\s+)?remember{_END}\s*{_THAT}{_TAIL}", re.IGNORECASE),
    re.compile(rf"^(?:please\s+)?note{_END}\s*{_THAT}{_TAIL}", re.IGNORECASE),
    re.compile(
        rf"^(?:please\s+)?(?:don'?t|do\s+not)\s+forget{_END}\s*{_THAT}{_TAIL}",
        re.IGNORECASE,
    ),
    re.compile(rf"^(?:please\s+)?keep\s+in\s+mind{_END}\s*{_THAT}{_TAIL}", re.IGNORECASE),
    re.compile(
        rf"^(?:please\s+)?make\s+sure\s+to\s+remember{_END}\s*{_THAT}{_TAIL}",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:never\s+forget|make\s+a\s+note|take\s+note){_END}\s*{_THAT}{_TAIL}",
        re.IGNORECASE,
    ),
]

# A single turn can queue at most this many suggestions (guards against a runaway model).
_MAX_SUGGESTIONS_PER_TURN = 5

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

# Common opposing terms for conservative contradiction detection.
# Terms are matched as whole words / phrases (never as substrings of other words).
_OPPOSING_PAIRS: list[tuple[frozenset[str], frozenset[str]]] = [
    (frozenset({"dark mode", "dark theme"}), frozenset({"light mode", "light theme"})),
    (frozenset({"tabs", "tab indentation"}), frozenset({"spaces", "space indentation"})),
    (
        frozenset({"python"}),
        frozenset({"javascript", "typescript", "rust", "go", "golang"}),
    ),
    (
        frozenset({"vegetarian", "vegan"}),
        frozenset({"meat", "non-vegetarian", "carnivore"}),
    ),
]

# Ambiguous common English words that are only treated as a topic term when written exactly
# as the proper name in the original text ("Go" the language, not "go home").
_CASE_SENSITIVE_TERMS = {"go": "Go"}

_TOKEN_RE = re.compile(r"[a-z0-9+#ऀ-ॿ]+")

# Words that carry no topic information: the generic "User <verb>" frame, auxiliaries, and the
# vocabulary of negation/transition markers themselves.
_STOPWORDS = frozenset(
    {
        "user", "users", "the", "and", "for", "with", "from", "that", "this", "they", "their",
        "them", "have", "has", "had", "been", "being", "are", "was", "were", "will", "would",
        "can", "could", "should", "not", "but", "into", "onto", "over", "very", "also", "just",
        "really", "when", "while", "about", "some", "much", "more", "most", "than", "then",
        "uses", "use", "using", "used", "prefers", "prefer", "preferred", "likes", "like",
        "loves", "love", "enjoys", "enjoy", "wants", "want", "now", "longer", "anymore",
        "stopped", "switched", "moved", "instead", "rather", "always", "never", "often",
        "daily", "usually", "currently", "mostly", "mainly", "regularly",
    }
)  # fmt: skip

_NEGATION_MARKERS = [
    "no longer",
    "not anymore",
    "switched to",
    "switched from",
    "moved to",
    "moved from",
    "migrated to",
    "migrated from",
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

        Returns a list of validated MemoryCreate objects, always with status 'candidate'.
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
            logger.warning("Local memory extraction failed: %s", describe_error(exc))
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

            try:
                # An inferred memory is only ever a suggestion: it becomes active solely
                # through explicit user approval, whatever its confidence.
                candidate = MemoryCreate(
                    content=validated_content,
                    category=category,
                    origin="inferred",
                    confidence=confidence,
                    importance=importance,
                    status="candidate",
                )
                results.append(candidate)
            except ValueError as exc:
                # Log only the validation messages: str(exc) would echo the memory content.
                reasons = [e.get("msg", "invalid") for e in getattr(exc, "errors", list)()]
                logger.warning(
                    "Dropped inferred memory candidate failing validation: %s",
                    "; ".join(reasons) or exc.__class__.__name__,
                )
                continue

        if len(results) > _MAX_SUGGESTIONS_PER_TURN:
            logger.warning(
                "Extraction returned %d suggestions; keeping the first %d",
                len(results),
                _MAX_SUGGESTIONS_PER_TURN,
            )
            results = results[:_MAX_SUGGESTIONS_PER_TURN]

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


def _find_group_terms(group: frozenset[str], norm: str, raw: str) -> bool:
    """Return True if any term of ``group`` occurs as a whole word/phrase in the text."""
    for term in group:
        if term in _CASE_SENSITIVE_TERMS:
            # Ambiguous English words ("go") only count when written as the proper name.
            if re.search(
                rf"(?<![A-Za-z0-9]){re.escape(_CASE_SENSITIVE_TERMS[term])}(?![A-Za-z0-9])", raw
            ):
                return True
            continue
        if re.search(rf"(?<![a-z0-9])(?<!non-){re.escape(term)}(?![a-z0-9])", norm):
            return True
    return False


def meaningful_tokens(norm: str) -> set[str]:
    """Whole-word content tokens, excluding stop words and the generic 'User ... prefers' frame."""
    return {t for t in _TOKEN_RE.findall(norm) if len(t) > 2 and t not in _STOPWORDS}


_meaningful_tokens = meaningful_tokens  # kept for existing internal callers


def _stem(token: str) -> str:
    """Very light English stemmer so "gardens"/"garden" and "running"/"runners"/"run" match."""
    if len(token) <= 3 or not token.isascii():
        return token
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    for suffix in ("ing", "er", "ed"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            token = token[: -len(suffix)]
            break
    if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "aeiouls":
        token = token[:-1]
    return token


def stemmed_tokens(norm: str) -> set[str]:
    """Meaningful tokens reduced to a rough stem; used to match a message to memories."""
    return {_stem(t) for t in meaningful_tokens(norm)}


def detect_contradiction(
    candidate: MemoryCreate,
    existing_memories: list[Memory],
) -> tuple[Memory | None, str | None]:
    """Conservatively check if incoming memory candidate conflicts with an existing active memory.

    Matching is token-based: opposing terms must appear as whole words/phrases, and topic
    overlap is computed over meaningful content words only (never over the generic subject
    "user" or stop words). No embeddings or fuzzy semantics are involved.

    Returns (conflicting_memory, reason) if a contradiction is detected, otherwise (None, None).
    """
    cand_norm = normalize_memory_content(candidate.content)
    cand_raw = candidate.content

    has_negation = any(m in cand_norm for m in _NEGATION_MARKERS)
    cand_tokens = _meaningful_tokens(cand_norm)

    for mem in existing_memories:
        if mem.status != "active":
            continue

        mem_norm = normalize_memory_content(mem.content)
        mem_raw = mem.content

        # 1. Opposing value pairs (e.g. dark mode vs light mode, tabs vs spaces).
        for group_a, group_b in _OPPOSING_PAIRS:
            c_a = _find_group_terms(group_a, cand_norm, cand_raw)
            c_b = _find_group_terms(group_b, cand_norm, cand_raw)
            m_a = _find_group_terms(group_a, mem_norm, mem_raw)
            m_b = _find_group_terms(group_b, mem_norm, mem_raw)

            if m_a == m_b:
                continue  # memory is neutral or mentions both sides: nothing to oppose
            if c_a and c_b:
                # Candidate names both sides: only a contradiction when it signals a change
                # ("switched from Python to Rust"), not when it is additive ("Python and Rust").
                if has_negation:
                    return mem, "opposing_value_detected"
                continue
            if (c_a and m_b) or (c_b and m_a):
                return mem, "opposing_value_detected"

        # 2/3. Negation / transition cue combined with topic overlap on content words.
        if has_negation:
            overlap = cand_tokens & _meaningful_tokens(mem_norm)
            if len(overlap) >= 2:
                return mem, "negation_and_topic_overlap"
            if len(overlap) == 1 and mem.category == candidate.category:
                return mem, "category_topic_contradiction"

    return None, None
