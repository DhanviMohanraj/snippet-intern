"""Asynchronous natural-language requirement extraction layer.

Translates conversational user input into a validated RequirementsDelta model
using the injected LLM client and deterministic normalization.
"""

from __future__ import annotations

import re

from app.agent.normalize import normalize
from app.llm.client import LLMClient
from app.llm.prompts import build_extraction_messages
from app.models.requirements import HardConstraints, Message, Requirements, RequirementsDelta

MAX_MESSAGE_LENGTH = 1000

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_RESULT_NOUNS = (
    r"(?:options?|recommendations?|products?|choices?|picks?|models?|suggestions?|"
    r"laptops?|notebooks?|phones?|smartphones?|mobiles?|headphones?|earbuds?|"
    r"tablets?|monitors?|displays?)"
)
_RESULT_NOUNS_PLURAL = (
    r"(?:options|recommendations|products|choices|picks|models|suggestions|"
    r"laptops|notebooks|phones|smartphones|mobiles|headphones|earbuds|"
    r"tablets|monitors|displays)"
)

# Previously digits only, so "top two monitors" (the exact phrasing from the
# failing test) was not recognised and the pipeline fell back to 5 results.
# Three forms, ordered from most to least explicit:
#   1. "top 2", "best three laptops", "top-2"           (noun optional)
#   2. "show me 2 options", "recommend two laptops"     (noun required)
#   3. "2 options", "three monitors"                    (plural noun required,
#      so "one laptop for college" is not read as a result count)
RE_RESULT_COUNT = re.compile(
    rf"\b(?:top|best)[\s-]+{_NUM}\b(?!\s*(?:gb|tb|hz|inch|in\b|k\b|mp\b))|"
    rf"\b(?:give\s+me|show\s+me|recommend|suggest|list|only|just)\s+(?:me\s+)?(?:the\s+)?(?:top\s+|best\s+)?"
    rf"{_NUM}\s+(?:best\s+|good\s+|different\s+)?{_RESULT_NOUNS}\b|"
    rf"\b{_NUM}\s+(?:best\s+|good\s+|different\s+)?{_RESULT_NOUNS_PLURAL}\b",
    re.IGNORECASE,
)


def _parse_count(token: str | None) -> int | None:
    """Convert a digit string or number word into an int, or None."""
    if not token:
        return None
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)

RE_VERIFIED_ONLY = re.compile(
    r"\bonly\s+recommend\s+(?:products\s+if|if)|must\s+be\s+verified\b",
    re.IGNORECASE,
)


def _apply_deterministic_fallbacks(message: str, delta: RequirementsDelta) -> RequirementsDelta:
    """Enhance extracted delta with deterministic rule-based checks."""
    msg_lower = message.lower()
    updated = delta.model_copy(deep=True)

    # 1. Deterministic requested result count
    if updated.requested_result_count is None:
        for count_match in RE_RESULT_COUNT.finditer(message):
            # "I have 2 monitors already" describes ownership, not a result count.
            preceding = message[max(0, count_match.start() - 20):count_match.start()].lower()
            if re.search(r"\b(?:have|has|own|owned|got|bought|already)\s*$", preceding):
                continue
            token = next((g for g in count_match.groups() if g), None)
            cnt = _parse_count(token)
            if cnt is not None and 1 <= cnt <= 20:
                updated.requested_result_count = cnt
                break

    # 2. Strict screen size vs soft preference for monitors
    if "only show" in msg_lower and re.search(r"(\d+(?:\.\d+)?)[-\s]*(?:inch|in|\")\s*monitors?", msg_lower):
        m = re.search(r"(\d+(?:\.\d+)?)[-\s]*(?:inch|in|\")\s*monitors?", msg_lower)
        if m:
            if updated.hard is None:
                updated.hard = HardConstraints()
            updated.hard.screen_size_in = float(m.group(1))
    elif re.search(r"(\d+(?:\.\d+)?)[-\s]*(?:inch|in|\")\s*monitors?\s+only", msg_lower):
        m = re.search(r"(\d+(?:\.\d+)?)[-\s]*(?:inch|in|\")\s*monitors?\s+only", msg_lower)
        if m:
            if updated.hard is None:
                updated.hard = HardConstraints()
            updated.hard.screen_size_in = float(m.group(1))

    # 3. Soft screen size preference (e.g. "I want a 24-inch display")
    if updated.hard is None or updated.hard.screen_size_in is None:
        # Allow up to three adjectives before the size: "prefer a sharp 24-inch display"
        # previously failed this rule, leaving the size preference to the LLM alone.
        m_soft = re.search(
            r"(?:want|prefer|like|look(?:ing)?\s+for)\s+(?:an?\s+)?(?:[a-z]+\s+){0,3}?"
            r"(\d+(?:\.\d+)?)[-\s]*(?:inch(?:es)?|in\b|\")",
            msg_lower,
        )
        if m_soft:
            target_sz = float(m_soft.group(1))
            if updated.hard is None:
                updated.hard = HardConstraints()
            if updated.hard.target_screen_in is None:
                updated.hard.target_screen_in = target_sz
            if updated.preferences is None:
                updated.preferences = {}
            if "screen_size" not in updated.preferences:
                updated.preferences["screen_size"] = 0.85

    # 3b. Explicitly unwanted refresh rate ("don't need a high refresh rate").
    # Ensures the negation is never stored as a positive refresh_rate preference.
    if re.search(
        r"(?:don'?t|do\s+not|doesn'?t|no)\s+(?:really\s+)?(?:need|want|care\s+about|require)\s+"
        r"(?:a\s+|an\s+|for\s+)?(?:high\s+|fast\s+)?(?:refresh|hz\b|\d+\s*hz)"
        r"|no\s+need\s+for\s+(?:a\s+)?(?:high\s+)?refresh",
        msg_lower,
    ):
        if updated.preferences and "refresh_rate" in updated.preferences:
            updated.preferences = {k: v for k, v in updated.preferences.items() if k != "refresh_rate"}
        if "refresh_rate" not in updated.removed_fields:
            updated.removed_fields = [*updated.removed_fields, "refresh_rate"]

    # 4. Mandatory verification requirements ("only recommend products if that feature is verified")
    if RE_VERIFIED_ONLY.search(msg_lower):
        # Only treat as a hard exclusion if the user expressed the feature as mandatory, not "nice to have" or "preferred"
        verified_feats: list[str] = list(updated.hard.requires_verified_features) if (updated.hard and updated.hard.requires_verified_features) else []
        if any(term in msg_lower for term in ("eye-comfort", "eye comfort", "low blue light", "low-blue-light")):
            if not any(opt in msg_lower for opt in ("nice to have", "would be nice", "prefer")):
                if "eye_comfort" not in verified_feats:
                    verified_feats.append("eye_comfort")
        if any(term in msg_lower for term in ("flicker-free", "flicker free", "anti-flicker")):
            if not any(opt in msg_lower for opt in ("nice to have", "would be nice", "prefer")):
                if "flicker_free" not in verified_feats:
                    verified_feats.append("flicker_free")
        if verified_feats:
            if updated.hard is None:
                updated.hard = HardConstraints()
            updated.hard.requires_verified_features = verified_feats

    return updated


async def extract(
    message: str,
    current: Requirements,
    recent: list[Message],
    llm: LLMClient,
) -> RequirementsDelta:
    """Extract requirements delta from a user message.

    Performs:
        1. Input validation (non-empty, length bounded to <= 1000 chars).
        2. Prompt construction including current state context and <= 6 recent messages.
        3. LLM structured JSON completion using the injected LLMClient.
        4. Pure Python deterministic normalization on extracted values.
        5. Deterministic fallback checks on requested count and constraints.

    Args:
        message: The latest raw user text message.
        current: Current accumulated session Requirements.
        recent: Recent conversation history (sliced to at most the last 6 messages).
        llm: Injected asynchronous LLM client conforming to LLMClient protocol.

    Returns:
        A normalized and validated RequirementsDelta instance.

    Raises:
        ValueError: If the user message is empty, whitespace, or exceeds max length.
        LLMError: If LLM completion fails after permitted retries.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("User message cannot be empty or whitespace only.")

    cleaned_message = message.strip()
    if len(cleaned_message) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"User message exceeds maximum permitted length of {MAX_MESSAGE_LENGTH} characters "
            f"(received {len(cleaned_message)})."
        )

    # Build chat messages with prompt and context
    chat_messages = build_extraction_messages(
        message=cleaned_message,
        current=current,
        recent=recent,
    )

    # Invoke LLM for structured JSON output validated against RequirementsDelta
    raw_delta = await llm.complete_json(
        messages=chat_messages,
        schema=RequirementsDelta,
        max_tokens=1500,
        temperature=0.1,
    )

    # Apply pure Python deterministic normalization
    normalized_delta = normalize(raw_delta)

    # Apply deterministic checks for counts and verification requirements
    final_delta = _apply_deterministic_fallbacks(cleaned_message, normalized_delta)

    return final_delta
