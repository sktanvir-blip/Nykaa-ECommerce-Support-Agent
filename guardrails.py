import re
import unicodedata


MAX_QUERY_CHARS = 4_000

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"
)

CARD_REFERENCE_RE = re.compile(
    r"\b(?:credit|debit|payment)\s*card\b.*?"
    r"\b(?:ending\s*in|number|no\.?)\s*[:#-]?\s*\d{4,19}\b",
    re.IGNORECASE,
)

CARD_PAN_RE = re.compile(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
)

PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+|previous\s+|earlier\s+)?instructions?\b",
    r"\bforget\s+(?:your\s+|all\s+)?instructions?\b",
    r"\bdisregard\s+(?:all\s+|previous\s+|earlier\s+)?(?:rules|instructions|policy)\b",
    r"\b(?:reveal|print|show|extract|leak|disclose)\b.*"
    r"\b(?:system\s+prompt|developer\s+message|hidden\s+instructions?|internal\s+config)\b",
    r"\b(?:system\s+prompt|developer\s+message|hidden\s+instructions?)\b.*"
    r"\b(?:reveal|print|show|extract|leak|disclose)\b",
    r"\bjailbreak\b",
    r"\bbypass\b.*\b(?:guardrails?|rules|safety|policy|restrictions?)\b",
    r"\bact\s+as\s+(?:the\s+)?system\b",
]


def canonicalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(
        character
        for character in normalized
        if character.isprintable() or character in "\n\t"
    )
    return " ".join(normalized.split())


def _passes_luhn(value: str) -> bool:
    digits = re.sub(r"\D", "", value)

    if not 13 <= len(digits) <= 19:
        return False

    total = 0
    reverse_digits = digits[::-1]

    for index, digit in enumerate(reverse_digits):
        number = int(digit)

        if index % 2 == 1:
            number *= 2

            if number > 9:
                number -= 9

        total += number

    return total % 10 == 0


def contains_payment_data(text: str) -> bool:
    if CARD_REFERENCE_RE.search(text):
        return True

    for candidate in CARD_PAN_RE.findall(text):
        if _passes_luhn(candidate):
            return True

    return False


def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text


def check_prompt_injection(text: str) -> dict:
    normalized = canonicalize_text(text).lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return {
                "allowed": False,
                "reason": "Potential prompt injection detected.",
            }

    return {
        "allowed": True,
        "reason": "No prompt injection detected.",
    }


def apply_input_guardrails(text: str) -> dict:
    if not isinstance(text, str):
        return {
            "allowed": False,
            "text": None,
            "reason": "Query must be text.",
        }

    normalized = canonicalize_text(text)

    if not normalized:
        return {
            "allowed": False,
            "text": None,
            "reason": "A non-empty query is required.",
        }

    if len(normalized) > MAX_QUERY_CHARS:
        return {
            "allowed": False,
            "text": None,
            "reason": f"Query exceeds the {MAX_QUERY_CHARS}-character limit.",
        }

    if contains_payment_data(normalized):
        return {
            "allowed": False,
            "text": None,
            "reason": (
                "For your security, do not send card or payment details here. "
                "Use an approved secure support channel."
            ),
        }

    masked_text = mask_pii(normalized)

    injection_result = check_prompt_injection(masked_text)

    if not injection_result["allowed"]:
        return {
            "allowed": False,
            "text": None,
            "reason": injection_result["reason"],
        }

    return {
        "allowed": True,
        "text": masked_text,
        "reason": "Input passed guardrails.",
    }