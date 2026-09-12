import math
import re
from dataclasses import dataclass
from enum import Enum


MAX_INPUT_TOKENS = 1_000
COST_PER_1000_TOKENS = 0.002


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class BudgetDecision:
    allowed: bool
    estimated_tokens: int
    estimated_cost: float
    reason: str


@dataclass
class GovernanceDecision:
    allowed: bool
    risk_level: RiskLevel
    reason: str


HIGH_RISK_PATTERNS = [
    r"\bdelete\s+(?:my\s+)?account\b",
    r"\bchange\s+(?:my\s+)?bank\s+account\b",
    r"\bchange\s+(?:my\s+)?payment\s+details\b",
    r"\brefund\b.*\b(?:another|different)\s+account\b",
    r"\bcancel\s+all\s+(?:my\s+)?orders\b",
    r"\breset\s+(?:my\s+)?password\b",
    r"\bchange\s+(?:my\s+)?email\b",
]

MEDIUM_RISK_PATTERNS = [
    r"\breturn\b",
    r"\bexchange\b",
    r"\brefund\b",
    r"\bcomplaint\b",
    r"\bescalat",
    r"\bpayment\b",
    r"\bcancel\b",
    r"\baddress\b",
]


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4) if text else 0


def check_runtime_budget(text: str) -> BudgetDecision:
    estimated_tokens = estimate_tokens(text)
    estimated_cost = round(
        (estimated_tokens / 1_000) * COST_PER_1000_TOKENS,
        6,
    )

    if estimated_tokens > MAX_INPUT_TOKENS:
        return BudgetDecision(
            allowed=False,
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
            reason=(
                f"Request exceeds the {MAX_INPUT_TOKENS}-token "
                "runtime budget."
            ),
        )

    return BudgetDecision(
        allowed=True,
        estimated_tokens=estimated_tokens,
        estimated_cost=estimated_cost,
        reason="Request is within the runtime budget.",
    )


def classify_risk(query: str) -> RiskLevel:
    query_lower = query.lower()

    if any(
        re.search(pattern, query_lower, flags=re.IGNORECASE)
        for pattern in HIGH_RISK_PATTERNS
    ):
        return RiskLevel.HIGH

    if any(
        re.search(pattern, query_lower, flags=re.IGNORECASE)
        for pattern in MEDIUM_RISK_PATTERNS
    ):
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


def governance_check(query: str) -> GovernanceDecision:
    if not query or not query.strip():
        return GovernanceDecision(
            allowed=False,
            risk_level=RiskLevel.LOW,
            reason="Empty requests are not allowed.",
        )

    budget_decision = check_runtime_budget(query)

    if not budget_decision.allowed:
        return GovernanceDecision(
            allowed=False,
            risk_level=RiskLevel.LOW,
            reason=budget_decision.reason,
        )

    risk_level = classify_risk(query)

    if risk_level == RiskLevel.HIGH:
        return GovernanceDecision(
            allowed=False,
            risk_level=risk_level,
            reason=(
                "This sensitive account or payment action requires "
                "authenticated human support and cannot be handled here."
            ),
        )

    return GovernanceDecision(
        allowed=True,
        risk_level=risk_level,
        reason="Request passed governance and runtime-budget checks.",
    )