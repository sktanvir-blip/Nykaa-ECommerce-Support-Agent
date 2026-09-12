from enum import Enum
from dataclasses import dataclass
import math


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

def classify_risk(query: str) -> RiskLevel:
    """
    Classify a user request into LOW, MEDIUM, or HIGH risk.
    """

    query_lower = query.lower()

    # High-risk requests
    high_risk_keywords = [
        "delete my account",
        "change bank account",
        "change payment details",
        "refund to another account",
        "cancel all my orders",
    ]

    # Medium-risk transactional requests
    medium_risk_keywords = [
        "i want to return",
        "i want an exchange",
        "refund",
        "complaint",
        "escalate",
        "payment failed",
    ]

    # Check HIGH risk first
    for keyword in high_risk_keywords:
        if keyword in query_lower:
            return RiskLevel.HIGH

    # Then check MEDIUM risk
    for keyword in medium_risk_keywords:
        if keyword in query_lower:
            return RiskLevel.MEDIUM

    # General informational questions are LOW risk
    return RiskLevel.LOW

MAX_INPUT_TOKENS = 1000

# Mock cost rate used only for budget estimation.
# This does not represent an actual provider charge.
COST_PER_1000_TOKENS = 0.002


@dataclass
class BudgetDecision:
    allowed: bool
    estimated_tokens: int
    estimated_cost: float
    reason: str


def estimate_tokens(text: str) -> int:
    """
    Estimate input tokens deterministically.

    A simple approximation of 1 token per 4 characters
    is sufficient for runtime budget enforcement in this
    MOCK_LLM project.
    """

    if not text:
        return 0

    return math.ceil(len(text) / 4)


def check_runtime_budget(text: str) -> BudgetDecision:
    """
    Reject requests that exceed the configured runtime budget.
    """

    estimated_tokens = estimate_tokens(text)

    estimated_cost = (
        estimated_tokens / 1000
    ) * COST_PER_1000_TOKENS

    if estimated_tokens > MAX_INPUT_TOKENS:

        return BudgetDecision(
            allowed=False,
            estimated_tokens=estimated_tokens,
            estimated_cost=round(estimated_cost, 6),
            reason=(
                "Request exceeds the maximum runtime token budget "
                f"of {MAX_INPUT_TOKENS} tokens."
            ),
        )

    return BudgetDecision(
        allowed=True,
        estimated_tokens=estimated_tokens,
        estimated_cost=round(estimated_cost, 6),
        reason="Request is within the runtime token/cost budget.",
    )

@dataclass
class GovernanceDecision:
    allowed: bool
    risk_level: RiskLevel
    reason: str


def governance_check(query: str) -> GovernanceDecision:
    """
    Apply four governance layers before agent execution.

    Layer 1: Policy
    Layer 2: Permission / least autonomy
    Layer 3: Risk control
    Layer 4: Runtime enforcement
    """

    if not query or not query.strip():

        return GovernanceDecision(
            allowed=False,
            risk_level=RiskLevel.LOW,
            reason="Empty requests are not allowed.",
        )

    risk_level = classify_risk(query)

    if risk_level == RiskLevel.HIGH:

        return GovernanceDecision(
            allowed=False,
            risk_level=risk_level,
            reason=(
                "High-risk request requires human review "
                "and cannot be autonomously executed."
            ),
        )


    budget_decision = check_runtime_budget(query)

    if not budget_decision.allowed:

        return GovernanceDecision(
            allowed=False,
            risk_level=risk_level,
            reason=(
                f"Runtime budget rejected the request. "
                f"{budget_decision.reason}"
            ),
        )

    return GovernanceDecision(
        allowed=True,
        risk_level=risk_level,
        reason=(
            "Request passed governance checks and "
            "runtime budget validation."
        ),
    )

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("=" * 70)

    test_queries = [
        "What is the return policy for footwear?",
        "I want to return my footwear.",
        "I want a refund for my order.",
        "Please delete my account.",
        "",
    ]

    for query in test_queries:

        decision = governance_check(query)

        print("\nQuery:")
        print(query if query else "[EMPTY QUERY]")

        print("Risk level:", decision.risk_level.value)
        print("Allowed:", decision.allowed)
        print("Reason:", decision.reason)


    print("\n")
    print("=" * 70)
    print("RUNTIME TOKEN/COST BUDGET TEST")
    print("=" * 70)

    normal_query = "What is the return policy for footwear?"

    oversized_query = "return policy " * 500

    for label, query in [
        ("NORMAL REQUEST", normal_query),
        ("OVERSIZED REQUEST", oversized_query),
    ]:

        budget = check_runtime_budget(query)

        print("\n" + label)
        print("Estimated tokens:", budget.estimated_tokens)
        print("Estimated cost:", budget.estimated_cost)
        print("Allowed:", budget.allowed)
        print("Reason:", budget.reason)

    print("\n")
    print("=" * 70)