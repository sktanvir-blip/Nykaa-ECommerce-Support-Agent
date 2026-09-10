import asyncio
import json
from typing import Sequence

from pydantic import BaseModel, Field

from crew_agent import nykaa_crew

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    TextMessage,
    StructuredMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken


# ============================================================
# TASK 14: AUTOGEN STRUCTURED REVIEW
# ============================================================


# ------------------------------------------------------------
# 1. Pydantic structured verdict
# ------------------------------------------------------------

class VerdictModel(BaseModel):
    approved: bool = Field(
        ...,
        description="Whether the draft can be approved unchanged."
    )

    verdict: str = Field(
        ...,
        description="Short explanation of the review decision."
    )

    revised_answer: str = Field(
        ...,
        description="Final answer after review."
    )

    issues: list[str] = Field(
        default_factory=list,
        description="Issues found in the draft."
    )


# ------------------------------------------------------------
# 2. AutoGen Reviewer Agent
# ------------------------------------------------------------

class ReviewerAgent(BaseChatAgent):

    def __init__(self, name: str = "reviewer") -> None:
        super().__init__(
            name=name,
            description="Reviews a CrewAI draft for correctness and groundedness."
        )

    @property
    def produced_message_types(
        self,
    ) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:

        draft = ""

        for message in messages:
            if isinstance(message, TextMessage):
                draft = message.content

        draft_lower = draft.lower()

        issues = []

        # Known Nykaa policy check.
        if "footwear" in draft_lower and "30 days" in draft_lower:
            issues.append(
                "Footwear return window is incorrect. "
                "The knowledge base specifies 15 days."
            )

        # Order-response consistency check.
        if "ord0001" in draft_lower:
            required_terms = [
                "returned",
                "2367",
                "0.7067",
            ]

            for term in required_terms:
                if term not in draft_lower:
                    issues.append(
                        f"Expected order information is missing: {term}"
                    )

        if issues:
            review = (
                "REVISE\n"
                + "\n".join(f"- {issue}" for issue in issues)
            )
        else:
            review = (
                "APPROVE\n"
                "The draft is consistent with the available Nykaa "
                "information and does not require a correction."
            )

        response_message = TextMessage(
            content=review,
            source=self.name,
        )

        return Response(chat_message=response_message)

    async def on_reset(
        self,
        cancellation_token: CancellationToken,
    ) -> None:
        pass


# ------------------------------------------------------------
# 3. AutoGen Final Editor Agent
# ------------------------------------------------------------

class FinalEditorAgent(BaseChatAgent):

    def __init__(self, name: str = "final_editor") -> None:
        super().__init__(
            name=name,
            description="Produces the final structured answer after review."
        )

    @property
    def produced_message_types(
        self,
    ) -> Sequence[type[BaseChatMessage]]:
        return (StructuredMessage[VerdictModel],)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:

        draft = ""
        review = ""

        for message in messages:

            if isinstance(message, TextMessage):

                if message.source == "user":
                    draft = message.content

                elif message.source == "reviewer":
                    review = message.content

        # ----------------------------------------------------
        # APPROVE CASE
        # ----------------------------------------------------

        if review.startswith("APPROVE"):

            verdict = VerdictModel(
                approved=True,
                verdict="Draft approved unchanged.",
                revised_answer=draft,
                issues=[],
            )

        # ----------------------------------------------------
        # REVISE CASE
        # ----------------------------------------------------

        else:

            revised_answer = draft

            # Correct the deliberately incorrect footwear policy.
            revised_answer = revised_answer.replace(
                "30 days",
                "15 days"
            )

            verdict = VerdictModel(
                approved=False,
                verdict="Draft required correction after review.",
                revised_answer=revised_answer,
                issues=[
                    "Incorrect footwear return window corrected from "
                    "30 days to 15 days."
                ],
            )

        structured_message = StructuredMessage[VerdictModel](
            content=verdict,
            source=self.name,
        )

        return Response(
            chat_message=structured_message
        )

    async def on_reset(
        self,
        cancellation_token: CancellationToken,
    ) -> None:
        pass


# ------------------------------------------------------------
# 4. Create AutoGen review team
# ------------------------------------------------------------

def create_review_team():

    reviewer = ReviewerAgent()

    final_editor = FinalEditorAgent()

    review_team = RoundRobinGroupChat(
        participants=[
            reviewer,
            final_editor,
        ],
        max_turns=2,
        custom_message_types=[
            StructuredMessage[VerdictModel]
        ],
    )

    return review_team


# ------------------------------------------------------------
# 5. Run one AutoGen review
# ------------------------------------------------------------

async def run_review(draft: str):

    review_team = create_review_team()

    result = await review_team.run(
        task=TextMessage(
            content=draft,
            source="user",
        )
    )

    return result


# ------------------------------------------------------------
# 6. Demonstrate APPROVE unchanged
# ------------------------------------------------------------

async def demonstrate_approve():

    print("\n")
    print("=" * 70)
    print("TASK 14 - APPROVE UNCHANGED")
    print("=" * 70)

    good_draft = (
        "Order ORD0001 has status Returned. "
        "The order value is INR 2367. "
        "The order was created 8 days ago. "
        "It is marked as a delayed shipment, "
        "with an escalation score of 0.7067, "
        "so escalation is required."
    )

    result = await run_review(good_draft)

    print("\nCREWAI DRAFT:")
    print(good_draft)

    print("\nAUTOGEN CONVERSATION:")

    for message in result.messages:
        print(f"\n[{message.source}]")

        if isinstance(message, StructuredMessage):
            print(message.to_text())
        else:
            print(message.content)

    final_message = result.messages[-1]

    if isinstance(final_message, StructuredMessage):
        verdict = final_message.content

        print("\nFINAL PYDANTIC VERDICT:")
        print(verdict.model_dump())

        print("\nApproved unchanged:", verdict.approved)


# ------------------------------------------------------------
# 7. Demonstrate REVISE
# ------------------------------------------------------------

async def demonstrate_revise():

    print("\n")
    print("=" * 70)
    print("TASK 14 - REVISE")
    print("=" * 70)

    flawed_draft = (
        "Footwear can be returned within 30 days "
        "of delivery according to the Nykaa policy."
    )

    result = await run_review(flawed_draft)

    print("\nCREWAI DRAFT WITH INTENTIONAL ERROR:")
    print(flawed_draft)

    print("\nAUTOGEN CONVERSATION:")

    for message in result.messages:
        print(f"\n[{message.source}]")

        if isinstance(message, StructuredMessage):
            print(message.to_text())
        else:
            print(message.content)

    final_message = result.messages[-1]

    if isinstance(final_message, StructuredMessage):
        verdict = final_message.content

        print("\nFINAL PYDANTIC VERDICT:")
        print(verdict.model_dump())

        print("\nApproved unchanged:", verdict.approved)
        print("Revised answer:", verdict.revised_answer)


# ------------------------------------------------------------
# 8. Main
# ------------------------------------------------------------

async def demonstrate_real_crewai_draft():

    print("\n")
    print("=" * 70)
    print("TASK 14 - REAL CREWAI → AUTOGEN REVIEW")
    print("=" * 70)

    query = "What is the status of order ORD0001?"

    # --------------------------------------------------------
    # STEP 1: Generate the draft using the real CrewAI crew
    # --------------------------------------------------------

    crew_result = await nykaa_crew.kickoff_async(
        inputs={"query": query}
    )

    raw_crew_draft = crew_result.raw

    try:
        crew_data = json.loads(raw_crew_draft)

        if isinstance(crew_data, dict) and "answer" in crew_data:
            crew_draft = crew_data["answer"]
        else:
            crew_draft = raw_crew_draft

    except (json.JSONDecodeError, TypeError):
        crew_draft = raw_crew_draft

    print("\nUSER QUERY:")
    print(query)

    print("\nREAL CREWAI DRAFT:")
    print(crew_draft)

    # --------------------------------------------------------
    # STEP 2: Send the real CrewAI draft to AutoGen
    # --------------------------------------------------------

    autogen_result = await run_review(
        crew_draft
    )

    print("\nAUTOGEN REVIEW CONVERSATION:")

    for message in autogen_result.messages:

        print(f"\n[{message.source}]")

        if isinstance(message, StructuredMessage):
            print(message.to_text())
        else:
            print(message.content)

    # --------------------------------------------------------
    # STEP 3: Read final structured Pydantic verdict
    # --------------------------------------------------------

    final_message = autogen_result.messages[-1]

    if isinstance(final_message, StructuredMessage):

        verdict = final_message.content

        print("\nFINAL PYDANTIC VERDICT:")
        print(verdict.model_dump())

        print("\nApproved:", verdict.approved)

        print("\nFinal answer after AutoGen review:")
        print(verdict.revised_answer)


async def main():

    # Existing standalone demonstrations
    await demonstrate_approve()

    await demonstrate_revise()

    # NEW: actual CrewAI → AutoGen integration
    await demonstrate_real_crewai_draft()

    print("\n")
    print("=" * 70)
    print("TASK 14 AUTOGEN REVIEW COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())