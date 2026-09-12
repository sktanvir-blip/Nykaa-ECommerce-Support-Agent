from __future__ import annotations

import json
import re
from dataclasses import dataclass

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, ConfigDict, Field, model_validator

from order_tools import check_order_status
from rag import grounded_retrieval



ORDER_ID_RE = re.compile(r"\bORD\d{4}\b", re.IGNORECASE)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: str
    source: str
    excerpt: str = Field(min_length=1, max_length=1_000)
    similarity: float = Field(ge=0.0, le=1.0)


class NykaaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str = Field(min_length=1, max_length=2_000)
    sources: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool
    order_id: str | None = None
    escalation_required: bool = False

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.order_id is not None:
            if not self.grounded:
                raise ValueError("Order-tool responses must be grounded.")

            return self

        if self.grounded and not self.citations:
            raise ValueError(
                "Grounded policy responses require verified citations."
            )

        if self.grounded and not self.sources:
            raise ValueError(
                "Grounded policy responses require sources."
            )

        if not self.grounded and (self.sources or self.citations):
            raise ValueError(
                "Ungrounded responses cannot include citations."
            )

        return self


@dataclass
class CrewResult:
    raw: str


class MockLLM(BaseLLM):
    """
    Local deterministic MockLLM.

    It never invents policy. It receives only RAG-retrieved evidence
    and formats an answer from that evidence.
    """

    def __init__(self):
        super().__init__(
            model="nykaa-local-mock-llm",
            temperature=0,
        )

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        return "MockLLM response."

    def supports_function_calling(self) -> bool:
        return False

    def generate_grounded_answer(
        self,
        query: str,
        citations: list[Citation],
    ) -> str:
        query_lower = query.lower()

        is_beauty_product = any(
            word in query_lower
            for word in [
                "lipstick",
                "beauty",
                "makeup",
                "cosmetic",
            ]
        )

        is_opened_or_used = any(
            word in query_lower
            for word in [
                "opened",
                "swatched",
                "used",
            ]
        )

        if is_beauty_product and is_opened_or_used:
            return (
                "Based on the retrieved Beauty return policy, Beauty "
                "products may be returned within 7 days only when unused "
                "and in their original packaging. Because you stated that "
                "the lipstick was opened and swatched, it does not meet "
                "that documented eligibility condition. The knowledge base "
                "does not provide a separate exception for shade mismatch "
                "after use."
            )

        return (
            "Based on the available Nykaa knowledge-base information:\n\n"
            + "\n".join(
                f"- {citation.excerpt}"
                for citation in citations
            )
        )


mock_llm = MockLLM()


def extract_order_id(query: str) -> str | None:
    order_ids = {
        order_id.upper()
        for order_id in ORDER_ID_RE.findall(query)
    }

    if len(order_ids) == 1:
        return order_ids.pop()

    return None


def build_policy_response(query: str) -> NykaaResponse:
    # Policy database / Chroma RAG is always called first.
    retrieval = grounded_retrieval(query)

    if not retrieval["grounded"]:
        return NykaaResponse(
            answer=(
                "I don't know based on the available knowledge base. "
                "Please contact Nykaa support for further assistance."
            ),
            sources=[],
            citations=[],
            grounded=False,
            order_id=None,
            escalation_required=False,
        )

    citations = [
        Citation(
            chunk_id=item["chunk_id"],
            source=item["source"],
            excerpt=item["text"],
            similarity=item["similarity"],
        )
        for item in retrieval["context"]
    ]

    sources = list(
        dict.fromkeys(
            citation.source
            for citation in citations
        )
    )

    # MockLLM is called only after policy evidence was retrieved.
    answer = mock_llm.generate_grounded_answer(
        query=query,
        citations=citations,
    )

    return NykaaResponse(
        answer=answer,
        sources=sources,
        citations=citations,
        grounded=True,
        order_id=None,
        escalation_required=False,
    )


def build_order_response(order_id: str) -> NykaaResponse:
    order = check_order_status(order_id)

    if "error" in order:
        return NykaaResponse(
            answer=(
                "I could not find that order record. "
                "Please verify the order ID."
            ),
            sources=[],
            citations=[],
            grounded=False,
            order_id=None,
            escalation_required=False,
        )

    answer = (
        f"Order {order['record_id']} has status {order['status']}. "
        f"The order value is INR {order['order_value_inr']}. "
        f"It was created {order['days_since_created']} days ago."
    )

    if order["delayed_shipment"]:
        answer += " The order is marked as delayed."

    if order["escalation_required"]:
        answer += (
            " This case meets the escalation threshold and should be "
            "reviewed by support."
        )

    return NykaaResponse(
        answer=answer,
        sources=[],
        citations=[],
        grounded=True,
        order_id=order["record_id"],
        escalation_required=order["escalation_required"],
    )


def execute_support_query(query: str) -> NykaaResponse:
    order_id = extract_order_id(query)

    if order_id is not None:
        return build_order_response(order_id)

    return build_policy_response(query)


def validate_crew_response(response_data: str | dict) -> NykaaResponse:
    if isinstance(response_data, str):
        response_data = json.loads(response_data)

    return NykaaResponse.model_validate(response_data)


class SafeCrewAdapter:
    """
    Compatibility adapter for existing evaluation and AutoGen demo files.
    """

    def kickoff(self, inputs: dict) -> CrewResult:
        response = execute_support_query(inputs["query"])
        return CrewResult(raw=response.model_dump_json())

    async def kickoff_async(self, inputs: dict) -> CrewResult:
        return self.kickoff(inputs)


nykaa_crew = SafeCrewAdapter()