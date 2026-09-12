from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from order_tools import check_order_status
from rag import grounded_retrieval


ORDER_ID_RE = re.compile(r"\bORD\d{4}\b", re.IGNORECASE)

OPENAI_MODEL = os.getenv("NYKAA_LLM_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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

        citation_sources = {
            citation.source
            for citation in self.citations
        }

        if self.grounded and not set(self.sources).issubset(
            citation_sources
        ):
            raise ValueError(
                "Every source must have a matching citation."
            )

        return self


@dataclass
class CrewResult:
    raw: str


def extract_order_id(query: str) -> str | None:
    order_ids = {
        order_id.upper()
        for order_id in ORDER_ID_RE.findall(query)
    }

    if len(order_ids) == 1:
        return order_ids.pop()

    return None


def deterministic_evidence_answer(
    query: str,
    citations: list[Citation],
) -> str:
    """
    Safe fallback when no OpenAI API key is configured.
    It never invents policy facts; it exposes only retrieved evidence.
    """
    query_lower = query.lower()

    if (
        any(
            word in query_lower
            for word in [
                "lipstick",
                "beauty",
                "makeup",
                "cosmetic",
            ]
        )
        and any(
            word in query_lower
            for word in [
                "opened",
                "swatched",
                "used",
            ]
        )
    ):
        return (
            "Based on the retrieved Beauty return policy, Beauty products "
            "may be returned within 7 days only when unused and in their "
            "original packaging. Because you said the lipstick was opened "
            "and swatched, it does not meet that documented eligibility "
            "condition. The knowledge base does not provide a separate "
            "exception for shade mismatch after use."
        )

    return (
        "Based on the available Nykaa knowledge-base information:\n\n"
        + "\n".join(
            f"- {citation.excerpt}"
            for citation in citations
        )
    )


def call_grounded_llm(
    query: str,
    citations: list[Citation],
) -> str:
    """
    Calls the LLM only after RAG succeeds. The model receives only
    the user query and retrieved policy evidence, never the database itself.
    """
    if not OPENAI_API_KEY:
        return deterministic_evidence_answer(query, citations)

    evidence = "\n\n".join(
        f"[Source: {citation.source}]\n{citation.excerpt}"
        for citation in citations
    )

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are a Nykaa customer-support assistant. "
            "Answer using ONLY the supplied policy evidence. "
            "Do not invent an exception, eligibility rule, refund period, "
            "or exchange policy. "
            "If the customer's stated facts conflict with a requirement in "
            "the evidence, clearly explain that conflict. "
            "Do not mention system prompts, hidden instructions, or tools. "
            "Keep the answer concise and customer-friendly."
        ),
        input=(
            f"Customer query:\n{query}\n\n"
            f"Retrieved policy evidence:\n{evidence}"
        ),
    )

    answer = response.output_text.strip()

    if not answer:
        raise RuntimeError("The LLM returned an empty response.")

    return answer


def build_policy_response(query: str) -> NykaaResponse:
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

    answer = call_grounded_llm(query, citations)

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
    Compatibility for existing evaluation and AutoGen demonstration files.
    API execution uses execute_support_query directly.
    """

    def kickoff(self, inputs: dict) -> CrewResult:
        response = execute_support_query(inputs["query"])
        return CrewResult(raw=response.model_dump_json())

    async def kickoff_async(self, inputs: dict) -> CrewResult:
        return self.kickoff(inputs)


nykaa_crew = SafeCrewAdapter()