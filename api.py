from __future__ import annotations

import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from crew_agent import execute_support_query
from governance import governance_check
from guardrails import MAX_QUERY_CHARS, apply_input_guardrails
from logger import create_trace_id, start_timer, write_log
from rag import RagNotReadyError, get_index_status


app = FastAPI(
    title="Nykaa Domain Support Agent",
    description="Grounded customer-support API for the Nykaa domain.",
    version="2.0.0",
)


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,
    )

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
    )


def blocked_response(
    *,
    answer: str,
    reason: str,
    trace_id: str,
) -> dict:
    return {
        "answer": answer,
        "reason": reason,
        "sources": [],
        "citations": [],
        "grounded": False,
        "order_id": None,
        "escalation_required": False,
        "trace_id": trace_id,
    }


def process_query(
    *,
    raw_query: str,
    endpoint: str,
) -> tuple[dict, int]:
    trace_id = create_trace_id()
    start_time = start_timer()

    write_log(
        trace_id=trace_id,
        endpoint=endpoint,
        event="request_started",
        status="started",
    )

    guardrail_result = apply_input_guardrails(raw_query)

    if not guardrail_result["allowed"]:
        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint=endpoint,
            event="request_blocked",
            status="blocked",
            duration_ms=duration_ms,
        )

        return (
            blocked_response(
                answer="Request blocked by input security controls.",
                reason=guardrail_result["reason"],
                trace_id=trace_id,
            ),
            422,
        )

    sanitized_query = guardrail_result["text"]
    governance_decision = governance_check(sanitized_query)

    if not governance_decision.allowed:
        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint=endpoint,
            event="governance_blocked",
            status="blocked",
            duration_ms=duration_ms,
        )

        return (
            blocked_response(
                answer="Request requires secure human support.",
                reason=governance_decision.reason,
                trace_id=trace_id,
            ),
            403,
        )

    try:
        response = execute_support_query(sanitized_query)

        response_data = response.model_dump()
        response_data["trace_id"] = trace_id
        response_data["risk_level"] = governance_decision.risk_level.value

        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint=endpoint,
            event="request_completed",
            status="success",
            duration_ms=duration_ms,
        )

        return response_data, 200

    except RagNotReadyError as error:
        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint=endpoint,
            event="request_failed",
            status="service_unavailable",
            duration_ms=duration_ms,
            error=error,
        )

        return (
            blocked_response(
                answer="The knowledge-base service is not ready.",
                reason="RAG index unavailable. Build the local index first.",
                trace_id=trace_id,
            ),
            503,
        )

    except Exception as error:
        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint=endpoint,
            event="request_failed",
            status="error",
            duration_ms=duration_ms,
            error=error,
        )

        return (
            blocked_response(
                answer="An internal error occurred while processing the request.",
                reason="Internal processing error.",
                trace_id=trace_id,
            ),
            500,
        )


@app.get("/health")
def health_check():
    index_status = get_index_status()

    return {
        "status": "ok" if index_status["ready"] else "degraded",
        "service": "Nykaa Domain Support Agent",
        "rag": index_status,
    }


@app.post("/query")
def query_agent(request: QueryRequest):
    response_data, status_code = process_query(
        raw_query=request.query,
        endpoint="/query",
    )

    return JSONResponse(
        content=response_data,
        status_code=status_code,
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            query = await websocket.receive_text()

            if len(query) > MAX_QUERY_CHARS:
                trace_id = create_trace_id()

                await websocket.send_json(
                    blocked_response(
                        answer="Request blocked by input security controls.",
                        reason=(
                            f"Query exceeds the {MAX_QUERY_CHARS}-character limit."
                        ),
                        trace_id=trace_id,
                    )
                )

                continue

            response_data, _ = await run_in_threadpool(
                process_query,
                raw_query=query,
                endpoint="/ws",
            )

            await websocket.send_json(response_data)

    except WebSocketDisconnect:
        return