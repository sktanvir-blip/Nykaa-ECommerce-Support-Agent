from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from crew_agent import nykaa_crew
from guardrails import apply_input_guardrails
from governance import governance_check
import time
from logger import create_trace_id, start_timer, write_log
from crew_agent import validate_crew_response


app = FastAPI(
    title="Nykaa Domain Support Agent",
    description="Agentic AI support API for the Nykaa e-commerce domain.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Nykaa Domain Support Agent",
    }


@app.post("/query")
def query_agent(request: QueryRequest):
    trace_id = create_trace_id()
    start_time = start_timer()

    write_log(
        trace_id=trace_id,
        endpoint="/query",
        event="request_started",
        status="started",
    )

    input_guardrail_result = apply_input_guardrails(request.query)

    if not input_guardrail_result["allowed"]:
        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint="/query",
            event="request_blocked",
            status="blocked",
            duration_ms=duration_ms,
        )

        return {
            "answer": "Request blocked by input guardrail.",
            "reason": input_guardrail_result["reason"],
            "grounded": False,
            "trace_id": trace_id,
        }

    sanitized_query = input_guardrail_result["text"]
    governance_decision = governance_check(sanitized_query)

    if not governance_decision.allowed:
        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint="/query",
            event="governance_blocked",
            status="blocked",
            duration_ms=duration_ms,
        )

        return {
            "answer": "Request blocked by governance policy.",
            "reason": governance_decision.reason,
            "grounded": False,
            "trace_id": trace_id,
        }

    try:
        result = nykaa_crew.kickoff(
            inputs={"query": sanitized_query}
        )

        validated_response = validate_crew_response(result.raw)

        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint="/query",
            event="request_completed",
            status="success",
            duration_ms=duration_ms,
        )

        response_data = validated_response.model_dump()
        response_data["trace_id"] = trace_id

        return response_data

    except Exception as error:
        duration_ms = (time.perf_counter() - start_time) * 1000

        write_log(
            trace_id=trace_id,
            endpoint="/query",
            event="request_failed",
            status="error",
            duration_ms=duration_ms,
            error=error,
        )

        return {
            "answer": "An internal error occurred while processing the request.",
            "grounded": False,
            "trace_id": trace_id,
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    trace_id = create_trace_id()

    write_log(
        trace_id=trace_id,
        endpoint="/ws",
        event="connection_started",
        status="started",
    )

    try:
        while True:
            query = await websocket.receive_text()

            start_time = time.perf_counter()

            input_guardrail_result = apply_input_guardrails(query)

            if not input_guardrail_result["allowed"]:
                duration_ms = (time.perf_counter() - start_time) * 1000

                write_log(
                    trace_id=trace_id,
                    endpoint="/ws",
                    event="request_blocked",
                    status="blocked",
                    duration_ms=duration_ms,
                )

                await websocket.send_json({
                    "answer": "Request blocked by input guardrail.",
                    "reason": input_guardrail_result["reason"],
                    "grounded": False,
                    "trace_id": trace_id,
                })

                continue

            sanitized_query = input_guardrail_result["text"]
            governance_decision = governance_check(sanitized_query)

            if not governance_decision.allowed:
                duration_ms = (time.perf_counter() - start_time) * 1000

                write_log(
                    trace_id=trace_id,
                    endpoint="/ws",
                    event="governance_blocked",
                    status="blocked",
                    duration_ms=duration_ms,
                )

                await websocket.send_json({
                    "answer": "Request blocked by governance policy.",
                    "reason": governance_decision.reason,
                    "grounded": False,
                    "trace_id": trace_id,
                })

                continue

            try:
                result = await nykaa_crew.kickoff_async(
                    inputs={"query": sanitized_query}
                )

                validated_response = validate_crew_response(result.raw)

                duration_ms = (time.perf_counter() - start_time) * 1000

                write_log(
                    trace_id=trace_id,
                    endpoint="/ws",
                    event="request_completed",
                    status="success",
                    duration_ms=duration_ms,
                )

                response_data = validated_response.model_dump()
                response_data["trace_id"] = trace_id

                await websocket.send_json(response_data)

            except Exception as error:
                duration_ms = (time.perf_counter() - start_time) * 1000

                write_log(
                    trace_id=trace_id,
                    endpoint="/ws",
                    event="request_failed",
                    status="error",
                    duration_ms=duration_ms,
                    error=error,
                )

                await websocket.send_json({
                    "answer": "An internal error occurred while processing the request.",
                    "grounded": False,
                    "trace_id": trace_id,
                })

    except WebSocketDisconnect:
        write_log(
            trace_id=trace_id,
            endpoint="/ws",
            event="connection_closed",
            status="disconnected",
        )

        print(f"WebSocket disconnected. Trace ID: {trace_id}")