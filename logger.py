import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


LOG_FILE = Path("logs") / "nykaa_agent.jsonl"


def create_trace_id():
    return str(uuid.uuid4())


def start_timer():
    return time.perf_counter()


def write_log(
    trace_id,
    endpoint,
    event,
    status,
    duration_ms=None,
    query=None,
    error=None,
):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "event": event,
        "status": status,
    }

    if duration_ms is not None:
        log_record["duration_ms"] = round(duration_ms, 2)

    if query is not None:
        log_record["query"] = query

    if error is not None:
        log_record["error"] = str(error)

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_record) + "\n")


if __name__ == "__main__":

    trace_id = create_trace_id()
    start_time = start_timer()

    time.sleep(0.05)

    duration_ms = (time.perf_counter() - start_time) * 1000

    write_log(
        trace_id=trace_id,
        endpoint="/query",
        event="test_request",
        status="success",
        duration_ms=duration_ms,
        query="My email is [EMAIL_REDACTED]",
    )

    print("Trace ID:", trace_id)
    print("Duration:", round(duration_ms, 2), "ms")
    print("Log file:", LOG_FILE)
    print("\nStructured logging test completed.")