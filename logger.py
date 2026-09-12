import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


LOG_FILE = Path(__file__).resolve().parent / "logs" / "nykaa_agent.jsonl"
LOG_LOCK = threading.Lock()

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"
)
LONG_NUMBER_RE = re.compile(
    r"(?<!\d)(?:\d[ -]?){8,18}\d(?!\d)"
)


def create_trace_id() -> str:
    return str(uuid.uuid4())


def start_timer() -> float:
    return time.perf_counter()


def redact_for_logging(value: object) -> str:
    text = str(value)
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    text = LONG_NUMBER_RE.sub("[NUMBER_REDACTED]", text)
    return text[:2_000]


def write_log(
    *,
    trace_id: str,
    endpoint: str,
    event: str,
    status: str,
    duration_ms: float | None = None,
    error: object | None = None,
) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "event": event,
        "status": status,
    }

    if duration_ms is not None:
        record["duration_ms"] = round(duration_ms, 2)

    if error is not None:
        record["error"] = redact_for_logging(error)

    with LOG_LOCK:
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")