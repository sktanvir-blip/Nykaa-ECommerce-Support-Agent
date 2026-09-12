# Nykaa Domain Support Agent

Agentic AI customer-support system for the **E-commerce & Retail (Nykaa)** domain.

The project demonstrates RAG, multi-agent orchestration, tool calling, conversational memory, structured outputs, guardrails, evaluation, governance, caching, logging, and FastAPI deployment using deterministic local/mock components.

## Architecture

```text
User
  |
  v
FastAPI REST / WebSocket
  |
  +--> Input Guardrails
  |      - PII masking
  |      - Prompt-injection detection
  |
  +--> Governance
  |      - Policy checks
  |      - Least-autonomy / permission control
  |      - Risk classification
  |      - Runtime token/cost budget
  |
  v
CrewAI Sequential Crew
  |
  +--> Retrieval Specialist
  |      |
  |      +--> RAG Tool
  |             SentenceTransformers
  |             + ChromaDB
  |
  +--> Order Lookup Specialist
  |      |
  |      +--> Order Lookup Tool
  |
  +--> Customer Support Composer
  |      |
  |      +--> Pydantic structured response
  |
  v
AutoGen Review
  |
  +--> Reviewer Agent
  +--> Final Editor
  |
  v
Validated customer response
```

### Framework responsibilities

- **CrewAI** — multi-agent orchestration and tool calling.
- **LangChain** — session-based conversational memory.
- **AutoGen** — independent post-generation review and correction.
- **FastAPI** — REST and WebSocket deployment.
- **ChromaDB** — persistent vector retrieval.
- **SentenceTransformers** — local text embeddings.
- **Pydantic** — structured response validation.

## Dataset

The synthetic order dataset is generated deterministically with a fixed random seed.

- 60 order records.
- Minimum required: 40.
- Categories include: Apparel, Electronics, Home, Footwear, Beauty.
- Statuses include: Placed, Shipped, Delivered, Returned, Refunded.
- Order values: INR 300–25,000.
- Delayed shipment rate: 20%.
- Every required category and status is represented.
- Record IDs use the format `ORD0001`, `ORD0002`, etc.

The deterministic seed makes demonstrations and evaluation reproducible.

## Knowledge Base and RAG

The knowledge base contains 12 Markdown documents covering:

1. Return window by product category
2. COD refund timelines
3. Delivery SLAs
4. Reverse-pickup eligibility
5. Warranty terms by category
6. Order-cancellation policy
7. Loyalty-points redemption
8. Payment-failure/retry
9. Size exchange
10. Damaged-item claims
11. International shipping restrictions
12. Support escalation matrix

Two chunking strategies are implemented and evaluated:

- Fixed-size chunks with overlap.
- Sentence-based chunks.

Embeddings use `all-MiniLM-L6-v2`.

Two separate ChromaDB collections are maintained:

- `nykaa_fixed_chunks`
- `nykaa_sentence_chunks`

Chunks are indexed with `collection.upsert()`.

Grounded generation uses retrieved knowledge-base context only. When the calibrated similarity threshold is not met, the system returns an explicit knowledge-base fallback instead of fabricating an answer.

The evaluation compared both chunking strategies on the same query set. On the included evaluation set, sentence-based chunking achieved higher document-level precision while both strategies achieved full recall.

## Multi-Agent System

The CrewAI crew contains three agents:

### Retrieval Specialist
Uses the RAG knowledge-base tool to retrieve policy information.

### Order Lookup Specialist
Uses the order lookup tool to retrieve order status, order value, and escalation information.

### Customer Support Composer
Combines the available evidence into a structured customer-facing response.

Only the Lookup Agent is granted the order lookup capability, following least-autonomy principles.

## Order Escalation

For an order record, the escalation score is:

```text
escalation_score =
    0.6 * delayed_shipment_score
    + 0.4 * normalized_recency_score
```

The score is capped at 1.0.

The current escalation threshold is `0.60`.

This makes delayed shipments the stronger escalation signal while still accounting for order recency.

## Memory

LangChain session memory is demonstrated using:

- `InMemoryChatMessageHistory`
- `RunnableWithMessageHistory`

Different session IDs maintain separate conversation histories.

## Structured Output

CrewAI responses are validated using the Pydantic `NykaaResponse` model with fields for:

- answer
- sources
- grounded
- order_id
- escalation_required

Invalid structured output is rejected by validation.

## Guardrails

### Input guardrails

- Masks email addresses.
- Masks 10-digit Indian mobile numbers.
- Detects common prompt-injection attempts.

### Output guardrail

If a generated response is not grounded, the system refuses to present it as a grounded answer and returns the configured fallback.

Both input and output guardrail firing cases are demonstrated in the project.

## FastAPI

Endpoints:

- `GET /health`
- `POST /query`
- `WebSocket /ws`

The WebSocket implementation catches `WebSocketDisconnect`.

Requests receive a trace ID and structured JSONL telemetry.

## Logging

Logs are written as JSON Lines to:

```text
logs/nykaa_agent.jsonl
```

Each event contains structured fields such as timestamp, trace ID, endpoint, event, status, and duration.

Raw fixed-format personal information is not intentionally logged.

The `logs/` directory is excluded from Git.

## Evaluation

A deterministic 15-query evaluation suite is included.

Each query is scored on:

- Accuracy
- Grounding
- Completeness
- Safety

The evaluation runs under `MOCK_LLM`, so it is reproducible without a paid LLM provider.

The current mock-evaluation averages are:

| Metric | Average / 5 |
|---|---:|
| Accuracy | 1.80 |
| Grounding | 4.47 |
| Completeness | 1.80 |
| Safety | 4.73 |

The lower Accuracy and Completeness values are a known limitation of the intentionally deterministic mock responder. The evaluation framework itself is implemented for repeatable testing; production quality would depend on the selected real LLM.

## AutoGen Review

After a CrewAI draft, AutoGen performs an independent two-agent review:

- Reviewer Agent
- Final Editor Agent

A `RoundRobinGroupChat` is used with a two-turn limit.

The final editor returns a Pydantic `VerdictModel` through a structured message.

The demo covers:

- Approve unchanged
- Detect an incorrect answer
- Revise the answer
- CrewAI → AutoGen review integration

## Governance

The governance layer implements four controls:

1. **Policy** — identifies disallowed high-risk requests.
2. **Permission / least autonomy** — limits sensitive tool access.
3. **Risk classification** — LOW, MEDIUM, HIGH.
4. **Runtime enforcement** — rejects oversized requests before agent execution.

The runtime token budget is capped at 1000 estimated tokens per request.

The cost estimate is a mock engineering estimate used for budget-control demonstration, not a provider billing rate.

High-risk operations and oversized inputs are rejected.

## Cache

`cache.py` implements an in-memory cache:

- Normalizes the query.
- Uses the normalized query as the cache key.
- Demonstrates a cache hit.
- Avoids repeating the expensive operation for an equivalent query.

## Mock / Offline Design

The project is designed to run without an LLM provider API key.

- CrewAI uses a deterministic `MockLLM`.
- Embeddings use the local SentenceTransformers model.
- ChromaDB runs locally.
- Order data is synthetic and deterministic.
- Evaluation uses a deterministic mock judge.

No OpenAI, Anthropic, Gemini, or other paid LLM API key is required.

## Telemetry

CrewAI telemetry is disabled for this project using:

```text
CREWAI_DISABLE_TELEMETRY=true
```

If preferred, the OpenTelemetry SDK can also be disabled with:

```text
OTEL_SDK_DISABLED=true
```

## Installation

Create and activate a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify dependencies:

```powershell
python -m pip check
```

## Run the components

Dataset:

```powershell
python dataset.py
```

Order tools:

```powershell
python order_tools.py
```

RAG:

```powershell
python rag.py
```

Memory:

```powershell
python memory.py
```

Guardrails:

```powershell
python guardrails.py
```

Governance:

```powershell
python governance.py
```

Cache:

```powershell
python cache.py
```

Evaluation:

```powershell
python evaluation.py
```

AutoGen review:

```powershell
python autogen_review.py
```

Start FastAPI:

```powershell
uvicorn api:app --reload
```

Health check:

```text
GET /health
```

Query endpoint:

```text
POST /query
```

Example request:

```json
{
  "query": "What is the return policy for footwear?"
}
```

WebSocket:

```text
/ws
```

The included `test_websocket.py` demonstrates WebSocket communication.

## Verification

A project-only syntax check can be run with:

```powershell
python -m compileall -q -x "\\.venv\\" .
```

A clean run produces no output.

## Project Structure

```text
Nykaa-ECommerce-Support-Agent/
├── Knowledge_base/
├── api.py
├── autogen_review.py
├── cache.py
├── crew_agent.py
├── dataset.py
├── evaluation.py
├── evaluation_results.json
├── governance.py
├── guardrails.py
├── logger.py
├── memory.py
├── order_tools.py
├── rag.py
├── requirements.txt
└── test_websocket.py
```

## Limitations

This is an educational capstone implementation.

- Order records are synthetic.
- Policy documents are project-created demonstration documents.
- `MOCK_LLM` is deterministic and intentionally limited compared with a production LLM.
- ChromaDB data is generated locally and is not committed to Git.
- The cache is in-memory and therefore resets when the process restarts.
- Real production deployment would require authenticated user identity, persistent order storage, production observability, stronger policy enforcement, and a real LLM/provider configuration.

## Track

**Capstone Track: E-commerce & Retail — Nykaa**

The implementation focuses on customer-support workflows such as policy questions, order status, refunds, returns, exchanges, delivery issues, escalation, and safe handling of unsupported requests.
