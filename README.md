# GridWise LLM API — Smart Campus Energy Optimization

BUP CSE Fest 2026 Hackathon submission.

Converts free-text campus operator notes into structured schedule directives with
Gemini, validates them deterministically, and solves a 24-hour MILP that minimizes
total grid electricity cost.

```
request → Pydantic contract validation → Gemini (untrusted) → guardrails → MILP → replay-verified response
```

## Endpoints

Both are public, unauthenticated, and are the only HTTP routes the service exposes
(FastAPI's `/docs`, `/redoc` and `/openapi.json` are disabled).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | readiness probe, returns `{"status": "ok"}` |
| POST | `/optimize-energy` | returns the optimized 24-hour schedule |

Malformed requests return a controlled `400` (set `VALIDATION_ERROR_STATUS=422`
for FastAPI's default). Internal failures return a generic `500` with no
traceback and no secret.

## Quickstart

```bash
git clone <repo-url>
cd smart_campus_energy_optimization

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then paste your key into .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

Tested on Python 3.11. The CBC solver ships inside the `pulp` wheel, so no
separate solver install is required.

## Docker

```bash
docker build -t gridwise .
docker run --rm -p 8000:8000 -e GEMINI_API_KEY="$GEMINI_API_KEY" gridwise
```

The container binds `0.0.0.0:8000`. The key is injected at runtime and is never
baked into the image (`.env` is excluded via `.dockerignore`).

## Sample request

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/optimize-energy \
     -H "Content-Type: application/json" \
     -d @sample_request.json | python -m json.tool
```

`sample_request.json` carries three notes — a solar reduction, a no-charge
window, and one irrelevant note. Expected behaviour: the first two become active
directives, the third becomes `no_op` with `applies: false`, and `hourly_plan`
has 24 entries whose battery state returns to `initial_energy_kwh` at hour 23.

## Configuration

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | yes | — | Google AI Studio key |
| `GEMINI_TIMEOUT_MS` | no | `8000` | provider call timeout |
| `VALIDATION_ERROR_STATUS` | no | `400` | status for schema violations |

Model: `gemini-2.5-flash`, temperature 0, thinking disabled, JSON response mode.
The client is built lazily, so a missing key degrades every note to `no_op`
rather than taking the process — and `/health` — down at import time.

## Tests

```bash
pip install pytest httpx
pytest -q
```

52 tests. The Gemini call is stubbed, so no key and no network are needed.

## Design notes

**The LLM never touches the optimizer directly.** `guardrails.py` sits between
them and treats model output as hostile input. Every malformed shape — wrong
type, NaN, `Infinity`, boolean-as-number, non-string explanation, duplicate or
out-of-range `note_index`, missing `structured_adjustment` — collapses to a safe
`no_op`. The response always carries exactly one interpretation per note, in
ascending index order.

**Charge and discharge are mutually exclusive.** A binary mode variable per hour
makes it impossible for the solver to do both, since the reported plan has a
single `battery_action` field and a judge replaying that action must reproduce
the same state transition.

**The reported numbers replay exactly.** The serialized plan is built from the
rounded battery state chain rather than from the solver's full-precision
internals, so the per-hour energy balance, the state transition and end-of-day
neutrality all hold at the precision that actually leaves the API.
`optimizer.verify_plan()` re-checks all of those invariants — plus rate limits
and capacity bounds — before the response is returned, and fails closed.

## Files

| File | Role |
| --- | --- |
| `main.py` | FastAPI app, both endpoints, error handlers |
| `models.py` | strict request/response contract |
| `llm_interpreter.py` | Gemini prompt and call |
| `guardrails.py` | deterministic validation of model output |
| `optimizer.py` | MILP model, serialization, replay verification |
| `tests/test_compliance.py` | regression tests |

## Security

No credential appears anywhere in the source. `.env` is git-ignored and
docker-ignored, the key is read only from the environment, and no error path
logs or returns it.
