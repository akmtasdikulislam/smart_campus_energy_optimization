"""
GridWise LLM API.

Two public, unauthenticated endpoints:
    GET  /health           -> {"status": "ok"}
    POST /optimize-energy  -> optimized 24-hour schedule

Request flow: Pydantic contract validation -> Gemini interpretation (untrusted)
-> deterministic guardrails -> MILP optimizer -> replay-verified response.
"""

import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from guardrails import validate_all
from llm_interpreter import interpret_notes
from models import DirectiveInterpretation, OptimizeResponse, ScenarioRequest
from optimizer import compute_cost, compute_totals, solve_schedule

# Set VALIDATION_ERROR_STATUS=422 to keep FastAPI's default instead.
VALIDATION_ERROR_STATUS = int(os.environ.get("VALIDATION_ERROR_STATUS", "400"))

# docs/redoc/openapi disabled so the service exposes exactly the two
# challenge routes and nothing else.
app = FastAPI(
    title="GridWise LLM API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Malformed or structurally invalid requests get a controlled error,
    never a 500 and never a traceback."""
    return JSONResponse(
        status_code=VALIDATION_ERROR_STATUS,
        content={
            "error": "invalid_request",
            "message": "Request does not match the required scenario schema.",
            "details": [
                {"field": ".".join(str(p) for p in err.get("loc", [])),
                 "issue": err.get("msg", "invalid")}
                for err in exc.errors()
            ][:20],
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(req: ScenarioRequest):
    try:
        # 1. LLM interpretation (untrusted)
        raw_directives = interpret_notes(req.operator_notes)

        # 2. Guardrail validation -> guaranteed-safe directives
        directives = validate_all(
            raw_directives,
            num_notes=len(req.operator_notes),
            battery_capacity=req.battery.capacity_kwh,
        )

        # 3. Apply directives + solve optimization
        plan = solve_schedule(req.hours, req.battery, directives)

        # 4. Totals
        total_grid, peak_grid = compute_totals(plan)
        total_cost = compute_cost(plan, req.hours)

        applied = [d["directive_type"] for d in directives if d["applies"]]
        summary = (
            f"Optimized 24-hour schedule minimizing grid cost. "
            f"Applied directives: {', '.join(applied) if applied else 'none'}."
        )

        return OptimizeResponse(
            scenario_id=req.scenario_id,
            directive_interpretation=[
                DirectiveInterpretation(**d) for d in directives
            ],
            hourly_plan=plan,
            total_grid_kwh=round(total_grid, 4),
            total_cost_bdt=round(total_cost, 4),
            peak_grid_kwh=round(peak_grid, 4),
            plan_summary=summary,
        )

    except Exception:
        # No exception text, traceback or secret is ever returned.
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "Failed to compute schedule.",
            },
        )
