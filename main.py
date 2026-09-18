from fastapi import FastAPI
from fastapi.responses import JSONResponse

from models import ScenarioRequest, OptimizeResponse, DirectiveInterpretation
from llm_interpreter import interpret_notes
from guardrails import validate_all
from optimizer import solve_schedule, compute_totals, compute_cost

app = FastAPI(title="GridWise LLM API")


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

        response = OptimizeResponse(
            scenario_id=req.scenario_id,
            directive_interpretation=[DirectiveInterpretation(**d) for d in directives],
            hourly_plan=plan,
            total_grid_kwh=round(total_grid, 4),
            total_cost_bdt=round(total_cost, 4),
            peak_grid_kwh=round(peak_grid, 4),
            plan_summary=summary,
        )
        return response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "Failed to compute schedule."},
        )
