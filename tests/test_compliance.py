"""
Regression tests for every defect raised in the compliance review.

Run:  pytest -q
The Gemini call is stubbed out, so no API key and no network are needed.
"""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrails import validate_all, validate_directive  # noqa: E402
from models import Battery, ScenarioRequest  # noqa: E402
from optimizer import solve_schedule, verify_plan  # noqa: E402
from pydantic import ValidationError  # noqa: E402

CAPACITY = 200.0


def base_request():
    return {
        "scenario_id": "scn-1",
        "operator_notes": ["Solar drops to 20% from 1 PM to 3 PM."],
        "hours": [
            {
                "hour": h,
                "demand_kwh": 40.0,
                "solar_kwh": 30.0 if 6 <= h <= 17 else 0.0,
                "tariff_bdt_per_kwh": 12.0 if 17 <= h <= 22 else 6.0,
            }
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": CAPACITY,
            "initial_energy_kwh": 100.0,
            "minimum_energy_kwh": 20.0,
            "max_charge_kwh_per_hour": 40.0,
            "max_discharge_kwh_per_hour": 40.0,
        },
    }


# ---------------- CF-1: request contract ----------------

def test_valid_request_is_accepted():
    ScenarioRequest(**base_request())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["hours"].pop(5),                                # missing hour
        lambda r: r["hours"].append(copy.deepcopy(r["hours"][0])),  # 25 hours
        lambda r: r["hours"][3].update(hour=4),                     # duplicate hour
        lambda r: r["hours"].reverse(),                             # wrong order
        lambda r: r["hours"][0].update(demand_kwh=-5.0),            # negative
        lambda r: r["hours"][0].update(demand_kwh=float("inf")),    # infinity
        lambda r: r["hours"][0].update(demand_kwh=float("nan")),    # NaN
        lambda r: r["hours"][0].update(hour=99),                    # out of range
        lambda r: r["operator_notes"].clear(),                      # no notes
        lambda r: r["operator_notes"].append("   "),                # blank note
        lambda r: r["operator_notes"].extend(["a", "b", "c"]),      # >3 notes
        lambda r: r.update(scenario_id=""),                         # empty id
        lambda r: r["battery"].update(capacity_kwh=0.0),            # zero capacity
        lambda r: r["battery"].update(initial_energy_kwh=999.0),    # above capacity
        lambda r: r["battery"].update(minimum_energy_kwh=999.0),    # reserve > capacity
        lambda r: r["battery"].update(max_charge_kwh_per_hour=-1),  # negative limit
        lambda r: r.update(unexpected_field=1),                     # extra top-level
        lambda r: r["battery"].update(extra=1),                     # extra nested
        lambda r: r["hours"][0].update(extra=1),                    # extra in hour
    ],
)
def test_malformed_requests_are_rejected(mutate):
    req = base_request()
    mutate(req)
    with pytest.raises(ValidationError):
        ScenarioRequest(**req)


# ---------------- CF-2: guardrails ----------------

def _raw(**kw):
    out = {
        "note_index": 0,
        "applies": True,
        "directive_type": "max_grid_window",
        "structured_adjustment": {"hours": [1, 2], "max_grid_kwh": 10.0},
        "explanation": "cap",
    }
    out.update(kw)
    return out


def _is_no_op(result):
    return (
        result["directive_type"] == "no_op"
        and result["applies"] is False
        and result["structured_adjustment"] is None
        and isinstance(result["explanation"], str)
    )


def test_infinite_grid_cap_is_rejected():
    raw = _raw(structured_adjustment={"hours": [1], "max_grid_kwh": float("inf")})
    assert _is_no_op(validate_directive(raw, 0, 1, CAPACITY))


def test_boolean_is_not_accepted_as_a_number():
    raw = _raw(
        directive_type="solar_reduction",
        structured_adjustment={"hours": [1], "factor": True},
    )
    assert _is_no_op(validate_directive(raw, 0, 1, CAPACITY))


@pytest.mark.parametrize("bad", [None, 123, ["a"], {"x": 1}])
def test_non_string_explanation_becomes_no_op(bad):
    result = validate_directive(_raw(explanation=bad), 0, 1, CAPACITY)
    assert _is_no_op(result)
    assert isinstance(result["explanation"], str)


def test_wrong_applies_flag_is_rejected():
    assert _is_no_op(validate_directive(_raw(applies=False), 0, 1, CAPACITY))
    assert _is_no_op(
        validate_directive(
            _raw(directive_type="no_op", applies=True, structured_adjustment=None),
            0, 1, CAPACITY,
        )
    )


def test_bad_note_mapping_is_rejected():
    assert _is_no_op(validate_directive(_raw(note_index=7), 0, 1, CAPACITY))


def test_invalid_battery_capacity_is_rejected():
    assert _is_no_op(validate_directive(_raw(), 0, 1, float("inf")))


@pytest.mark.parametrize(
    "hours", [[3, 1], [1, 1], [-1], [24], ["2"], [1.5], [], "1,2", None]
)
def test_invalid_hour_arrays_become_no_op(hours):
    raw = _raw(structured_adjustment={"hours": hours, "max_grid_kwh": 10.0})
    assert _is_no_op(validate_directive(raw, 0, 1, CAPACITY))


def test_unsupported_directive_type_becomes_no_op():
    assert _is_no_op(validate_directive(_raw(directive_type="shutdown"), 0, 1, CAPACITY))


# ---------------- W-1: note index integrity ----------------

def test_duplicate_llm_indices_are_rejected_not_collapsed():
    raws = [_raw(note_index=0), _raw(note_index=0)]
    result = validate_all(raws, num_notes=2, battery_capacity=CAPACITY)
    assert [r["note_index"] for r in result] == [0, 1]
    assert all(_is_no_op(r) for r in result)


def test_one_entry_per_note_in_order():
    result = validate_all([], num_notes=3, battery_capacity=CAPACITY)
    assert [r["note_index"] for r in result] == [0, 1, 2]
    assert all(_is_no_op(r) for r in result)


@pytest.mark.parametrize("garbage", [None, "text", 42, [None, "x", 7]])
def test_garbage_llm_output_never_raises(garbage):
    result = validate_all(garbage, num_notes=2, battery_capacity=CAPACITY)
    assert len(result) == 2


# ---------------- CF-3 + optimizer invariants ----------------

def _solve(directives=None, request=None):
    req = ScenarioRequest(**(request or base_request()))
    return req, solve_schedule(req.hours, req.battery, directives or [])


def test_plan_replays_exactly():
    req, plan = _solve()
    demand = {h.hour: h.demand_kwh for h in req.hours}
    verify_plan(plan, demand, req.battery)  # raises on any violation
    assert len(plan) == 24
    assert abs(sum(
        p.battery_kwh * (1 if p.battery_action == "charge" else -1)
        for p in plan if p.battery_action != "idle"
    )) < 1e-3


def test_no_hour_charges_and_discharges_at_once():
    _, plan = _solve()
    for p in plan:
        assert p.battery_action in ("charge", "discharge", "idle")
        if p.battery_action == "idle":
            assert p.battery_kwh == 0.0


def test_zero_tariff_degeneracy_still_replays():
    """Free electricity removes the natural penalty on pointless cycling."""
    req = base_request()
    for h in req["hours"]:
        h["tariff_bdt_per_kwh"] = 0.0
    parsed, plan = _solve(request=req)
    verify_plan(plan, {h.hour: h.demand_kwh for h in parsed.hours}, parsed.battery)


def test_all_five_directives_take_effect():
    directives = [
        {"note_index": 0, "applies": True, "directive_type": "solar_reduction",
         "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
         "explanation": ""},
        {"note_index": 1, "applies": True, "directive_type": "no_charge_window",
         "structured_adjustment": {"hours": [10, 11]}, "explanation": ""},
        {"note_index": 2, "applies": True, "directive_type": "no_discharge_window",
         "structured_adjustment": {"hours": [18, 19]}, "explanation": ""},
        {"note_index": 3, "applies": True, "directive_type": "minimum_battery_reserve",
         "structured_adjustment": {"hours": [20, 21], "minimum_energy_kwh": 90.0},
         "explanation": ""},
        {"note_index": 4, "applies": True, "directive_type": "max_grid_window",
         "structured_adjustment": {"hours": [8, 9], "max_grid_kwh": 35.0},
         "explanation": ""},
    ]
    req, plan = _solve(directives)
    by_hour = {p.hour: p for p in plan}

    assert by_hour[10].battery_action != "charge"
    assert by_hour[11].battery_action != "charge"
    assert by_hour[18].battery_action != "discharge"
    assert by_hour[19].battery_action != "discharge"
    assert by_hour[20].battery_energy_after_kwh >= 90.0 - 1e-3
    assert by_hour[21].battery_energy_after_kwh >= 90.0 - 1e-3
    assert by_hour[8].grid_kwh <= 35.0 + 1e-3
    assert by_hour[9].grid_kwh <= 35.0 + 1e-3
    assert by_hour[13].solar_used_kwh <= 30.0 * 0.2 + 1e-3
    verify_plan(plan, {h.hour: h.demand_kwh for h in req.hours}, req.battery)


def test_end_of_day_neutrality_is_exact_at_reported_precision():
    req, plan = _solve()
    assert plan[23].battery_energy_after_kwh == pytest.approx(
        req.battery.initial_energy_kwh, abs=1e-4
    )


def test_optimizer_rejects_incomplete_hours():
    req = ScenarioRequest(**base_request())
    with pytest.raises(ValueError):
        solve_schedule(req.hours[:-1], req.battery, [])


# ---------------- API surface ----------------

def test_endpoints_and_error_handling(monkeypatch):
    import llm_interpreter

    monkeypatch.setattr(llm_interpreter, "interpret_notes", lambda notes: [])
    from fastapi.testclient import TestClient
    import main

    monkeypatch.setattr(main, "interpret_notes", lambda notes: [])
    client = TestClient(main.app)

    assert client.get("/health").json() == {"status": "ok"}

    ok = client.post("/optimize-energy", json=base_request())
    assert ok.status_code == 200
    body = ok.json()
    assert body["scenario_id"] == "scn-1"
    assert len(body["hourly_plan"]) == 24
    assert len(body["directive_interpretation"]) == 1

    bad = base_request()
    bad["hours"].pop()
    assert client.post("/optimize-energy", json=bad).status_code in (400, 422)

    # documentation routes are disabled
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
