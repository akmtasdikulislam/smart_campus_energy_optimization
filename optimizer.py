"""
Core LP optimizer for the 24-hour energy schedule.
Directives (already validated by guardrails.py) are applied as extra
constraints before solving.
"""

from typing import List, Dict, Any
import pulp

from models import HourData, Battery, HourlyPlan


def solve_schedule(
    hours: List[HourData],
    battery: Battery,
    directives: List[Dict[str, Any]],
) -> List[HourlyPlan]:
    H = list(range(24))
    demand = {h.hour: h.demand_kwh for h in hours}
    base_solar = {h.hour: h.solar_kwh for h in hours}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}

    # ---- Apply directive effects to model inputs ----
    effective_solar = dict(base_solar)
    min_reserve = {h: battery.minimum_energy_kwh for h in H}
    no_charge_hours = set()
    no_discharge_hours = set()
    max_grid = {}  # hour -> cap

    for d in directives:
        dtype = d["directive_type"]
        adj = d.get("structured_adjustment") or {}
        if dtype == "solar_reduction":
            factor = adj["factor"]
            for h in adj["hours"]:
                effective_solar[h] = base_solar[h] * factor
        elif dtype == "minimum_battery_reserve":
            req = adj["minimum_energy_kwh"]
            for h in adj["hours"]:
                min_reserve[h] = max(min_reserve[h], req)
        elif dtype == "no_charge_window":
            no_charge_hours.update(adj["hours"])
        elif dtype == "no_discharge_window":
            no_discharge_hours.update(adj["hours"])
        elif dtype == "max_grid_window":
            cap = adj["max_grid_kwh"]
            for h in adj["hours"]:
                max_grid[h] = cap if h not in max_grid else min(max_grid[h], cap)
        # no_op -> no effect

    # ---- Build LP model ----
    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)

    grid = {h: pulp.LpVariable(f"grid_{h}", lowBound=0) for h in H}
    solar_used = {h: pulp.LpVariable(f"solar_used_{h}", lowBound=0) for h in H}
    charge = {h: pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=battery.max_charge_kwh_per_hour) for h in H}
    discharge = {h: pulp.LpVariable(f"discharge_{h}", lowBound=0, upBound=battery.max_discharge_kwh_per_hour) for h in H}
    energy = {h: pulp.LpVariable(f"energy_{h}", lowBound=0, upBound=battery.capacity_kwh) for h in H}

    prob += pulp.lpSum(grid[h] * tariff[h] for h in H)

    for h in H:
        # energy balance
        prob += grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h]
        # solar usage bound
        prob += solar_used[h] <= effective_solar[h]
        # battery state transition
        prev_energy = battery.initial_energy_kwh if h == 0 else energy[h - 1]
        prob += energy[h] == prev_energy + charge[h] - discharge[h]
        # reserve (min) bound for this hour
        prob += energy[h] >= min_reserve[h]
        # directive: no charge / no discharge windows
        if h in no_charge_hours:
            prob += charge[h] == 0
        if h in no_discharge_hours:
            prob += discharge[h] == 0
        # directive: max grid cap
        if h in max_grid:
            prob += grid[h] <= max_grid[h]

    # end-of-day battery neutrality
    prob += energy[23] == battery.initial_energy_kwh

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Optimizer failed to find an optimal solution: {pulp.LpStatus[status]}")

    plan: List[HourlyPlan] = []
    for h in H:
        c = charge[h].value() or 0.0
        dch = discharge[h].value() or 0.0
        if c > 1e-6:
            action, amount = "charge", c
        elif dch > 1e-6:
            action, amount = "discharge", dch
        else:
            action, amount = "idle", 0.0

        plan.append(
            HourlyPlan(
                hour=h,
                grid_kwh=round(max(grid[h].value() or 0.0, 0.0), 4),
                solar_used_kwh=round(max(solar_used[h].value() or 0.0, 0.0), 4),
                battery_action=action,
                battery_kwh=round(amount, 4),
                battery_energy_after_kwh=round(energy[h].value() or 0.0, 4),
            )
        )
    return plan


def compute_totals(plan: List[HourlyPlan]):
    total_grid = sum(p.grid_kwh for p in plan)
    peak_grid = max(p.grid_kwh for p in plan)
    return total_grid, peak_grid


def compute_cost(plan: List[HourlyPlan], hours: List[HourData]) -> float:
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}
    return sum(p.grid_kwh * tariff[p.hour] for p in plan)
