"""
Request / response contract for the GridWise LLM API.

Validation policy (strict):
  * the request must be an EXACT contract instance (no extra fields anywhere),
  * `hours` must contain exactly one entry per hour 0..23 in canonical order,
  * every numeric value must be a real, finite, non-negative number,
  * battery values must be mutually consistent with capacity.

Anything violating the above is rejected at the API boundary with a controlled
validation error, so downstream code (optimizer.py) may assume a well-formed day.
"""

from math import isfinite
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------- shared numeric helper ----------

def _finite_number(value: Any, *, nonnegative: bool = False) -> float:
    """Accept only real, finite numbers. Booleans are rejected explicitly
    because bool is a subclass of int in Python."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError("must be finite")
    if nonnegative and value < 0:
        raise ValueError("must be non-negative")
    return value


# ---------- Request ----------

class HourData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(ge=0, le=23)
    demand_kwh: float
    solar_kwh: float
    tariff_bdt_per_kwh: float

    @field_validator("hour", mode="before")
    @classmethod
    def hour_must_be_integer(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("hour must be an integer")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("hour must be an integer")
        if not isinstance(value, (int, float)):
            raise ValueError("hour must be an integer")
        return int(value)

    @field_validator("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh", mode="before")
    @classmethod
    def values_must_be_finite_and_nonnegative(cls, value: Any) -> float:
        return _finite_number(value, nonnegative=True)


class Battery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float

    @field_validator(
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
        mode="before",
    )
    @classmethod
    def battery_values_must_be_finite_and_nonnegative(cls, value: Any) -> float:
        return _finite_number(value, nonnegative=True)

    @model_validator(mode="after")
    def battery_relationships_are_valid(self):
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError("initial_energy_kwh exceeds capacity_kwh")
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh")
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            raise ValueError("initial_energy_kwh is below minimum_energy_kwh")
        return self


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourData] = Field(min_length=24, max_length=24)
    battery: Battery

    @field_validator("scenario_id")
    @classmethod
    def scenario_id_must_be_nonempty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("scenario_id must be a non-empty string")
        return value

    @field_validator("operator_notes")
    @classmethod
    def notes_must_be_nonempty_strings(cls, value: List[str]) -> List[str]:
        if any(not isinstance(note, str) or not note.strip() for note in value):
            raise ValueError("operator_notes entries must be non-empty strings")
        return value

    @model_validator(mode="after")
    def hours_must_cover_the_day(self):
        actual = [item.hour for item in self.hours]
        expected = list(range(24))
        if actual != expected:
            raise ValueError(
                "hours must contain exactly one ordered entry for every hour 0..23"
            )
        return self


# ---------- Response ----------

class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: str
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str


class HourlyPlan(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str  # charge / discharge / idle
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# ---------- Internal directive representation ----------

ALLOWED_DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}
