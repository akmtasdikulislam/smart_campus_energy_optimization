"""
Deterministic validation of LLM-produced directive interpretations.

The model output is UNTRUSTED. Anything malformed - wrong type, missing field,
NaN/Infinity, boolean-as-number, bad note mapping, non-string explanation - is
forced to a safe no_op instead of crashing or being trusted blindly.

Invariants guaranteed to callers:
  * exactly one entry per operator note, in ascending note_index order,
  * directive_type is always one of ALLOWED_DIRECTIVE_TYPES,
  * explanation is always a str,
  * applies is False iff directive_type == "no_op",
  * structured_adjustment is None for no_op and a fully cleaned dict of
    finite floats / valid hour lists otherwise.
"""

import math
from typing import Any, Dict, List, Optional

from models import ALLOWED_DIRECTIVE_TYPES


def _safe_no_op(note_index: int, reason: str) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": reason,
    }


def _finite_number(value: Any) -> bool:
    """True only for real, finite numbers. Rejects bool (subclass of int),
    NaN and +/-Infinity."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_hours(hours: Any) -> bool:
    if not isinstance(hours, list) or not hours:
        return False
    for h in hours:
        if not isinstance(h, int) or isinstance(h, bool):
            return False
        if h < 0 or h > 23:
            return False
    if len(set(hours)) != len(hours):
        return False
    return hours == sorted(hours)


def _clean_explanation(raw: Dict[str, Any], default: str) -> Optional[str]:
    """Return a usable string explanation, or None if the model supplied a
    non-string value (null / list / object) that would break the response model."""
    value = raw.get("explanation", default)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value else default


def validate_directive(
    raw: Dict[str, Any],
    note_index: int,
    num_notes: int,
    battery_capacity: float,
) -> Dict[str, Any]:
    """
    raw: one directive_interpretation-shaped dict as produced by the LLM.
    Returns a guaranteed-valid directive dict (falls back to no_op on any problem).
    """
    try:
        if not isinstance(raw, dict):
            return _safe_no_op(note_index, "Malformed LLM output; defaulted to no_op.")

        # --- note mapping integrity (not silently ignored any more) ---
        if not (0 <= note_index < num_notes):
            return _safe_no_op(note_index, "Invalid note index; defaulted to no_op.")
        idx = raw.get("note_index")
        if type(idx) is not int or idx != note_index:
            return _safe_no_op(note_index, "Invalid note mapping; defaulted to no_op.")

        # --- the capacity we compare reserves against must itself be sane ---
        if not _finite_number(battery_capacity) or float(battery_capacity) < 0:
            return _safe_no_op(
                note_index, "Invalid battery capacity; defaulted to no_op."
            )

        dtype = raw.get("directive_type")
        if not isinstance(dtype, str) or dtype not in ALLOWED_DIRECTIVE_TYPES:
            return _safe_no_op(
                note_index, "Unsupported directive type; defaulted to no_op."
            )

        applies = raw.get("applies")
        explanation = _clean_explanation(raw, "Note does not affect the schedule.")
        if explanation is None:
            return _safe_no_op(
                note_index, "Malformed explanation; defaulted to no_op."
            )

        if dtype == "no_op":
            if applies is not False:
                return _safe_no_op(
                    note_index, "Invalid no_op applies value; defaulted to no_op."
                )
            return {
                "note_index": note_index,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": explanation,
            }

        if applies is not True:
            return _safe_no_op(
                note_index, "Invalid applies value; defaulted to no_op."
            )

        adj = raw.get("structured_adjustment")
        if not isinstance(adj, dict):
            return _safe_no_op(
                note_index, "Missing structured_adjustment; defaulted to no_op."
            )

        hours = adj.get("hours")
        if not _valid_hours(hours):
            return _safe_no_op(note_index, "Invalid hours array; defaulted to no_op.")

        if dtype == "solar_reduction":
            factor = adj.get("factor")
            if not _finite_number(factor) or not 0.0 <= float(factor) <= 1.0:
                return _safe_no_op(
                    note_index, "Invalid solar factor; defaulted to no_op."
                )
            clean_adj = {"hours": list(hours), "factor": float(factor)}

        elif dtype == "minimum_battery_reserve":
            val = adj.get("minimum_energy_kwh")
            if (
                not _finite_number(val)
                or float(val) < 0
                or float(val) > float(battery_capacity)
            ):
                return _safe_no_op(
                    note_index, "Invalid reserve value; defaulted to no_op."
                )
            clean_adj = {"hours": list(hours), "minimum_energy_kwh": float(val)}

        elif dtype in ("no_charge_window", "no_discharge_window"):
            clean_adj = {"hours": list(hours)}

        elif dtype == "max_grid_window":
            val = adj.get("max_grid_kwh")
            # NOTE: `val < 0` alone would accept +Infinity; _finite_number closes that.
            if not _finite_number(val) or float(val) < 0:
                return _safe_no_op(note_index, "Invalid grid cap; defaulted to no_op.")
            clean_adj = {"hours": list(hours), "max_grid_kwh": float(val)}

        else:
            return _safe_no_op(
                note_index, "Unhandled directive type; defaulted to no_op."
            )

        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": dtype,
            "structured_adjustment": clean_adj,
            "explanation": explanation,
        }

    except Exception:
        # Last-resort net only; every known malformed shape is handled above.
        return _safe_no_op(note_index, "Validation error; defaulted to no_op.")


def validate_all(
    raw_directives: Any,
    num_notes: int,
    battery_capacity: float,
) -> List[Dict[str, Any]]:
    """
    Ensures exactly one entry per note, in note_index order, each valid.
    Duplicate or out-of-range indices are rejected explicitly instead of
    silently overwriting / being dropped.
    """
    by_index: Dict[int, Dict[str, Any]] = {}
    invalid_indices = set()

    if not isinstance(raw_directives, list):
        raw_directives = []

    for raw in raw_directives:
        if not isinstance(raw, dict) or type(raw.get("note_index")) is not int:
            continue
        index = raw["note_index"]
        if not (0 <= index < num_notes) or index in by_index:
            invalid_indices.add(index)
            continue
        by_index[index] = raw

    result: List[Dict[str, Any]] = []
    for index in range(num_notes):
        if index in invalid_indices:
            result.append(
                _safe_no_op(
                    index, "Duplicate or invalid note mapping; defaulted to no_op."
                )
            )
        elif index not in by_index:
            result.append(
                _safe_no_op(index, "No interpretation returned; defaulted to no_op.")
            )
        else:
            result.append(
                validate_directive(by_index[index], index, num_notes, battery_capacity)
            )
    return result
