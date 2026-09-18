"""
Deterministic validation of LLM-produced directive interpretations.
Anything malformed is forced to a safe no_op instead of crashing or
being trusted blindly.
"""

from typing import Dict, Any, List

from models import ALLOWED_DIRECTIVE_TYPES


def _safe_no_op(note_index: int, reason: str) -> Dict[str, Any]:
    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": reason,
    }


def _valid_hours(hours) -> bool:
    if not isinstance(hours, list) or not hours:
        return False
    if any(not isinstance(h, int) or h < 0 or h > 23 for h in hours):
        return False
    if len(set(hours)) != len(hours):
        return False
    return hours == sorted(hours)


def validate_directive(raw: Dict[str, Any], note_index: int, num_notes: int, battery_capacity: float) -> Dict[str, Any]:
    """
    raw: one directive_interpretation-shaped dict as produced by the LLM
    Returns a guaranteed-valid directive dict (falls back to no_op on any problem).
    """
    try:
        if not isinstance(raw, dict):
            return _safe_no_op(note_index, "Malformed LLM output; defaulted to no_op.")

        idx = raw.get("note_index")
        if idx != note_index or not (0 <= note_index < num_notes):
            # still index correctly to this note even if LLM mislabeled it
            pass

        dtype = raw.get("directive_type")
        if dtype not in ALLOWED_DIRECTIVE_TYPES:
            return _safe_no_op(note_index, "Unsupported directive type; defaulted to no_op.")

        if dtype == "no_op":
            return {
                "note_index": note_index,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": raw.get("explanation", "Note does not affect the schedule."),
            }

        adj = raw.get("structured_adjustment")
        if not isinstance(adj, dict):
            return _safe_no_op(note_index, "Missing structured_adjustment; defaulted to no_op.")

        hours = adj.get("hours")
        if not _valid_hours(hours):
            return _safe_no_op(note_index, "Invalid hours array; defaulted to no_op.")

        if dtype == "solar_reduction":
            factor = adj.get("factor")
            if not isinstance(factor, (int, float)) or not (0 <= factor <= 1):
                return _safe_no_op(note_index, "Invalid solar factor; defaulted to no_op.")
            clean_adj = {"hours": hours, "factor": float(factor)}

        elif dtype == "minimum_battery_reserve":
            val = adj.get("minimum_energy_kwh")
            if not isinstance(val, (int, float)) or val < 0 or val > battery_capacity:
                return _safe_no_op(note_index, "Invalid reserve value; defaulted to no_op.")
            clean_adj = {"hours": hours, "minimum_energy_kwh": float(val)}

        elif dtype == "no_charge_window":
            clean_adj = {"hours": hours}

        elif dtype == "no_discharge_window":
            clean_adj = {"hours": hours}

        elif dtype == "max_grid_window":
            val = adj.get("max_grid_kwh")
            if not isinstance(val, (int, float)) or val < 0:
                return _safe_no_op(note_index, "Invalid grid cap; defaulted to no_op.")
            clean_adj = {"hours": hours, "max_grid_kwh": float(val)}

        else:
            return _safe_no_op(note_index, "Unhandled directive type; defaulted to no_op.")

        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": dtype,
            "structured_adjustment": clean_adj,
            "explanation": raw.get("explanation", ""),
        }

    except Exception:
        return _safe_no_op(note_index, "Validation error; defaulted to no_op.")


def validate_all(raw_directives: List[Dict[str, Any]], num_notes: int, battery_capacity: float) -> List[Dict[str, Any]]:
    """
    Ensures exactly one entry per note, in note_index order, each valid.
    """
    by_index = {}
    for raw in raw_directives or []:
        if isinstance(raw, dict) and isinstance(raw.get("note_index"), int):
            by_index[raw["note_index"]] = raw

    result = []
    for i in range(num_notes):
        raw = by_index.get(i)
        if raw is None:
            result.append(_safe_no_op(i, "No interpretation returned; defaulted to no_op."))
        else:
            result.append(validate_directive(raw, i, num_notes, battery_capacity))
    return result
