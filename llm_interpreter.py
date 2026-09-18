"""
Calls Gemini (Google AI Studio) to convert operator notes into structured directives.
Output is treated as UNTRUSTED until guardrails.py validates it.

Setup:
    pip install -r requirements.txt
    export GEMINI_API_KEY="..."      # from https://aistudio.google.com/apikey

The client is built lazily on first use. A missing or rejected key therefore
degrades to an all-no_op interpretation instead of killing the process at import
time and taking /health down with it.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

# Keep well inside the 30 s response budget so a slow provider call cannot hold
# the HTTP request open; on timeout we fall back to all-no_op.
REQUEST_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "8000"))

_client: Optional[Any] = None
_client_initialized = False


def _get_client():
    """Build the Gemini client once, on demand. Returns None if unavailable."""
    global _client, _client_initialized
    if _client_initialized:
        return _client
    _client_initialized = True

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        if api_key.startswith("AQ"):
            _client = genai.Client(
                vertexai=True, 
                project="gen-lang-client-0778426882",
                location="us-central1"
            )
        else:
            _client = genai.Client(api_key=api_key)
    except Exception:
        _client = None
    return _client


SYSTEM_PROMPT = """You convert campus operator notes into structured energy-schedule directives.

Supported directive types (use exactly these, nothing else):
- solar_reduction: {"hours":[...], "factor": number}   (factor = fraction of solar that REMAINS, e.g. 80% reduction -> factor 0.2)
- minimum_battery_reserve: {"hours":[...], "minimum_energy_kwh": number}
- no_charge_window: {"hours":[...]}
- no_discharge_window: {"hours":[...]}
- max_grid_window: {"hours":[...], "max_grid_kwh": number}
- no_op: structured_adjustment must be null (note does not affect the energy schedule)

Rules:
- Hours are whole-hour intervals, start included end excluded. "1 PM to 3 PM" -> hours [13,14].
- "hours" arrays must contain unique integers 0-23 in ascending order.
- All numbers must be finite JSON numbers. Never emit NaN, Infinity, null or a string in a numeric field.
- "explanation" must always be a plain non-empty string.
- Never invent demand, tariff, or battery parameter changes.
- If a note is irrelevant to the energy schedule (e.g. menu changes, unrelated campus news), mark it no_op.
- Return ONLY a JSON array, one object per note, in the SAME order as the notes given, with this exact shape:
  {"note_index": int, "applies": bool, "directive_type": string, "structured_adjustment": object|null, "explanation": string}
- note_index must be the 0-based index of the note, used exactly once across the array.
- applies must be false only for no_op; true for every other directive type.
- Do not include any text outside the JSON array. No markdown fences.

Examples:
Note: "Solar output will drop to about 20% from 1 PM to 3 PM."
-> {"note_index":0,"applies":true,"directive_type":"solar_reduction","structured_adjustment":{"hours":[13,14],"factor":0.2},"explanation":"Solar reduced to 20% during stated window."}

Note: "Do not charge the battery between 2 PM and 4 PM."
-> {"note_index":0,"applies":true,"directive_type":"no_charge_window","structured_adjustment":{"hours":[14,15]},"explanation":"Charging disabled during stated window."}

Note: "Keep at least 120 kWh in reserve from 6 PM until 9 PM."
-> {"note_index":0,"applies":true,"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[18,19,20],"minimum_energy_kwh":120},"explanation":"Reserve requirement during stated window."}

Note: "The cafeteria menu changes tomorrow."
-> {"note_index":0,"applies":false,"directive_type":"no_op","structured_adjustment":null,"explanation":"Not related to the energy schedule."}
"""


def _extract_json_array(text: str):
    """Best-effort extraction if the model wraps output in extra text/fences."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def interpret_notes(
    operator_notes: List[str],
    battery_capacity_kwh: float
) -> List[Dict[str, Any]]:
    """
    Returns a list of RAW (untrusted) directive dicts, one per note.
    On any failure - missing key, provider error, timeout, unparsable output -
    returns an empty list so guardrails.validate_all can safely fill in no_op
    defaults for every note.
    """
    client = _get_client()
    if client is None:
        return []

    notes_block = "\n".join(f"{i}: {n}" for i, n in enumerate(operator_notes))
    
    user_prompt = f"""
Scenario context:
- Battery capacity: {battery_capacity_kwh} kWh

Operator notes:
{notes_block}

If a reserve is expressed as a percentage or fraction of battery
capacity, convert it to absolute kWh using the supplied capacity.

Return the JSON array now.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=2000,
                response_mime_type="application/json",
                # 2.5 models think by default; 0 disables it for this cheap parsing task.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (response.text or "").strip()
        if not text:
            return []
        parsed = _extract_json_array(text)
        if not isinstance(parsed, list):
            return []
        return parsed
    except Exception:
        # Safe failure: caller/guardrails will default every note to no_op.
        # Never log or re-raise anything that could contain the API key.
        return []
