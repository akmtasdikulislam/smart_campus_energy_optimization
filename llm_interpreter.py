"""
Calls Gemini (Google AI Studio) to convert operator notes into structured directives.
Output is treated as UNTRUSTED until guardrails.py validates it.

Setup:
    pip install google-genai
    export GEMINI_API_KEY="...."      # from https://aistudio.google.com/apikey
"""

import json
import os
import re
from typing import List, Dict, Any

from dotenv import load_dotenv  # Add this import
from google import genai
from google.genai import types

# Load environment variables from the .env file
load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

# Now os.environ.get will successfully find the key from your .env file
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
- Never invent demand, tariff, or battery parameter changes.
- If a note is irrelevant to the energy schedule (e.g. menu changes, unrelated campus news), mark it no_op.
- Return ONLY a JSON array, one object per note, in the SAME order as the notes given, with this exact shape:
  {"note_index": int, "applies": bool, "directive_type": string, "structured_adjustment": object|null, "explanation": string}
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


def interpret_notes(operator_notes: List[str]) -> List[Dict[str, Any]]:
    """
    Returns a list of RAW (untrusted) directive dicts, one per note.
    On any failure, returns an empty list so guardrails.validate_all
    can safely fill in no_op defaults for every note.
    """
    notes_block = "\n".join(f"{i}: {n}" for i, n in enumerate(operator_notes))
    user_prompt = f"Operator notes:\n{notes_block}\n\nReturn the JSON array now."

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
        # Safe failure: caller/guardrails will default every note to no_op
        return []
