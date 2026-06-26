import json
import re

def extract_json_from_markdown(text: str) -> str:
    """
    Extract the first complete JSON object from a string.
    Handles markdown code blocks and stray text before/after.
    """
    text = text.strip()

    # Try extracting from a ```json ... ``` block first
    code_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_match:
        candidate = code_match.group(1).strip()
    else:
        candidate = text

    # Find the first '{' and track brace depth to get the complete object
    start = candidate.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in LLM output:\n{text}")

    depth = 0
    for i in range(start, len(candidate)):
        if candidate[i] == "{":
            depth += 1
        elif candidate[i] == "}":
            depth -= 1
            if depth == 0:
                return candidate[start:i+1]

    raise ValueError(f"Unmatched braces in LLM output:\n{text}")


def resolve_args(args: dict, state: dict) -> dict:
    resolved = {}

    for key, value in args.items():
        if isinstance(value, str) and value.startswith("$"):
            state_key = value[1:]
            if state_key not in state:
                raise KeyError(
                    f"State key '{state_key}' not found for argument '{key}'"
                )
            resolved[key] = state[state_key]
        else:
            resolved[key] = value

    return resolved


def validate_required_args(tool, args: dict):
    schema = tool.args_schema
    required_fields = [
        name for name, field in schema.model_fields.items()
        if field.is_required()
    ]

    missing = [f for f in required_fields if f not in args]

    if missing:
        raise ValueError(
            f"Tool '{tool.name}' missing required args: {missing}"
        )

