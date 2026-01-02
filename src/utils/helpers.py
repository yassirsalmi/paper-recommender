import re

def extract_json_from_markdown(text: str) -> dict:
    """
    Extract JSON from a string
    """
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"```$", "", text.strip())

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM output:\n{text}")

    return match.group()


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

