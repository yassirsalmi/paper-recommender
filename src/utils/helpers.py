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
