import json

def func_judgemental_prompt(user_query, available_tools, plan):
    return f"""
you are a judge LLM and you role is to give your feedback on a plan generated 
to resolve a user query while using the provided tools

User query:
"{user_query}"

Tools:
"{available_tools}"

Plan:
"{plan}"

please provide your feedback on a scale for 0 to 10 don't add anything else to you output.
"""


def func_planning_prompt(user_query, tools_schema):
    return f"""
You are a research planning agent.

Your task is to produce a step-by-step plan to answer the user's query.

User query:
"{user_query}"

Available tools:
{tools_schema}

STATE REFERENCES:
- Each tool writes its output into a shared state under a key (shown as "writes to state: $key")
- To use a previous tool's output as an arg, reference it with a "$" prefix: e.g. "$papers"
- "$user_query" always refers to the original user query

MANDATORY RULES:
1. Steps must be in logical execution order.
2. Args that reference prior outputs MUST use $variable syntax, not placeholders like [papers].

OUTPUT FORMAT:
Return JSON only, following this schema exactly:
{{
    "steps": [
        {{"action": "action_name", "tool": "tool_name", "args": {{"arg1": "$state_var", "arg2": "literal_value"}}}}
    ]
}}

IMPORTANT: Return ONLY valid JSON. No markdown, no explanation.
"""


def func_synthesis_prompt(user_query, intermediate_results):
    return f"""
You are a research assistant.

User query:
"{user_query}"

Here are the research findings:
{intermediate_results}

Provide a concise list of recommended papers and
explain why each is relevant.
"""

def _build_explanation_prompt(sections):
        """
        Build a structured explanation prompt for the LLM.
        """

        return f"""
You are a research assistant.

Explain the following paper clearly and concisely.

Focus on:
- Main contributions
- Core method
- How it impacts LLM inference
- Limitations

Paper sections:
{json.dumps(sections, indent=2)}
"""