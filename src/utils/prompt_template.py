

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


def func_planning_prompt(user_query, tools_list):
    return f"""
You are a research planning agent. Your role is to produce a step-by-step plan
to answer the user's query as effectively as possible.

User query:
"{user_query}"

Available tools:
{tools_list}

Instructions:
- Return JSON only.
- And the JSON returned must follow this schema exactly:
{{
    "steps": [
        {{"action": "action_name_1", "tool": "tool_name_1", "args": {{}}}},
        {{"action": "action_name_2", "tool": "tool_name_2", "args": {{}}}},
        {{"action": "action_name_3", "tool": "tool_name_3", "args": {{}}}}
    ]
}}

- Do NOT include any explanation, markdown, or extra text.
- Only return JSON; it must be valid and complete.

Remember: the steps should be ordered to produce the best research results.
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