

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
You are a research planning agent.

Your task is to produce a step-by-step plan to answer the user's query.

User query:
"{user_query}"

Available tools:
{tools_list}

TOOLS AND REQUIRED ARGUMENTS:

paper_search:
- args:
  - search_query (string) → use "$user_query"
  - limit (int, optional)

paper_relevance:
- args:
  - user_query (string) → use "$user_query"
  - papers (list) → use "$papers"

paper_ranker:
- args:
  - papers (list) → use "$relevant_papers"

paper_summarizer:
- args:
  - papers (list) → use "$ranked_papers"

MANDATORY RULES:
1. The plan MUST start with "paper_search".
2. The plan MUST include "paper_relevance" immediately AFTER "paper_search".
3. The plan MUST NOT include "paper_ranker" or "paper_summarizer" BEFORE "paper_relevance".
4. Any plan that does not include "paper_relevance" is INVALID.
5. Use each tool at most once.
6. Steps must be in a logical execution order.

OUTPUT FORMAT:
- Return JSON only.
- The JSON must follow this schema exactly:
{{
    "steps": [
        {{"action": "action_name_1", "tool": "tool_name_1", "args": {{}}}},
        {{"action": "action_name_2", "tool": "tool_name_2", "args": {{}}}},
        {{"action": "action_name_3", "tool": "tool_name_3", "args": {{}}}}
    ]
}}

IMPORTANT:
- Do NOT include explanations.
- Do NOT include markdown.
- Return ONLY valid JSON.
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