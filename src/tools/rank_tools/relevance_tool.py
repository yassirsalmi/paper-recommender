from typing import ClassVar, Type, List, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from llm.llm import LLM
import ast
import re

class RelevanceToolArgs(BaseModel):
    user_query: str = Field(description="User query")
    papers: List[Dict[str, Any]] = Field(description="Papers to check relevance")


class RelevanceTool(BaseTool):
    """
    Uses an LLM to filter papers based on relevance to the user query
    """
    name: ClassVar[str] = "paper_relevance"
    description: ClassVar[str] = "Filters papers by relevance"
    args_schema: ClassVar[Type[BaseModel]] = RelevanceToolArgs
    output_key: str = "relevant_papers"

    _llm: LLM = PrivateAttr()

    def __init__(self, llm: LLM):
        super().__init__()
        self._llm = llm

    def _run(self, user_query: str, papers: List[Dict[str, Any]], max_papers: int = 30):
        # Limit papers to avoid overwhelming the LLM
        papers = papers[:max_papers]
        summaries = [paper["summary"] for paper in papers]

        prompt = f"""
You are a research assistant.
For each paper summary, decide if it is relevant to the user query.

Return ONLY a valid Python list of strings.
Allowed values: "yes", "no".

User query:
{user_query}

Paper summaries:
{summaries}
"""

        result = self._llm.invoke(prompt)

        try:
            text = result.content.strip()
            # Find a list that looks like ["yes", "no", ...], ['yes', 'no', ...] or [1, 0, ...]
            match = re.search(r"""\[(?:\s*["'](?:yes|no)["']\s*,?\s*)+\]""", text, re.IGNORECASE)
            if not match:
                match = re.search(r"""\[[\s,\d]+\]""", text)
            if not match:
                raise ValueError(f"No decision list found in:\n{text[:500]}")
            decisions = ast.literal_eval(match.group())
        except Exception:
            raise ValueError("Failed to parse relevance output")

        relevant_papers = [
            paper for paper, decision in zip(papers, decisions)
            if (isinstance(decision, str) and decision.lower() == "yes")
            or (isinstance(decision, (int, float)) and decision == 1)
        ]

        return {
            "relevant_papers": relevant_papers
        }
