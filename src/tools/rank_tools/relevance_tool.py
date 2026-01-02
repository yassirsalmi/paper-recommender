from typing import ClassVar, Type, List, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from llm.llm import LLM
import ast

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

    _llm: LLM = PrivateAttr()

    def __init__(self, llm: LLM):
        super().__init__()
        self._llm = llm

    def _run(self, user_query: str, papers: List[Dict[str, Any]]):
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
            decisions = ast.literal_eval(result.content)
        except Exception:
            raise ValueError("Failed to parse relevance output")

        relevant_papers = [
            paper for paper, decision in zip(papers, decisions)
            if decision.lower() == "yes"
        ]

        return {
            "relevant_papers": relevant_papers
        }
