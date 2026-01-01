from typing import ClassVar, Type, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from llm.llm import LLM

class RelevanceToolArgs(BaseModel):
    user_query: str = Field(
        description="User Query"
    )
    summaries: List[str] = Field(
        description="Summary Of The Paper"
    )

class RelevanceTool(BaseTool):
    """
        use an llm after the upvote ranking
        use the summary of the paper and compare the user query 
    """
    name: ClassVar[str] = "paper_relevance"
    description: ClassVar[str] = "check relevance of papers to a user query"
    args_schema: ClassVar[Type[BaseModel]] = RelevanceToolArgs

    _llm: LLM = PrivateAttr(default=None)

    def __init__(self, llm: LLM):
        super().__init__() 
        self._llm = llm
    
    def _run(self, user_query: str, summaries: List[str]):
        RelevancePrompt = f"""
you are a research assistant and your role is to choose research papers related to the user requested topic for each paper summary from the list 
Return ONLY a valid Python list of strings.
Allowed values: "yes", "no".
Do not add explanations.

user request:
{user_query}

paper's summary
{[summary for summary in summaries]}
"""
        result = self._llm.invoke(RelevancePrompt)

        return result
    
    async def _arun(self, user_query: str, summaries: list[str]):
        return self._run(user_query, summaries)
        
