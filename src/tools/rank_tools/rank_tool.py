from typing import ClassVar, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from typing import List, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class RankToolArgs(BaseModel):
    papers: List[Dict[str, Any]] = Field(
        description="List of research papers to rank"
    )
    threshold: int = Field(
        default=75,
        description="Minimum upvote threshold"
    )

class RankTool(BaseTool):
    name: ClassVar[str] = "rank_tool"
    description: ClassVar[str] = "A tool used when we want to rank the retrieved research papers by the number of upvotes"
    args_schema: ClassVar[Type[BaseModel]] = RankToolArgs

    def _run(self, papers, threshold):
        return [paper for paper in papers if paper.get("upvote", "") > threshold]