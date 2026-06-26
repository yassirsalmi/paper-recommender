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
    rank_metric: int = Field(
        default=25,
        description="Minimum rank_metric value"
    )

class RankTool(BaseTool):
    name: ClassVar[str] = "rank_tool"
    description: ClassVar[str] = "A tool used when we want to rank the retrieved research papers by the number of upvotes"
    args_schema: ClassVar[Type[BaseModel]] = RankToolArgs
    output_key: str = "ranked_papers"

    def _run(self, papers: List[Dict[str, Any]], rank_metric: int = 25):

        ranked = [
            paper for paper in papers
            if paper.get("upvote", 0) > rank_metric 
        ]

        return {
            "ranked_papers": ranked
        }