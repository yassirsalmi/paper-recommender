from .search_tool import SearchTool
from semanticscholar import SemanticScholar
from typing import ClassVar, List, Type
from pydantic import BaseModel, Field

class SemanticScholarSearchArgs(BaseModel):
    limit: int = Field(default=10, description="Maximum number of papers to fetch")
    keywords: List[str] = Field(default= None, description="Keywords from the user query")

class SemanticScholarSearch(SearchTool):
    name: ClassVar[str] = "semantic_scholar_search"
    description: ClassVar[str] = "Fetches recent papers from SemanticScholar"
    args_schema: ClassVar[Type[BaseModel]] = SemanticScholarSearchArgs

    def _run(self, keywords: List[str], limit: int = 10):
        sch = SemanticScholar()
        results = []

        for keyword in keywords:
            results.extend(sch.search_paper(keyword)[:limit])

        return results