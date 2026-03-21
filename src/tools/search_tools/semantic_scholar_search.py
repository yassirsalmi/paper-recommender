from .search_tool import SearchTool
from semanticscholar import SemanticScholar
from typing import ClassVar, List, Type
from pydantic import BaseModel, Field

class SemanticScholarSearchArgs(BaseModel):
    limit: int = Field(default=100, description="Maximum number of papers to fetch")
    search_query: str = Field(default= None, description="Keyword from the user query")

class SemanticScholarSearch(SearchTool):
    name: ClassVar[str] = "semantic_scholar_search"
    description: ClassVar[str] = "Fetches recent papers from SemanticScholar"
    args_schema: ClassVar[Type[BaseModel]] = SemanticScholarSearchArgs

    def _run(self, search_query: str, limit: int = 100):
        sch = SemanticScholar()
        papers = []

        papers.extend(sch.search_paper(search_query)[:limit])

        return {
            "paper_search_results": [
                {
                    "title": paper.title,
                    "summary": paper.abstract, # using abstract instead of summary will fix later
                    # "summary": paper.summary, #! 'Paper' object has no attribute 'summary' need to check this 

                }
                for paper in papers
            ]
            }