from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from .search_tool import SearchTool
from typing import List, ClassVar, Type
import arxiv

class ArxivSearchArgs(BaseModel):
    search_query: str = Field(description="the search query fron the user input")
    limit: int = Field(default=100, description="Maximum number of papers to fetch")


class ArxivSearch(SearchTool):
    name: ClassVar[str] = "fetch_articles_from_arxiv"
    description: ClassVar[str] = "Fetches papers from arxiv"
    args_schema: ClassVar[Type[BaseModel]] = ArxivSearchArgs

    def _run(self, search_query: str, limit: int = 100):
        client = arxiv.Client()

        search = arxiv.Search(
            query=search_query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = list(client.results(search))
        
        return {
            "papers": [
                {
                    "title": result.title,
                    "summary": result.summary,
                    "authors": result.authors
                }
            ]
            for result in results
        }