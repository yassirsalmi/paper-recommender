from typing import ClassVar, Type
from .search_tool import SearchTool
from datetime import date, timedelta
from itertools import islice
from huggingface_hub import list_daily_papers
from pydantic import BaseModel, Field


class HuggingFaceSearchArgs(BaseModel):
    search_query: str = Field(description="the search query fron the user input")
    limit: int = Field(default=100, description="Maximum number of papers to fetch")


class HuggingFaceSearch(SearchTool):
    name: ClassVar[str] = "fetch_articles_from_hf"
    description: ClassVar[str] = "Fetches recent papers from HuggingFace"
    args_schema: ClassVar[Type[BaseModel]] = HuggingFaceSearchArgs

    def _run(self, search_query: str, limit: int = 100):
        target_date = (date.today() - timedelta(days=7)).isoformat()
        papers = list_daily_papers(date=target_date)

        return {
            "paper_search_results": [
                {
                    "title": paper.title,
                    "authors": paper.authors,
                    "summary": paper.summary,
                    "upvote": paper.upvotes,
                }
                for paper in islice(papers, limit)
            ]
        }
