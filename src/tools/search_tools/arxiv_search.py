import arxiv
from .search_tool import SearchTool
from typing import List, ClassVar, Type
from pydantic import BaseModel, Field

class ArxivSearchArgs(BaseModel):
    search_query: str = Field(description="the search query from the user input")
    limit: int = Field(default=100, description="Maximum number of papers to fetch")


class ArxivSearch(SearchTool):
    name: ClassVar[str] = "fetch_articles_from_arxiv"
    description: ClassVar[str] = """
            This Tool is used whenever we want to fetch paper from arxiv plateform
        """
    args_schema: ClassVar[Type[BaseModel]] = ArxivSearchArgs

    def _run(self, search_query: str, limit: int = 100):
        try: 
            client = arxiv.Client()

            search = arxiv.Search(
                query=search_query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            results = list(client.results(search))
            
            return {
                "paper_search_results": [
                    {
                        "title": result.title,
                        "summary": result.summary,
                        "authors": result.authors
                    }
                    for result in results
                ]
            }
        except Exception as e:
            return {"error": {str(e)}}