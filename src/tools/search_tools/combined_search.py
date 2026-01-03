from typing import ClassVar, Type
from pydantic import BaseModel, Field, PrivateAttr
from .search_tool import SearchTool

class CombinedSearchArgs(BaseModel):
    search_query: str = Field(description="Search query from user")
    limit: int = Field(default=10, description="Max papers per source")


class CombinedPaperSearch(SearchTool):
    name: ClassVar[str] = "fetch_articles_from_all_sources"
    description: ClassVar[str] = "Fetches papers from both Arxiv and HuggingFace"
    args_schema: ClassVar[Type[BaseModel]] = CombinedSearchArgs

    _arxiv_tool: any = PrivateAttr()
    _hf_tool: any = PrivateAttr()

    def __init__(self, arxiv_tool, hf_tool, **kwargs):
        super().__init__(**kwargs)
        self._arxiv_tool = arxiv_tool
        self._hf_tool = hf_tool

    def _run(self, search_query: str, limit: int = 10):
        arxiv_result = self._arxiv_tool.run({
            "search_query": search_query,
            "limit": limit
        })

        hf_result = self._hf_tool.run({
            "limit": limit
        })

        return {
            "search_papers": arxiv_result["papers"] + hf_result["papers"]
        }

