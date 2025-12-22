from typing import ClassVar
from langchain.tools import BaseTool

class SearchTool(BaseTool):
    name: ClassVar[str]  = "search_tool"
    description: ClassVar[str]  = "A search tool"

    def _run(self, **kwargs):
        raise NotImplementedError
