from typing import ClassVar, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class SearchTool(BaseTool):
    name: ClassVar[str]  = "search_tool"
    description: ClassVar[str]  = "A search tool"
    args_schema: ClassVar[Type[BaseModel]] = Field()

    def _run(self, **kwargs):
        raise NotImplementedError
