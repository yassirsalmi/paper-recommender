from typing import List, Dict, Any, ClassVar
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class FetchToolArgs(BaseModel):
    paper_identifier: str = Field(description="Paper Identifier")


class FetchTool(BaseTool):
    name: ClassVar[str] = "paper_fetcher"
    description: ClassVar[str] = "Fetch a paper"
    args_schema: ClassVar[str] = FetchToolArgs

    def _run():
        pass
    
    