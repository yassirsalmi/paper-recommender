from typing import List, Dict, Any, ClassVar, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class ReadPaperToolArgs(BaseModel):
    path_to_paper: str = Field(description="Path to Paper")

class ReadPaperTool(BaseTool):
    name: ClassVar[str] = "paper_reader"
    description: ClassVar[str] = "Read a Paper"
    args_schema: ClassVar[Type[BaseModel]] = ReadPaperToolArgs


    def _run():
        pass