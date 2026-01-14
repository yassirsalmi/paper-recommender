from typing import List, Dict, Any, ClassVar, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pathlib import Path
import pdfplumber

class ReadPaperToolArgs(BaseModel):
    path_to_paper: str = Field(description="Path to Paper")


class ReadPaperTool(BaseTool):
    name: ClassVar[str] = "paper_reader"
    description: ClassVar[str] = "Read a Paper"
    args_schema: ClassVar[Type[BaseModel]] = ReadPaperToolArgs


    def _run(self, path_to_paper: str):
        path = Path(path_to_paper)

        if not path.exists():
            raise FileNotFoundError(f"Paper not found at: {path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF is the only supported format")
        
        pages_txt = []

        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages_txt.append(text)

        if not pages_txt:
            raise ValueError(f"Error Extracting text from: {path}")
        
        return {
            "paper_path": str(path),
            "number_pages": len(pages_txt),
            "content": "\n\n".join(pages_txt),
        }

