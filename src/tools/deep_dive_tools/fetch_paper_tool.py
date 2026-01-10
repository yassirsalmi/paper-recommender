from typing import List, Dict, Any, ClassVar, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pathlib import Path
import requests

class FetchToolArgs(BaseModel):
    paper_identifier: str = Field(description="Paper Identifier")


class FetchTool(BaseTool):
    name: ClassVar[str] = "paper_fetcher"
    description: ClassVar[str] = "Fetch a paper"
    args_schema: ClassVar[Type[BaseModel]] = FetchToolArgs

    save_dir: Path = Path("papers")


    def _run(self, paper_identifier) -> str:
        self.save_dir.mkdir(exist_ok=True) 

        pdf_url = self._resolve_to_pdf(paper_identifier)
        file_name = pdf_url.split("/")[-1]
        file_path = self.save_dir / file_name

        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(response.content)

        return {
            "status": "success",
            "file_path": str(file_path.absolute()),
        }


    async def _arun(self, paper_identifier: str) -> str:
        raise NotImplementedError("Use sync version")


    def _resolve_to_pdf(self, identifier: str) -> str:
        if identifier.startswith("http"):
            return identifier
        
        # arxiv identifier 
        # arXiv:YYMM.number
        if identifier.startswith("arxiv:") or identifier.replace(".", "").isdigit():
            arxiv_id = identifier.replace("arxiv:", "")
            paper_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            return paper_url
        
        # TODO: fix later
        else: 
            return identifier

    