from typing import ClassVar, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pathlib import Path
import requests


class FetchToolArgs(BaseModel):
    paper_identifier: str = Field(description="Paper identifier or arXiv URL")


class FetchTool(BaseTool):
    name: ClassVar[str] = "paper_fetcher"
    description: ClassVar[str] = "Fetch a paper PDF (arXiv supported)"
    args_schema: ClassVar[Type[BaseModel]] = FetchToolArgs

    save_dir: Path = Path("papers")

    def _run(self, paper_identifier: str):
        self.save_dir.mkdir(exist_ok=True)

        paper_id = self._extract_arxiv_id(paper_identifier)
        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        file_path = self.save_dir / f"{paper_id}.pdf"

        response = requests.get(
            pdf_url,
            timeout=30,
            allow_redirects=True,
        )
        response.raise_for_status()

        if not response.content.startswith(b"%PDF"):
            raise ValueError(
                f"Downloaded file is not a PDF."
            )

        with open(file_path, "wb") as f:
            f.write(response.content)

        return {
            "status": "success",
            "file_path": str(file_path.absolute()),
        }

    async def _arun(self, paper_identifier: str):
        raise NotImplementedError("Use sync version")

    def _extract_arxiv_id(self, identifier: str) -> str:
        """
        Accepts:
        - arxiv:****.*****
        - ****.*****
        - https://arxiv.org/abs/****.*****
        - https://arxiv.org/pdf/****.*****.pdf
        """
        identifier = identifier.strip()

        if identifier.startswith("http"):
            return identifier.rstrip("/").split("/")[-1].replace(".pdf", "")

        if identifier.startswith("arxiv:"):
            return identifier.replace("arxiv:", "")

        if identifier.replace(".", "").isdigit():
            return identifier

        raise ValueError(f"Unsupported paper identifier: {identifier}")
