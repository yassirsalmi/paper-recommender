from typing import Dict, Any, Optional
import json
import logging
from utils.prompt_template import _build_explanation_prompt


class DeepDiveAgent:
    def __init__(self, llm, tools: Dict[str, Any], memory: Optional[Any] = None):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, paper_identifier: str) -> Dict[str, Any]:
        """
        Entry point for the deep dive flow.

        Flow:
            1. Download paper
            2. Read and extract content
            3. Explain paper
            4. Return structured explanation
        """

        self.logger.info(f"Starting deep dive for: {paper_identifier}")

        paper_info = self.download(paper_identifier)
        if paper_info["status"] == "success":
            paper_path = paper_info["file_path"]
        paper = self.read_paper(paper_path)
        explanation = self.explain(paper["content"])

        return {
            "paper": paper_identifier,
            "explanation": explanation,
        }

    def download(self, paper_identifier: str) -> str:
        # NOTE: since the hf/arxiv api don't provide a way to have extract the content of the paper
        # Downloading the paper to local machine and extracting the content will be the best solution
        # submiting the full content to an llm can be challeging but will try this first
        self.logger.info("Downloading paper content")

        fetch_tool = self.tools.get("paper_fetcher")
        if not fetch_tool:
            raise ValueError("paper_fetcher is not a registered tool")

        # returns a dict with the file_path
        return fetch_tool.invoke({"paper_identifier": paper_identifier})

    def read_paper(self, path_to_paper: str) -> Dict[str, str]:
        self.logger.info("Extracting paper sections")

        section_extractor = self.tools.get("section_extractor")
        if not section_extractor:
            raise ValueError("section_extractor is not a registered tool")

        return section_extractor.invoke({"path_to_paper": path_to_paper})

    def explain(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Explain the paper using the LLM.
        """

        self.logger.info("Explaining paper content")

        prompt = _build_explanation_prompt(sections)
        response = self.llm.invoke(prompt)

        return self._parse_explanation(response)

    def _parse_explanation(self, llm_response: Any) -> Dict[str, str]:
        """
        Parse LLM response into structured output.
        """

        return {
            "summary": llm_response.content
        }
