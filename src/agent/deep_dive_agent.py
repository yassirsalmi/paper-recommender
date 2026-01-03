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

        paper_content = self.download(paper_identifier)
        sections = self.read_paper(paper_content)
        explanation = self.explain(sections)

        return {
            "paper": paper_identifier,
            "explanation": explanation,
        }

    def download(self, paper_identifier: str) -> str:
        self.logger.info("Downloading paper content")

        fetch_tool = self.tools.get("paper_fetcher")
        if not fetch_tool:
            raise ValueError("paper_fetcher is not a registered tool")

        return fetch_tool(paper_identifier)

    def read_paper(self, paper_content: str) -> Dict[str, str]:
        self.logger.info("Extracting paper sections")

        extractor = self.tools.get("section_extractor")
        if not extractor:
            raise ValueError("section_extractor is not a registered tool")

        return extractor(paper_content)

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
