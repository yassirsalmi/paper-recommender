import os
import json
from dotenv import load_dotenv
from llm.llm import LLM
from tools.tools import summarize_papers
from utils.logger import get_logger
from agent.research_agent import ResearchAgent
from agent.deep_dive_agent import DeepDiveAgent
from tools.search_tools.hugging_face_search import HuggingFaceSearch
from tools.search_tools.arxiv_search import ArxivSearch
from tools.search_tools.semantic_scholar_search import SemanticScholarSearch
from tools.search_tools.combined_search import CombinedPaperSearch
from tools.rank_tools.rank_tool import RankTool
from tools.rank_tools.relevance_tool import RelevanceTool
from tools.deep_dive_tools.fetch_paper_tool import FetchTool
from tools.deep_dive_tools.read_paper_tool import ReadPaperTool


def main():
    load_dotenv()

    llm = LLM(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=os.getenv("LLM_TEMPERATURE")
    )

    paper_search_tool = CombinedPaperSearch(
        arxiv_tool=ArxivSearch(),
        hf_tool=HuggingFaceSearch(),
        semantic_scholar_tool=SemanticScholarSearch()
    )

    paper_relevance_tool = RelevanceTool(llm=llm)
    paper_rank_tool = RankTool()

    search_tools = {
        "paper_search": paper_search_tool,
        "paper_relevance": paper_relevance_tool,
        "paper_ranker": paper_rank_tool,
        "paper_summarizer": summarize_papers,
    }

    paper_fetcher = FetchTool()
    section_extractor = ReadPaperTool()

    deep_dive_tools = {
        "paper_fetcher": paper_fetcher,
        "section_extractor": section_extractor
    }

    while True:
        print("\nSelect an option:")
        print("1 - Paper Recommendation")
        print("2 - Paper Deep Dive")
        print("0 - Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            agent = ResearchAgent(
                llm=llm,
                tools=search_tools
            )
            user_request = input("Enter your research request for paper recommendations: ")
            result = agent.run(user_request)
            print(json.dumps(result.content, indent=2))

        elif choice == "2":
            agent = DeepDiveAgent(
                llm=llm,
                tools=deep_dive_tools
            )
            paper_identifier = input("Please provide the link or identifier of the paper: ")
            result = agent.run(paper_identifier=paper_identifier)
            print(json.dumps(result, indent=2))

        elif choice == "0":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 0.")


if __name__ == "__main__":
    logger = get_logger(__name__)
    main()
