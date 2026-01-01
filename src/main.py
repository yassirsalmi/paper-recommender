import os
from dotenv import load_dotenv
from llm.llm import LLM
from tools.tools import  summarize_papers
from utils.logger import get_logger
from agent.research_agent import ResearchAgent
from tools.search_tools.hugging_face_search import HuggingFaceSearch
from tools.rank_tools.rank_tool import RankTool
from tools.rank_tools.relevance_tool import RelevanceTool


def main():
    load_dotenv()

    llm = LLM(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=os.getenv("LLM_TEMPERATURE")
    )

    paper_search_tool = HuggingFaceSearch()
    paper_relevance_tool = RelevanceTool(llm=llm)
    paper_rank_tool = RankTool()

    tools = {
        "paper_search": paper_search_tool,
        "paper_relevance": paper_relevance_tool,
        "paper_ranker": paper_rank_tool,
        "paper_summarizer": summarize_papers,
    }

    agent = ResearchAgent(
        llm=llm,
        tools=tools
    )

    user_request = input("your request: ")
    result = agent.run(
        user_request
    )

    print(result.content)


if __name__ == "__main__":
    logger = get_logger(__name__)
    main()
