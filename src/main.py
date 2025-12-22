import os
from dotenv import load_dotenv
from llm.llm import LLM
from tools.tools import rank_papers, summarize_papers
from utils.logger import get_logger
from agent.research_agent import ResearchAgent
from tools.search_tools.hugging_face_search import HuggingFaceSearch


def main():
    load_dotenv()

    llm = LLM(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=os.getenv("LLM_TEMPERATURE")
    )

    paper_search_tool = HuggingFaceSearch()

    tools = {
        "paper_search": paper_search_tool,
        "paper_ranker": rank_papers,
        "paper_summarizer": summarize_papers,
    }

    agent = ResearchAgent(
        llm=llm,
        tools=tools
    )

    request = "can you please check for me the latest research papers about LLM inference optimization"
    result = agent.run(
        request
    )

    print(result.content)


if __name__ == "__main__":
    logger = get_logger(__name__)
    main()
