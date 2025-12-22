from datetime import date, timedelta
from itertools import islice
from typing import List
from langchain.tools import tool
from huggingface_hub import list_daily_papers
from semanticscholar import SemanticScholar
import arxiv

# TODO: start creating a separate tool file for each category of tools
# TODO: tools to implement paper search, papers ranker, papers summarization tool
    
@tool(
        "paper_ranker",
        description="rank the fetched papers")
def rank_papers(papers):
    """
    Rank a list of research papers by relevance and quality.
    
    Input:
    - papers: list of paper objects with title, authors, summary

    Output:
    - ranked list of papers (most relevant first)
    """
    # mock for the moment
    return papers

@tool(
        "summarize_papers",
        description="provide a brief summary from the retrived summary of the paper from the API"
)
def summarize_papers(papers):
    """
    Summarize a list of research papers and explain their relevance.

    Input:
    - papers: list of ranked paper objects

    Output:
    - list of short human-readable summaries
    """
    return [
        f"{p['title']}: relevant because {p['summary'][:50]}..."
        for p in papers
    ]
