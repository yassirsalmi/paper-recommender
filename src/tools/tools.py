from datetime import date, timedelta
from itertools import islice
from langchain.tools import tool
from huggingface_hub import list_daily_papers
import arxiv

# other sources for tools search semanticscholar, PubMed
# TODO: tools to implement paper search, papers ranker, papers summarization tool

LIMIT: int = 10
DURATION: int = 7

@tool(
        "fetch_articles_from_huggingface_hub", 
        description="Fetch articles from Hugging Face")
def fetch_articles_from_hf(limit: int = LIMIT):
    """
    Fetch daily papers from Hugging Face Hub.
    
    - limit: number of papers to fetch
    """
    try:
        if limit <= 0:
            raise ValueError("limit must be > 0")

        target_date = (date.today() - timedelta(days=7)).isoformat()
        papers = list_daily_papers(date=target_date)
        
        results = []
        for paper in islice(papers, limit):
            results.append({
                "title": paper.title,
                "authors": paper.authors,
                "summary": paper.summary,
            })
        
        return results

    except Exception as e:
        raise RuntimeError(f"Failed to fetch HF daily papers: {str(e)}") from e
    
@tool(
        "fetch_articles_from_arxiv", 
        description="Fetch articles from arxiv")
def fetch_articles_from_arxiv(limit: int = LIMIT):
    """
    Fetch papers from arxiv 

    - limit: number of papers to fetch
    """
    try:
        if limit < 0:
            raise ValueError("limit must be > 0")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query="",
            max_results = limit,
            sort_by = arxiv.SortCrterion.SubmittedDate,
        )
        results = client.results(search)

        papers = []
        for result in results:
            papers.append({
                "title": result.title,
                "authors": result.authors,
                "summary": result.summary,
            })
        return papers

    except Exception as e:
        raise RuntimeError(f"Failed to fetch papers from arxiv: {str(e)}") from e
    
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
