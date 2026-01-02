from langchain.tools import tool

# TODO: start creating a separate tool file for each category of tools
# TODO: tools to implement paper search, papers ranker, papers summarization tool

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
    return {
        "summaries": [
                f"{p['title']}: relevant because {p['summary'][:50]}..." for p in papers
            ]
    }
