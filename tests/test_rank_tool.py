import pytest 
from tools.rank_tools.rank_tool import RankTool

def test_rank_tool_scoring_threshold():
    list_of_papers = [
        {"title": "paper title 1", "upvote": 50},
        {"title": "paper title 2", "upvote": 80},
        {"title": "paper title 3", "upvote": 99}
    ]

    expected_paper_after_ranking = {"ranked_papers": [
        {"title": "paper title 2", "upvote": 80},
        {"title": "paper title 3", "upvote": 99}
    ]}

    tool = RankTool()

    result = tool.run({
        "papers": list_of_papers,
        "rank_metric": 75
    })

    assert result == expected_paper_after_ranking

def test_rank_tool_default_threshold():
    list_of_papers = [
        {"title": "paper title 1", "upvote": 15},
        {"title": "paper title 2", "upvote": 80}
    ]

    expected = {"ranked_papers": [{"title": "paper title 2", "upvote": 80}]}

    tool = RankTool()

    result = tool.run({
        "papers": list_of_papers
    })

    assert result == expected
