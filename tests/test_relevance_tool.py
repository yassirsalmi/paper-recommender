import pytest
from unittest.mock import MagicMock
from llm.llm import LLM
from tools.rank_tools.relevance_tool import RelevanceTool

def test_relevance_tool_returns_llm_output():
    llm = MagicMock(spec=LLM)
    llm.invoke.return_value = MagicMock(content="['yes', 'no']")

    tool = RelevanceTool(llm=llm)

    result = tool.run({
        "user_query": "machine learning",
        "papers": [
            {
                "title": "Paper about GNNs", 
                "summary": "Paper about GNNs"
            },
            {
                "title": "Paper about medical things", 
                "summary": "Paper about medical things"
            }
        ]
    })

    expected = {"relevant_papers": [{"title": "Paper about GNNs", "summary": "Paper about GNNs"}]}

    assert result == expected
    llm.invoke.assert_called_once()

def test_relevance_tool_prompt_contains_inputs():
    llm = MagicMock(spec=LLM)
    llm.invoke.return_value = MagicMock(content="['yes']")

    tool = RelevanceTool(llm=llm)

    tool.run({
        "user_query": "graph neural networks",
        "papers": [
            {
                "title": "Paper about GNNs", 
                "summary": "Study of GNN architectures"
            },
        ]
    })

    prompt = llm.invoke.call_args[0][0]

    assert "graph neural networks" in prompt
    assert "Study of GNN architectures" in prompt
    assert "Return ONLY a valid Python list of strings" in prompt


