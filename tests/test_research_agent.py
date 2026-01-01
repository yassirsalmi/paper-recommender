from unittest.mock import MagicMock
from llm.llm import LLM
from agent.research_agent import ResearchAgent

def test_agent_happy_path():
    
    llm = MagicMock(spec=LLM)

    llm.invoke.side_effect = [
        MagicMock(content="""
        ```json
        {
          "steps": [
            {"tool": "search", "action": "search papers", "args": {}}
          ]
        }
        ```
        """),

        MagicMock(content="8"),

        MagicMock(content="Final synthesized answer")
    ]

    search_tool = MagicMock()
    search_tool.invoke.return_value = [
        {"title": "Paper 1", "summary": "ML paper"}
    ]

    tools = {
        "search": search_tool
    }

    agent = ResearchAgent(llm=llm, tools=tools)

    result = agent.run("machine learning")

    assert result.content == "Final synthesized answer"
    assert search_tool.invoke.called
