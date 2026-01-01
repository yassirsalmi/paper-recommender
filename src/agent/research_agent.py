from typing import Dict, List, Any
import json
from utils.helpers import extract_json_from_markdown
from utils.prompt_template import func_judgemental_prompt, func_planning_prompt, func_synthesis_prompt
from agent.plan import Plan

class ResearchAgent():
    """
    Research Agent High Level Presentation:
    - understand the users research interest
    - decide the tool to use
    - synthesize a final response
    """
    def __init__(self, llm, tools: Dict[str, Any], memory = None):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def run(self, user_query: str) -> str:
        """
        Generate plan with up to 3 retries
        the 3rd attempt continue the execution even when the judge LLM give a weak score
        """
        max_attempts = 3
        plan = None
        judge_result = -1
        
        for attempt in range(1, max_attempts + 1):  
            plan = self.plan(user_query=user_query, tools=self.tools)

            # validation before judge
            # plan = Plan.model_validate_json(str(plan))

            judge_result = self.judge_generated_plan(
                user_query=user_query, 
                tools=self.tools, 
                plan=plan
            )

            if judge_result > 7 or attempt == max_attempts:
                if attempt == max_attempts and judge_result <= 7:
                    print(f"Final attempt - executing plan despite low score ({judge_result})")
                break
        
        intermediate_results = self.execute(plan=plan, user_query= user_query)

        response = self.synthesize(
            user_query=user_query, 
            intermediate_results=intermediate_results
        )

        if self.memory:
            self.memory.add_interaction(user_query, response)

        return response
    
    def plan(self, user_query: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        tools_list = ", ".join(tools.keys())

        planning_prompt = func_planning_prompt(user_query=user_query,
                                               tools_list=tools_list)
        
        ai_message = self.llm.invoke(planning_prompt)
        plan_text = ai_message.content.strip()

        if not plan_text:
            raise ValueError("Planner returned empty output")

        # extract json
        json_output = extract_json_from_markdown(plan_text)
        print(json_output)
        return json.loads(json_output)
    
    def judge_generated_plan(self, user_query: str, tools: Dict[str, Any], plan: Dict[str, Any]) -> float:
        available_tools = ", ".join(tools.keys())

        judgemental_prompt = func_judgemental_prompt(user_query=user_query,
                           available_tools=available_tools,
                           plan=plan)

        judge_message = self.llm.invoke(judgemental_prompt)
        score_text = judge_message.content.strip()

        try:
            score = float(score_text)
        except ValueError:
            return -1

        return score

    def execute(self, plan: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        # TODO: tool calls are hardcoded in here need to find a better way to do it
        #! need to clean up this and do this in a better way since a lot of errors raises
        state = {}

        for step in plan["steps"]:
            tool_name = step["tool"]
            action = step["action"]

            if tool_name not in self.tools:
                raise KeyError(f"Unknown tool: {tool_name}")

            tool = self.tools[tool_name]

            if "search" in action:
                state["papers"] = tool.invoke(step.get("args", {}))
            
            elif "rank" in action:
                papers_to_rank = state.get("ranked_relevant_papers", state.get("papers"))
                if papers_to_rank is None:
                    raise RuntimeError("Cannot rank without papers")
                state["ranked_papers"] = tool.invoke({"papers": papers_to_rank, "threshold": 75})

            elif tool_name == "paper_relevance":
                if "papers" not in state:
                    raise RuntimeError("Cannot filter relevance without papers")

                summaries = [paper["summary"] for paper in state["papers"]]

                relevance_decisions = tool.invoke({
                    "user_query": user_query,
                    "summaries": summaries
                })

                state["ranked_relevant_papers"] = [
                    paper for paper, decision in zip(state["papers"], relevance_decisions)
                    if str(decision[0] if isinstance(decision, tuple) else decision).lower() == "yes"
                ]

            elif "summarize" in action:
                if "ranked_papers" not in state:
                    raise RuntimeError("Cannot summarize without ranked papers")
                state["summaries"] = tool.invoke({"papers": state["ranked_papers"]})

            else:
                raise ValueError(f"Unknown action: {action}")

        return state


    def synthesize(self, user_query: str, intermediate_results: Dict[str, Any]):
        """
        combine the user_query with the tool results into the final response
        """

        synthesis_prompt = func_synthesis_prompt(user_query=user_query,
                                                 intermediate_results=intermediate_results)
        

        final_response = self.llm.invoke(synthesis_prompt)

        return final_response