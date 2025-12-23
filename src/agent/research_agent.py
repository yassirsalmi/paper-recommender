from typing import Dict, List, Any
import json
from utils.helpers import extract_json_from_markdown
from utils.prompt_template import func_judgemental_prompt, func_planning_prompt, func_synthesis_prompt

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

            judge_result = self.judge_generated_plan(
                user_query=user_query, 
                tools=self.tools, 
                plan=plan
            )

            if judge_result > 7 or attempt == max_attempts:
                if attempt == max_attempts and judge_result <= 7:
                    print(f"Final attempt - executing plan despite low score ({judge_result})")
                break
        
        intermediate_results = self.execute(plan=plan)

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

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        state = {}

        for step in plan["steps"]:
            tool_name = step["tool"]
            tool = self.tools[tool_name]

            if step["action"] == "search":
                state["papers"] = tool.invoke(step.get("args", {}))

            elif step["action"] == "rank":
                state["ranked_papers"] = tool.invoke({
                    "papers": state["papers"]
                })

            elif step["action"] == "summarize":
                state["summaries"] = tool.invoke({
                    "papers": state["ranked_papers"]
                })

        return state


    def synthesize(self, user_query: str, intermediate_results: Dict[str, Any]):
        """
        combine the user_query with the tool results into the final response
        """

        synthesis_prompt = func_synthesis_prompt(user_query=user_query,
                                                 intermediate_results=intermediate_results)
        

        final_response = self.llm.invoke(synthesis_prompt)

        return final_response