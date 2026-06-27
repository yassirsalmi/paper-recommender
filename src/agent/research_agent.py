from typing import Dict, List, Any
import json
from utils.helpers import extract_json_from_markdown, resolve_args, validate_required_args
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
        the 3rd attempt continue the execution even when the judge LLM give a low score
        """
        max_attempts = 3
        plan = None
        judge_result = -1
        
        for attempt in range(1, max_attempts + 1):  
            try:
                plan = self.plan(user_query=user_query, tools=self.tools)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Attempt {attempt} - plan generation failed: {e}")
                if attempt == max_attempts:
                    raise
                continue

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

        from utils.tool_schema import render_tool_schemas
        tools_schema = render_tool_schemas(tools)
        planning_prompt = func_planning_prompt(user_query=user_query, tools_schema=tools_schema)

        # print(f"-------------------------")
        # print(planning_prompt)
        # print(f"-------------------------")

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
        state = {"user_query": user_query}

        steps = plan.get("steps")
        if steps is None:
            steps = [plan]

        for step in steps:
            action_name = step.get("action", "unknown")
            tool_name   = step.get("tool", step.get("action", "unknown"))

            print(f"\n--- --- --- --- ---")
            print(f"\n--- Step: {action_name} | Tool: {tool_name} ---")
            print(f"\n--- --- --- --- ---")
            print(f"State keys before: {list(state.keys())}")

            if tool_name not in self.tools:
                raise KeyError(f"Unknown tool: {tool_name}")

            tool = self.tools[tool_name]

            raw_args = step.get("args", {})
            print(f"Raw args: {raw_args}")

            resolved_args = resolve_args(raw_args, state)
            print(f"Resolved args keys: {list(resolved_args.keys())}")

            validate_required_args(tool, resolved_args)
            tool_output = tool.invoke(resolved_args)

            print(f"Tool output type: {type(tool_output)}")
            print(f"Tool output keys: {list(tool_output.keys()) if isinstance(tool_output, dict) else 'NOT A DICT'}")

            if not isinstance(tool_output, dict):
                raise TypeError(f"Tool {tool_name} must return a dict, got {type(tool_output)}")

            state.update(tool_output)
            print(f"State keys after: {list(state.keys())}")

        return state


    def synthesize(self, user_query: str, intermediate_results: Dict[str, Any]):
        """
        combine the user_query with the tool results into the final response
        """

        synthesis_prompt = func_synthesis_prompt(user_query=user_query,
                                                 intermediate_results=intermediate_results)
        

        final_response = self.llm.invoke(synthesis_prompt)

        return final_response