from pydantic import BaseModel
from typing import List, Dict, Any

class PlanStep(BaseModel):
    tool: str
    action: str
    args: Dict[str, Any] = {}

class Plan(BaseModel):
    steps: List[PlanStep]
