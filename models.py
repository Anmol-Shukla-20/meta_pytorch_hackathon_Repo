from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    email_text: str
    sender: str
    urgency: float
    step: int

class Action(BaseModel):
    action: str  # must match action space

class Reward(BaseModel):
    score: float
    reason: Optional[str] = None