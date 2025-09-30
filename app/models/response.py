from pydantic import BaseModel
from typing import List
from app.models.request import ClassificationInput

class ClassificationResponse(BaseModel):
    label: str
    is_winner: bool
    logprobs: float
    logprob_interp: str
    relative_probs: float

class ClassificationOutput(BaseModel):
    input: ClassificationInput
    outputs: List[ClassificationResponse]