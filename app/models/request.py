from pydantic import BaseModel
from typing import List

class ClassificationInput(BaseModel):
    topic: str
    labels: List[str]
    text: str
