from pydantic import BaseModel
from typing import List


class LLMDataRequest(BaseModel):
    symbols: List[str]
    source: str
    frequency: str
    start: str
    end: str
    data_type: str