
from pydantic import BaseModel


class LLMDataRequest(BaseModel):
    symbols: list[str]
    source: str
    frequency: str
    start: str
    end: str
    data_type: str