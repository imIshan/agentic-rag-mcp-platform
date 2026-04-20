from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]

class QueryPlan(BaseModel):
    question_type: str
    should_retrieve: bool
    reason: str