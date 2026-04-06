from datetime import datetime
from pydantic import BaseModel


# --- Request ---
class ChatRequest(BaseModel):
    query: str


# --- Response ---
class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    messages: list[MessageOut] = []
