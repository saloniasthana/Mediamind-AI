from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: int
    filename: str
    content_type: str
    summary: str


class ChatRequest(BaseModel):
    question: str
    document_id: int | None = None


class Citation(BaseModel):
    document_id: int
    filename: str
    text: str
    start_time: float | None = None
    end_time: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class TopicRequest(BaseModel):
    topic: str
    document_id: int | None = None


class TimestampResult(BaseModel):
    document_id: int
    filename: str
    topic: str
    start_time: float | None = None
    end_time: float | None = None
    snippet: str
