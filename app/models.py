from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: int | None = None
    filename: str
    content_type: str
    file_path: str
    extracted_text: str
    summary: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(BaseModel):
    id: int | None = None
    document_id: int
    text: str
    start_time: float | None = None
    end_time: float | None = None
    token_set: str
