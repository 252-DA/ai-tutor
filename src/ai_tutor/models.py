from typing import Literal

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class TutorRequest(BaseModel):
    course_id: str = Field(min_length=1, max_length=100)
    lo_code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=8000)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=5, ge=1, le=10)


class TutorCitation(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = None


class TutorResponse(BaseModel):
    answer: str
    model: str
    citations: list[TutorCitation] = Field(default_factory=list)
