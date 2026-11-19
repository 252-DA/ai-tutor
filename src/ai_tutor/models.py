from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["vector", "tools", "graph"]


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class TutorRequest(BaseModel):
    course_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=8000)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=6, ge=1, le=10)
    # vector: một lần tìm vector; tools: mô hình tự gọi công cụ; graph: tools + quan hệ giữa các ý.
    mode: RetrievalMode = "vector"
    # Ngữ cảnh người học đang mở (trình xem tài liệu). Chế độ vector chỉ dùng document_id để lọc;
    # chế độ tools/graph được báo trang đang mở và tự quyết có tra theo trang hay không.
    document_id: str | None = Field(default=None, max_length=100)
    page_number: int | None = Field(default=None, ge=1)
    lo_code: str | None = Field(default=None, max_length=100)


class TutorCitation(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = None
    quote: str | None = None
    # chunk: đoạn tài liệu; outline: sự kiện lấy từ đề cương môn học (không có tài liệu để mở).
    kind: Literal["chunk", "outline"] = "chunk"


class ToolCallTrace(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)
    ok: bool = True
    found: int | None = None
    ms: int = 0
    error: str | None = None


class TutorTrace(BaseModel):
    mode: RetrievalMode
    steps: int = 0
    llm_calls: int = 0
    retrieved: int = 0
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    forced_answer: bool = False


class TutorResponse(BaseModel):
    answer: str
    model: str
    citations: list[TutorCitation] = Field(default_factory=list)
    trace: TutorTrace
