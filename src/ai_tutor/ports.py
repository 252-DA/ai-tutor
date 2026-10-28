from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelReply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class TutorModel(Protocol):
    @property
    def model_id(self) -> str:
        ...

    def generate(self, prompt: str, system: str) -> ModelReply:
        ...


class StreamingTutorModel(TutorModel, Protocol):
    def stream(self, prompt: str, system: str) -> Iterator[str | ModelReply]:
        """Các mẩu chữ mới theo thứ tự, phần tử cuối là ModelReply (toàn văn và số token)."""
        ...


def stream_reply(model: TutorModel, prompt: str, system: str) -> Iterator[str | ModelReply]:
    """Sinh theo luồng nếu mô hình hỗ trợ; nếu không thì sinh một lần, trả toàn văn thành một mẩu."""
    stream = getattr(model, "stream", None)
    if stream is None:
        reply = model.generate(prompt, system)
        yield reply.text
        yield reply
        return
    yield from stream(prompt, system)


class CourseTools(Protocol):
    """Công cụ tra cứu khóa học (MCP). Tên là tên công cụ MCP, tham số đã được kiểm."""

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...
