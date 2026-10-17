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


class CourseTools(Protocol):
    """Công cụ tra cứu khóa học (MCP). Tên là tên công cụ MCP, tham số đã được kiểm."""

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...
