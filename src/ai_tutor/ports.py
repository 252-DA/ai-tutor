from typing import Protocol

from ai_runtime.mcp import QuizContext


class TutorModel(Protocol):
    @property
    def model_id(self) -> str:
        ...

    def answer(self, prompt: str, system: str) -> str:
        ...


class LearningContext(Protocol):
    def retrieve(
        self,
        *,
        course_id: str,
        lo_code: str,
        query: str,
        top_k: int,
    ) -> QuizContext:
        ...
