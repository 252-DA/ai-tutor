"""Prompt của AI tutor: trả lời câu hỏi người học chỉ từ ngữ cảnh khoá học."""
from ai_runtime.mcp import QuizContext

from ai_tutor.models import TutorRequest

SYSTEM_PROMPT = """You are a careful AI tutor.
Answer using only the supplied course context.
Guide the learner with a concise explanation and an example when useful.
Do not invent facts absent from the context.
Reference supporting context with markers such as [1] and [2].
If the context is insufficient, say so and ask a focused follow-up question."""


def build_prompt(request: TutorRequest, context: QuizContext) -> str:
    history = "\n".join(
        f"{item.role}: {item.content}" for item in request.history
    ) or "(none)"
    sources = "\n\n".join(
        f"[{index}]"
        + (" (" + " · ".join(filter(None, (
            " > ".join(chunk.heading_path),
            f"tr. {chunk.page_number}" if chunk.page_number is not None else "",
        ))) + ")" if chunk.heading_path or chunk.page_number is not None else "")
        + f"\n{chunk.content}"
        for index, chunk in enumerate(context.chunks, start=1)
    ) or "(no relevant course context found)"
    return (
        f"Course: {request.course_id}\n"
        f"Learning outcome: {context.lo_code} — {context.lo_statement}\n\n"
        f"Recent conversation:\n{history}\n\n"
        f"Course context:\n{sources}\n\n"
        f"Learner question: {request.message}"
    )
