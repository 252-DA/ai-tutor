from ai_tutor.models import TutorCitation, TutorRequest, TutorResponse
from ai_tutor.ports import LearningContext, TutorModel

_SYSTEM_PROMPT = """You are a careful AI tutor.
Answer using only the supplied course context.
Guide the learner with a concise explanation and an example when useful.
Do not invent facts absent from the context.
Reference supporting context with markers such as [1] and [2].
If the context is insufficient, say so and ask a focused follow-up question."""


class TutorService:
    def __init__(self, model: TutorModel, context: LearningContext) -> None:
        self._model = model
        self._context = context

    def answer(self, request: TutorRequest) -> TutorResponse:
        context = self._context.retrieve(
            course_id=request.course_id,
            lo_code=request.lo_code,
            query=request.message,
            top_k=request.top_k,
        )
        prompt = self._build_prompt(request, context)
        answer = self._model.answer(prompt=prompt, system=_SYSTEM_PROMPT).strip()
        citations = [
            TutorCitation(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                heading_path=chunk.heading_path,
                page_number=chunk.page_number,
            )
            for index, chunk in enumerate(context.chunks, start=1)
        ]
        return TutorResponse(
            answer=answer,
            model=self._model.model_id,
            citations=citations,
        )

    @staticmethod
    def _build_prompt(request: TutorRequest, context) -> str:
        history = "\n".join(
            f"{item.role}: {item.content}" for item in request.history
        ) or "(none)"
        sources = "\n\n".join(
            f"[{index}] {chunk.content}"
            for index, chunk in enumerate(context.chunks, start=1)
        ) or "(no relevant course context found)"
        return (
            f"Course: {request.course_id}\n"
            f"Learning outcome: {context.lo_code} — {context.lo_statement}\n\n"
            f"Recent conversation:\n{history}\n\n"
            f"Course context:\n{sources}\n\n"
            f"Learner question: {request.message}"
        )
