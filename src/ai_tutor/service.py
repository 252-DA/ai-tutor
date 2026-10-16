from ai_tutor.models import TutorCitation, TutorRequest, TutorResponse
from ai_tutor.ports import LearningContext, TutorModel
from ai_tutor.prompts import SYSTEM_PROMPT, build_prompt


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
        answer = self._model.answer(
            prompt=build_prompt(request, context), system=SYSTEM_PROMPT
        ).strip()
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
