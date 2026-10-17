from typing import Annotated

from ai_runtime.errors import AIRuntimeError
from fastapi import Depends, FastAPI, HTTPException, status

from ai_tutor.container import get_tutor_service
from ai_tutor.models import TutorRequest, TutorResponse
from ai_tutor.service import TutorService

app = FastAPI(title="AI Tutor Service", version="0.2.0")

MODES = [
    {"mode": "vector", "description": "One vector search, then answer."},
    {"mode": "tools", "description": "The model calls tools: search, page, outline, neighbours."},
    {"mode": "graph", "description": "Tools plus concept relations verified against quotes."},
]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/tutor/modes")
def list_modes() -> list[dict[str, str]]:
    return MODES


@app.post("/v1/tutor/messages", response_model=TutorResponse)
def answer_tutor_message(
    request: TutorRequest,
    service: Annotated[TutorService, Depends(get_tutor_service)],
) -> TutorResponse:
    try:
        return service.answer(request)
    except AIRuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI tutor dependency failed",
        ) from exc
