import json
import logging
from collections.abc import Iterator
from typing import Annotated

from ai_runtime.errors import AIRuntimeError
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse

from ai_tutor.container import get_tutor_service
from ai_tutor.models import TutorRequest, TutorResponse
from ai_tutor.service import TutorService

logger = logging.getLogger(__name__)

app = FastAPI(title="AI Tutor Service", version="0.2.0")

MODES = [
    {"mode": "vector", "description": "One vector search, then answer."},
    {"mode": "tools", "description": "The model calls tools: search, page, outline, neighbours."},
    {"mode": "graph", "description": "Tools plus concept relations verified against quotes."},
]


def _error_event() -> str:
    payload = {"event": "error", "detail": "AI tutor dependency failed"}
    return f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


@app.post("/v1/tutor/messages/stream")
def stream_tutor_message(
    request: TutorRequest,
    service: Annotated[TutorService, Depends(get_tutor_service)],
) -> StreamingResponse:
    def generate() -> Iterator[str]:
        try:
            for item in service.stream(request):
                event = str(item.get("event", "message"))
                yield f"event: {event}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
        except AIRuntimeError:
            yield _error_event()
        except Exception:
            # Luồng không bao giờ được cắt ngang mà không có sự kiện kết thúc.
            logger.exception("AI tutor stream failed")
            yield _error_event()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
