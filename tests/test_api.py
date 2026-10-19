from fastapi.testclient import TestClient

from ai_tutor.api import app
from ai_tutor.container import get_tutor_service
from ai_tutor.models import TutorResponse, TutorTrace


class _Tutor:
    def __init__(self) -> None:
        self.request = None

    def answer(self, request):
        self.request = request
        return TutorResponse(
            answer=f"Answer: {request.message}", model="fake", trace=TutorTrace(mode=request.mode)
        )

    def stream(self, request):
        self.request = request
        yield {"event": "step", "phase": "searching"}
        resp = TutorResponse(
            answer=f"Answer: {request.message}", model="fake", trace=TutorTrace(mode=request.mode)
        )
        yield {"event": "done", "response": resp.model_dump()}


def _post(tutor, body):
    app.dependency_overrides[get_tutor_service] = lambda: tutor
    try:
        return TestClient(app).post("/v1/tutor/messages", json=body)
    finally:
        app.dependency_overrides.clear()


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_modes_are_listed():
    modes = TestClient(app).get("/v1/tutor/modes").json()
    assert [m["mode"] for m in modes] == ["vector", "tools", "graph"]


def test_answer_endpoint_defaults_to_vector_and_needs_no_lo():
    tutor = _Tutor()
    response = _post(tutor, {"course_id": "CS101", "message": "Explain trees"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Answer: Explain trees"
    assert response.json()["trace"]["mode"] == "vector"
    assert tutor.request.lo_code is None


def test_mode_and_open_page_reach_the_service():
    tutor = _Tutor()
    _post(tutor, {"course_id": "CS101", "message": "q", "mode": "graph", "document_id": "d1", "page_number": 7})
    assert (tutor.request.mode, tutor.request.document_id, tutor.request.page_number) == ("graph", "d1", 7)


def test_unknown_mode_is_rejected():
    assert _post(_Tutor(), {"course_id": "CS101", "message": "q", "mode": "agent"}).status_code == 422


def test_stream_endpoint_emits_sse_frames():
    tutor = _Tutor()
    app.dependency_overrides[get_tutor_service] = lambda: tutor
    try:
        response = TestClient(app).post("/v1/tutor/messages/stream", json={"course_id": "CS101", "message": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: step" in response.text
    assert "event: done" in response.text

