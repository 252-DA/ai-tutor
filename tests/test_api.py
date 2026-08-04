from fastapi.testclient import TestClient

from ai_tutor.api import app
from ai_tutor.container import get_tutor_service
from ai_tutor.models import TutorResponse


class _Tutor:
    def answer(self, request):
        return TutorResponse(answer=f"Answer: {request.message}", model="fake")


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_answer_endpoint():
    app.dependency_overrides[get_tutor_service] = lambda: _Tutor()
    try:
        response = TestClient(app).post(
            "/v1/tutor/messages",
            json={
                "course_id": "CS101",
                "lo_code": "LO.1",
                "message": "Explain trees",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "Answer: Explain trees"
