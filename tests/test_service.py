from ai_runtime.mcp import QuizContext, QuizContextChunk

from ai_tutor.models import ConversationMessage, TutorRequest
from ai_tutor.service import TutorService


class _Model:
    model_id = "test-model"

    def __init__(self) -> None:
        self.prompt = ""
        self.system = ""

    def answer(self, prompt: str, system: str) -> str:
        self.prompt = prompt
        self.system = system
        return "A grounded answer [1]."


class _Context:
    def retrieve(self, **kwargs) -> QuizContext:
        self.arguments = kwargs
        return QuizContext(
            course_id=kwargs["course_id"],
            lo_id="lo-1",
            lo_code=kwargs["lo_code"],
            lo_statement="Explain trees",
            bloom_level="understand",
            assessment_style="tutor",
            query=kwargs["query"],
            chunks=[
                QuizContextChunk(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content="A tree is a connected acyclic graph.",
                    heading_path=["Graphs", "Trees"],
                    page_number=12,
                    rank=1,
                    score=0.9,
                )
            ],
            total_chars=36,
        )


def test_tutor_grounds_answer_and_returns_citations():
    model = _Model()
    context = _Context()
    service = TutorService(model=model, context=context)

    response = service.answer(
        TutorRequest(
            course_id="CS101",
            lo_code="LO.1",
            message="What is a tree?",
            history=[ConversationMessage(role="user", content="Teach me graphs")],
        )
    )

    assert response.answer == "A grounded answer [1]."
    assert response.model == "test-model"
    assert response.citations[0].chunk_id == "chunk-1"
    assert "A tree is a connected acyclic graph." in model.prompt
    assert "[1] (Graphs > Trees · tr. 12)\n" in model.prompt
    assert "Teach me graphs" in model.prompt
    assert context.arguments["query"] == "What is a tree?"
