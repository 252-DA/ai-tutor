from ai_runtime.mcp import LearningContextClient, QuizContext


class MCPLearningContext:
    def __init__(self, client: LearningContextClient) -> None:
        self._client = client

    def retrieve(
        self,
        *,
        course_id: str,
        lo_code: str,
        query: str,
        top_k: int,
    ) -> QuizContext:
        return self._client.retrieve_quiz_context(
            course_id=course_id,
            lo_code=lo_code,
            query=query,
            assessment_style="tutor",
            top_k=top_k,
        )
