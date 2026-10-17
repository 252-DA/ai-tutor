import time

from ai_tutor import prompts
from ai_tutor.agent import run_tool_loop
from ai_tutor.evidence import EvidenceBook
from ai_tutor.models import ToolCallTrace, TutorRequest, TutorResponse, TutorTrace
from ai_tutor.ports import CourseTools, TutorModel
from ai_tutor.tools import add_chunks


class TutorService:
    """
    Ba chế độ truy xuất, cùng mô hình và cùng quy tắc trả lời:

    - ``vector``: một lần ``search_course_chunks``, rồi một lần gọi mô hình.
    - ``tools``:  mô hình tự gọi công cụ (tìm vector, tra trang, đề cương, láng giềng).
    - ``graph``:  như ``tools`` cộng ``get_relations`` (quan hệ giữa các ý đã kiểm trích dẫn).
    """

    def __init__(
        self,
        model: TutorModel,
        tools: CourseTools,
        *,
        max_tool_calls_tools: int = 5,
        max_tool_calls_graph: int = 6,
    ) -> None:
        self._model = model
        self._tools = tools
        self._max_calls = {"tools": max_tool_calls_tools, "graph": max_tool_calls_graph}

    def answer(self, request: TutorRequest) -> TutorResponse:
        started = time.monotonic()
        book = EvidenceBook()
        trace = TutorTrace(mode=request.mode)

        if request.mode == "vector":
            answer = self._answer_vector(request, book, trace)
        else:
            result = run_tool_loop(
                request=request,
                model=self._model,
                tools=self._tools,
                book=book,
                trace=trace,
                max_tool_calls=self._max_calls[request.mode],
            )
            answer, trace.forced_answer = result.answer, result.forced

        trace.retrieved = len(book)
        trace.latency_ms = int((time.monotonic() - started) * 1000)
        return TutorResponse(
            answer=answer,
            model=self._model.model_id,
            citations=book.citations(answer),
            trace=trace,
        )

    def _answer_vector(self, request: TutorRequest, book: EvidenceBook, trace: TutorTrace) -> str:
        arguments = {"course_id": request.course_id, "query": request.message, "top_k": request.top_k}
        if request.document_id:
            arguments["document_ids"] = [request.document_id]
        started = time.monotonic()
        result = self._tools.call("search_course_chunks", arguments)
        items = add_chunks(book, result.get("results") or [], name_key="document_name")
        trace.tool_calls.append(
            _trace_call("search_chunks", arguments, len(items), started)
        )
        reply = self._model.generate(prompts.vector_prompt(request, book), prompts.SYSTEM_VECTOR)
        trace.steps = 1
        trace.llm_calls = 1
        trace.input_tokens = reply.input_tokens
        trace.output_tokens = reply.output_tokens
        return reply.text.strip()


def _trace_call(tool: str, args: dict, found: int, started: float):
    return ToolCallTrace(
        tool=tool,
        args={k: v for k, v in args.items() if k != "course_id"},
        found=found,
        ms=int((time.monotonic() - started) * 1000),
    )
