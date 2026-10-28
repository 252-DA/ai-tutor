"""Ba chế độ truy xuất với mô hình và công cụ giả: không mạng, không LLM thật."""
import json
from typing import Any

import pytest
from ai_runtime.errors import MCPToolError

from ai_tutor.agent import parse_action
from ai_tutor.models import ConversationMessage, TutorRequest
from ai_tutor.ports import ModelReply
from ai_tutor.service import TutorService

COURSE = "co3137-uuid"


class FakeModel:
    """Trả lần lượt các câu đã định; ghi lại prompt và system của từng lần gọi."""

    model_id = "fake-model"

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def generate(self, prompt: str, system: str) -> ModelReply:
        self.prompts.append(prompt)
        self.systems.append(system)
        return ModelReply(text=self._replies.pop(0), input_tokens=100, output_tokens=20)


class FakeTools:
    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        result = self.results.get(name)
        if isinstance(result, Exception):
            raise result
        return result if result is not None else {}


def chunk(i: int, text: str, page: int = 5, heading=("Chương 5", "Deadlock")) -> dict:
    return {
        "chunk_id": f"c{i}", "document_id": "d1", "document_name": "Ch5.pdf",
        "heading_path": list(heading), "content": text, "page_number": page, "score": 0.9, "rank": i,
    }


def tool(name: str, **args) -> str:
    return json.dumps({"action": "tool", "tool": name, "args": args}, ensure_ascii=False)


def answer(text: str) -> str:
    return json.dumps({"action": "answer", "answer": text}, ensure_ascii=False)


def ask(service, message="Deadlock là gì?", **kw):
    return service.answer(TutorRequest(course_id=COURSE, message=message, **kw))


SEARCH = {"search_course_chunks": {"results": [chunk(1, "Deadlock là trạng thái các tiến trình chờ nhau."), chunk(2, "Bốn điều kiện cần.", page=6)]}}


# ----------------------------------------------------------------- vector

def test_vector_mode_searches_once_and_answers_once():
    model, tools = FakeModel("Deadlock là chờ vòng tròn [1]. Cần bốn điều kiện [2]."), FakeTools(SEARCH)
    response = ask(TutorService(model, tools))

    assert tools.calls == [("search_course_chunks", {"course_id": COURSE, "query": "Deadlock là gì?", "top_k": 6})]
    assert response.trace.mode == "vector"
    assert (response.trace.steps, response.trace.llm_calls, response.trace.retrieved) == (1, 1, 2)
    assert (response.trace.input_tokens, response.trace.output_tokens) == (100, 20)
    assert [c.index for c in response.citations] == [1, 2]
    assert response.citations[1].page_number == 6
    assert "Ch5.pdf · Chương 5 > Deadlock · tr. 5" in model.prompts[0]


def test_vector_mode_filters_by_open_document():
    model = FakeModel("ok")
    tools = FakeTools(SEARCH)
    ask(TutorService(model, tools), document_id="d1", page_number=5, lo_code="L.O.1.2")
    assert tools.calls[0][1]["document_ids"] == ["d1"]
    assert "document d1 open, page 5" in model.prompts[0]
    assert "Current learning outcome focus: L.O.1.2." in model.prompts[0]


def test_vector_mode_citation_markers_not_in_the_book_are_ignored():
    response = ask(TutorService(FakeModel("Theo [1] và [9]."), FakeTools(SEARCH)))
    assert [c.index for c in response.citations] == [1]


def test_citation_parser_ignores_array_indexing_and_code_blocks():
    reply = (
        "Ta truy cập phần tử bằng `arr[1]` hoặc viết a[2] trong câu, và khối:\n"
        "```python\nx = [1, 2]\n```\n"
        "Nhưng khẳng định thực tế thì trích [1]."
    )
    response = ask(TutorService(FakeModel(reply), FakeTools(SEARCH)))
    assert [c.index for c in response.citations] == [1]


def test_vector_mode_with_no_results_still_answers():
    response = ask(TutorService(FakeModel("Tài liệu không nói về điều này."), FakeTools({"search_course_chunks": {"results": []}})))
    assert response.citations == [] and response.trace.retrieved == 0


# ----------------------------------------------------------------- tools

def test_tool_loop_searches_then_answers_with_evidence_numbers():
    model = FakeModel(tool("search_chunks", query="deadlock"), answer("Deadlock là … [1]."))
    tools = FakeTools(SEARCH)
    response = ask(TutorService(model, tools), mode="tools")

    assert tools.calls[0][0] == "search_course_chunks"
    assert tools.calls[0][1]["query"] == "deadlock"
    assert response.answer == "Deadlock là … [1]."
    assert response.trace.llm_calls == 2
    assert [t.tool for t in response.trace.tool_calls] == ["search_chunks"]
    assert response.trace.tool_calls[0].found == 2
    assert response.citations[0].chunk_id == "c1"
    # Quan sát của bước 1 được đưa lại cho mô hình ở bước 2.
    assert "[1] (Ch5.pdf" in model.prompts[1]


def test_model_cannot_override_the_course():
    model = FakeModel(
        tool("search_chunks", query="x", course_id="another-course"),
        tool("get_page", document_id="d1", page=3, course_id="another-course"),
        answer("ok"),
    )
    tools = FakeTools({**SEARCH, "get_document_page": {"chunks": [chunk(3, "Trang 3.", page=3)]}})
    ask(TutorService(model, tools), mode="tools")
    assert tools.calls[0][1]["course_id"] == COURSE
    assert tools.calls[1] == ("get_document_page", {"course_id": COURSE, "document_id": "d1", "page": 3})


def test_page_lookup_by_evidence_number_uses_that_chunks_page():
    model = FakeModel(tool("search_chunks", query="x"), tool("get_page", evidence=2), answer("Xong [1]."))
    tools = FakeTools({**SEARCH, "get_document_page": {"chunks": [chunk(3, "Cả trang 6.", page=6)]}})
    response = ask(TutorService(model, tools), mode="tools")
    assert tools.calls[1] == ("get_document_page", {"course_id": COURSE, "document_id": "d1", "page": 6})
    assert response.trace.tool_calls[1].found == 1


def test_neighbors_use_the_chunk_behind_the_evidence_number():
    model = FakeModel(tool("search_chunks", query="x"), tool("get_neighbors", evidence=1, window=2), answer("ok"))
    tools = FakeTools({**SEARCH, "get_neighbor_chunks": {"chunks": [chunk(3, "Đoạn kế.")]}})
    ask(TutorService(model, tools), mode="tools")
    assert tools.calls[1] == ("get_neighbor_chunks", {"chunk_id": "c1", "window": 2})


def test_outline_observation_lists_chapters_and_los():
    outline = {"course": {"code": "CO3137", "title": "Dữ liệu lớn"},
               "chapters": [{"code": "5", "title": "Streaming", "topics": "Kafka", "los": ["L.O.2.2"]}],
               "learning_outcomes": [{"code": "L.O.2.2", "statement": "So sánh", "bloom_level": "analyze"}],
               "assessments": []}
    model = FakeModel(tool("get_outline"), answer("Chương 5 học Streaming."))
    tools = FakeTools({"get_course_outline": outline})
    ask(TutorService(model, tools), mode="tools")
    assert "Chapter 5: Streaming — Kafka (LO: L.O.2.2)" in model.prompts[1]


def test_unknown_tool_and_bad_args_are_reported_back_not_raised():
    model = FakeModel(
        tool("drop_database"),
        tool("get_page", evidence=7),
        tool("search_chunks"),
        answer("Không tìm được."),
    )
    tools = FakeTools(SEARCH)
    response = ask(TutorService(model, tools), mode="tools")
    assert tools.calls == []                      # không lời gọi nào tới MCP
    assert [t.ok for t in response.trace.tool_calls] == [False, False, False]
    assert "unknown tool" in model.prompts[1]
    assert "there is no evidence [7]" in model.prompts[2]
    assert "'query' is required" in model.prompts[3]


def test_repeating_the_same_call_is_refused():
    model = FakeModel(tool("search_chunks", query="x"), tool("search_chunks", query="x"), answer("ok"))
    tools = FakeTools(SEARCH)
    response = ask(TutorService(model, tools), mode="tools")
    assert len(tools.calls) == 1
    assert response.trace.tool_calls[1].error == "repeated call"


def test_a_tool_failure_becomes_an_observation():
    model = FakeModel(tool("search_chunks", query="x"), answer("Không tra được."))
    tools = FakeTools({"search_course_chunks": MCPToolError("mcp down")})
    response = ask(TutorService(model, tools), mode="tools")
    assert response.trace.tool_calls[0].ok is False
    assert "mcp down" in model.prompts[1]
    assert response.answer == "Không tra được."


def test_running_out_of_calls_forces_an_answer():
    tools = FakeTools(SEARCH)
    model = FakeModel(
        tool("search_chunks", query="a"), tool("search_chunks", query="b"),
        answer("Đành trả lời với những gì có [1]."),   # lượt bị ép, sau khi hết 2 lời gọi
    )
    response = ask(TutorService(model, tools, max_tool_calls_tools=2), mode="tools")
    assert len(response.trace.tool_calls) == 2
    assert response.trace.forced_answer is True
    assert "no tool calls left" in model.prompts[2].lower()
    assert response.answer.startswith("Đành trả lời")


def test_a_model_that_keeps_asking_for_tools_still_ends():
    model = FakeModel(tool("search_chunks", query="a"), tool("search_chunks", query="b"),
                      tool("search_chunks", query="c"), "câu trả lời bằng chữ thường")
    response = ask(TutorService(model, FakeTools(SEARCH), max_tool_calls_tools=2), mode="tools")
    assert response.trace.forced_answer is True
    assert response.answer == "câu trả lời bằng chữ thường"


def test_invalid_json_is_retried_then_used_as_the_answer():
    model = FakeModel("không phải json", "vẫn không phải json", "lần ba cũng không")
    response = ask(TutorService(model, FakeTools()), mode="tools")
    assert response.answer == "lần ba cũng không"
    assert response.trace.llm_calls == 3
    assert "not a single valid JSON" in model.prompts[1]


def test_json_in_a_code_fence_is_accepted():
    fenced_tool = "```json\n" + tool("search_chunks", query="deadlock") + "\n```"
    fenced_answer = "```json\n" + answer("Đáp án [1]") + "\n```"
    assert ask(TutorService(FakeModel(fenced_tool, fenced_answer), FakeTools(SEARCH)), mode="tools").answer == "Đáp án [1]"


def test_answer_without_any_tool_call_prompts_the_model_to_search_first():
    model = FakeModel(answer("Tự trả lời không tra cứu"), tool("search_chunks", query="deadlock"), answer("Deadlock là … [1]."))
    tools = FakeTools(SEARCH)
    response = ask(TutorService(model, tools), mode="tools")
    assert "not called any tool yet" in model.prompts[1]
    assert len(tools.calls) == 1
    assert response.answer == "Deadlock là … [1]."


def test_tool_name_in_action_with_a_premature_answer_runs_the_tool_first():
    # Lỗi thật của deepseek-chat: tên công cụ nằm ở "action", tham số phẳng, rồi tự viết luôn câu trả lời.
    premature = '{"action": "search_chunks", "query": "deadlock", "document_id": ""}\n\n' + answer("bịa")
    model = FakeModel(premature, answer("Deadlock là … [1]."))
    tools = FakeTools(SEARCH)
    response = ask(TutorService(model, tools), mode="tools")

    assert tools.calls[0] == ("search_course_chunks", {"course_id": COURSE, "query": "deadlock", "top_k": 6})
    assert response.answer == "Deadlock là … [1]."
    assert response.citations[0].chunk_id == "c1"


def test_open_page_is_told_to_the_model_in_tool_modes():
    model = FakeModel(tool("get_page", document_id="d1", page=7), answer("ok"))
    ask(TutorService(model, FakeTools()), mode="tools", document_id="d1", page_number=7, lo_code="L.O.2.1")
    assert "document d1 open, page 7" in model.prompts[0]
    assert "Current learning outcome focus: L.O.2.1." in model.prompts[0]


# ----------------------------------------------------------------- graph

RELATIONS = {"get_concept_relations": {"relations": [
    {"source": "Hold and wait", "relation": "cause", "target": "Deadlock",
     "evidence_quote": "giữ và chờ gây ra deadlock", "chunk_id": "c9", "document_id": "d1", "page_number": 8, "hop": 1},
]}}


def test_graph_mode_offers_relations_and_cites_the_quote():
    model = FakeModel(tool("get_relations", concept="deadlock", hops=2), answer("Giữ và chờ gây deadlock [1]."))
    tools = FakeTools(RELATIONS)
    response = ask(TutorService(model, tools), mode="graph")

    assert "get_relations" in model.systems[0]
    assert tools.calls[0] == ("get_concept_relations", {"course_id": COURSE, "concept": "deadlock", "hops": 2, "limit": 15})
    assert "[1] Hold and wait --cause--> Deadlock" in model.prompts[1]
    assert response.citations[0].chunk_id == "c9"
    assert response.citations[0].quote == "giữ và chờ gây ra deadlock"
    assert response.citations[0].page_number == 8


def test_tools_mode_does_not_offer_or_allow_relations():
    model = FakeModel(tool("get_relations", concept="deadlock"), answer("ok"))
    tools = FakeTools(RELATIONS)
    response = ask(TutorService(model, tools), mode="tools")
    assert "get_relations" not in model.systems[0]
    assert tools.calls == []
    assert response.trace.tool_calls[0].ok is False


def test_call_budget_differs_by_mode():
    tools_model = FakeModel(tool("search_chunks", query="a"), answer("a"))
    graph_model = FakeModel(tool("search_chunks", query="b"), answer("b"))
    ask(TutorService(tools_model, FakeTools(SEARCH)), mode="tools")
    ask(TutorService(graph_model, FakeTools(SEARCH)), mode="graph")
    assert "at most 5 tool calls" in tools_model.systems[0]
    assert "at most 6 tool calls" in graph_model.systems[0]


# ----------------------------------------------------------------- parse_action

@pytest.mark.parametrize("text,expected", [
    ('{"action": "answer", "answer": "x"}', {"action": "answer", "answer": "x"}),
    ('Sure! {"action": "tool", "tool": "get_outline", "args": {}} thanks', {"action": "tool", "tool": "get_outline", "args": {}}),
    ('{"action": "tool", "tool": "get_outline", "args": {}}\n{"action": "answer", "answer": "x"}',
     {"action": "tool", "tool": "get_outline", "args": {}}),
    ('{broken {"action": "answer", "answer": "x"}', {"action": "answer", "answer": "x"}),
    ("no json here", None),
    ("[1, 2, 3]", None),
])
def test_parse_action(text, expected):
    assert parse_action(text) == expected


def test_history_strips_citation_markers_from_assistant_replies_only():
    model = FakeModel("Câu trả lời [1].")
    ask(
        TutorService(model, FakeTools(SEARCH)),
        mode="vector",
        history=[
            ConversationMessage(role="user", content="Mục [1] nói gì?"),
            ConversationMessage(role="assistant", content="Deadlock gồm 4 điều kiện [1, 2] và `arr[1]` trong code [3]."),
        ],
    )
    prompt = model.prompts[0]
    assert "user: Mục [1] nói gì?" in prompt
    assert "assistant: Deadlock gồm 4 điều kiện và `arr[1]` trong code." in prompt


def test_stream_emits_step_evidence_and_done_events():
    model = FakeModel(tool("search_chunks", query="deadlock"), answer("Deadlock là … [1]."))
    events = list(
        TutorService(model, FakeTools(SEARCH)).stream(
            TutorRequest(course_id=COURSE, message="Deadlock là gì?", mode="tools")
        )
    )
    assert [e["event"] for e in events] == ["step", "step", "evidence", "step", "delta", "done"]
    assert events[1]["tool"] == "search_chunks"
    assert events[2]["citations"][0]["chunk_id"] == "c1"
    assert events[3]["phase"] == "answering"
    assert events[5]["response"]["answer"] == "Deadlock là … [1]."


# ----------------------------------------------------------------- streaming thật

class StreamingFakeModel(FakeModel):
    """Như FakeModel nhưng nhả câu trả lời từng vài ký tự, như provider thật."""

    def __init__(self, *replies: str, piece: int = 3) -> None:
        super().__init__(*replies)
        self.piece = piece
        self.streamed_calls = 0

    def stream(self, prompt: str, system: str):
        self.streamed_calls += 1
        reply = self.generate(prompt, system)
        for i in range(0, len(reply.text), self.piece):
            yield reply.text[i : i + self.piece]
        yield reply


def stream_events(service, mode, message="Deadlock là gì?"):
    return list(service.stream(TutorRequest(course_id=COURSE, message=message, mode=mode)))


def deltas(events) -> str:
    return "".join(e["text"] for e in events if e["event"] == "delta")


def test_vector_stream_sends_the_answer_piece_by_piece():
    model = StreamingFakeModel("Deadlock là chờ vòng tròn [1].")
    events = stream_events(TutorService(model, FakeTools(SEARCH)), "vector")

    assert sum(e["event"] == "delta" for e in events) > 1
    assert deltas(events) == "Deadlock là chờ vòng tròn [1]."
    assert events[-1]["event"] == "done"
    assert events[-1]["response"]["citations"][0]["chunk_id"] == "c1"


def test_tool_stream_sends_only_the_decoded_answer_not_the_json():
    text = 'Deadlock là "chờ vòng" [1].\nHai dòng.'
    model = StreamingFakeModel(tool("search_chunks", query="deadlock"), answer(text))
    events = stream_events(TutorService(model, FakeTools(SEARCH)), "tools")

    assert deltas(events) == text
    assert events[-1]["response"]["answer"] == text
    # Lượt gọi công cụ đầu tiên chưa được nhận làm câu trả lời nên không chạy theo luồng.
    assert model.streamed_calls == 1


def test_premature_answer_before_any_tool_is_never_streamed():
    model = StreamingFakeModel(answer("Tự bịa"), tool("search_chunks", query="deadlock"), answer("Đúng [1]."))
    events = stream_events(TutorService(model, FakeTools(SEARCH)), "tools")

    assert "Tự bịa" not in deltas(events)
    assert deltas(events) == "Đúng [1]."
    assert all(e["event"] != "reset" for e in events)


def test_a_streamed_answer_that_turns_out_broken_is_reset():
    broken = '{"action": "answer", "answer": "nửa câu'  # JSON đứt giữa chừng
    model = StreamingFakeModel(tool("search_chunks", query="deadlock"), broken, answer("Câu đầy đủ [1]."))
    events = stream_events(TutorService(model, FakeTools(SEARCH)), "tools")
    names = [e["event"] for e in events]

    reset_at = names.index("reset")
    before = "".join(e["text"] for e in events[:reset_at] if e["event"] == "delta")
    after = "".join(e["text"] for e in events[reset_at:] if e["event"] == "delta")
    assert before == "nửa câu"
    assert after == "Câu đầy đủ [1]."
    assert events[-1]["response"]["answer"] == "Câu đầy đủ [1]."


def test_non_stream_endpoint_does_not_use_streaming():
    model = StreamingFakeModel(tool("search_chunks", query="deadlock"), answer("ok [1]"))
    ask(TutorService(model, FakeTools(SEARCH)), mode="tools")
    assert model.streamed_calls == 0

