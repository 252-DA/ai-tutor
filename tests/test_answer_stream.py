"""Rút câu trả lời từ JSON đang sinh: mẩu chữ cắt ở bất kỳ đâu cũng phải ra đúng văn bản."""
import json

import pytest

from ai_tutor.answer_stream import AnswerStream


def feed_in_pieces(text: str, size: int) -> tuple[str, AnswerStream]:
    stream = AnswerStream()
    out = "".join(stream.feed(text[i : i + size]) for i in range(0, len(text), size))
    return out, stream


ANSWER = 'Deadlock là "chờ vòng" [1].\nDòng hai\t— emoji 🎓 và \\ gạch.'


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_decodes_the_answer_whatever_the_chunk_boundaries(size):
    raw = json.dumps({"action": "answer", "answer": ANSWER})  # ensure_ascii: có \uXXXX và cặp surrogate
    out, stream = feed_in_pieces(raw, size)
    assert out == ANSWER
    assert stream.finished


def test_non_ascii_written_raw_is_kept():
    raw = json.dumps({"action": "answer", "answer": ANSWER}, ensure_ascii=False)
    assert feed_in_pieces(raw, 4)[0] == ANSWER


def test_tool_call_produces_no_text_even_if_args_mention_answer():
    raw = json.dumps({"action": "tool", "tool": "search_chunks", "args": {"query": '"answer": "x"', "answer": "nested"}})
    out, stream = feed_in_pieces(raw, 3)
    assert out == ""
    assert stream.finished


def test_text_before_the_object_and_code_fence_are_skipped():
    raw = "Sure!\n```json\n" + json.dumps({"action": "answer", "answer": "ok"}) + "\n```"
    assert feed_in_pieces(raw, 5)[0] == "ok"


def test_only_the_first_object_counts():
    raw = json.dumps({"action": "tool", "tool": "get_outline", "args": {}}) + "\n" + json.dumps({"action": "answer", "answer": "bịa"})
    assert feed_in_pieces(raw, 2)[0] == ""


def test_answer_key_before_action_still_streams():
    raw = '{"answer": "a, b: {c}", "action": "answer"}'
    assert feed_in_pieces(raw, 1)[0] == "a, b: {c}"


def test_plain_prose_produces_nothing():
    assert feed_in_pieces("Deadlock là trạng thái chờ nhau.", 4)[0] == ""

