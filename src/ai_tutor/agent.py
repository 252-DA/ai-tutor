"""
Vòng gọi công cụ cho chế độ tools và graph.

Mỗi lượt mô hình trả đúng một JSON: gọi một công cụ hoặc trả lời. Giao thức tự
định nghĩa (không dùng function-calling của nhà cung cấp) để hai provider hiện có
chạy giống hệt nhau; mã kiểm tên công cụ, tham số, số lượt và việc lặp lại.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any

from ai_runtime.errors import MCPToolError

from ai_tutor import prompts
from ai_tutor.answer_stream import AnswerStream
from ai_tutor.evidence import EvidenceBook
from ai_tutor.models import ToolCallTrace, TutorRequest, TutorTrace
from ai_tutor.ports import CourseTools, ModelReply, TutorModel, stream_reply
from ai_tutor.tools import (
    TOOLS,
    ToolArgumentError,
    build_mcp_call,
    format_observation,
    toolset_for,
)

# Trả lời sai định dạng thì được nhắc lại tối đa chừng này lần trong cả một lượt hỏi.
MAX_FORMAT_RETRIES = 2


@dataclass(frozen=True)
class AgentResult:
    answer: str
    forced: bool


def parse_action(text: str) -> dict[str, Any] | None:
    """
    Lấy object JSON đầu tiên trong câu trả lời (bỏ rào ```json nếu có).

    Mô hình đôi khi viết hai object liền nhau trong một lượt (gọi công cụ rồi tự trả lời luôn,
    khi chưa thấy kết quả); chỉ object đầu tiên được tính, phần sau bỏ.
    """
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    start = raw.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            start = raw.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = raw.find("{", start + 1)
    return None


def _normalize(action: dict[str, Any]) -> dict[str, Any]:
    """{"action": "search_chunks", "query": ...}: tên công cụ đặt nhầm vào "action", tham số để phẳng."""
    name = str(action.get("action", "")).strip()
    if name not in TOOLS:
        return action
    args = action.get("args")
    if not isinstance(args, dict):
        args = {k: v for k, v in action.items() if k not in ("action", "tool")}
    return {"action": "tool", "tool": name, "args": args}


def _call_key(tool: str, args: dict[str, Any]) -> str:
    return tool + json.dumps(args, sort_keys=True, ensure_ascii=False)


def iter_tool_loop(
    *,
    request: TutorRequest,
    model: TutorModel,
    tools: CourseTools,
    book: EvidenceBook,
    trace: TutorTrace,
    max_tool_calls: int,
    stream_answer: bool = False,
) -> Iterator[dict[str, Any] | AgentResult]:
    """
    stream_answer: phát {"event": "delta"} khi mô hình đang viết câu trả lời (luồng SSE).
    Nếu lượt đó rốt cuộc không được nhận làm câu trả lời (JSON hỏng, chưa gọi công cụ…)
    thì phát {"event": "reset"} để giao diện xóa phần chữ đã hiện.
    """
    names = toolset_for(request.mode)
    system = prompts.tool_system_prompt(request.mode, max_tool_calls)
    scratchpad: list[str] = []
    seen_calls: set[str] = set()
    format_retries = 0
    streamed = False

    def ask(extra: str | None = None, *, live: bool = False) -> Generator[dict[str, Any], None, str]:
        nonlocal streamed
        prompt = prompts.tool_prompt(request, scratchpad)
        if extra:
            prompt += f"\n\n{extra}"
        if live:
            extractor = AnswerStream()
            reply: ModelReply | None = None
            for item in stream_reply(model, prompt, system):
                if isinstance(item, ModelReply):
                    reply = item
                    continue
                piece = extractor.feed(item)
                if piece:
                    if not streamed:
                        yield {"event": "step", "phase": "answering"}
                        streamed = True
                    yield {"event": "delta", "text": piece}
            if reply is None:
                reply = ModelReply(text=extractor.text)
        else:
            reply = model.generate(prompt, system)
        trace.llm_calls += 1
        trace.input_tokens += reply.input_tokens
        trace.output_tokens += reply.output_tokens
        return reply.text

    def would_accept_answer(out_of_calls: bool) -> bool:
        # Cùng điều kiện nhận câu trả lời ở dưới: chỉ cho chữ chạy khi lượt này sẽ được dùng.
        return bool(trace.tool_calls) or out_of_calls or format_retries >= MAX_FORMAT_RETRIES

    while True:
        if streamed:
            # Lượt trước đã cho chữ chạy nhưng không thành câu trả lời.
            streamed = False
            yield {"event": "reset"}
        out_of_calls = len(trace.tool_calls) >= max_tool_calls
        if out_of_calls:
            yield {"event": "step", "phase": "answering"}
        text = yield from ask(
            prompts.FORCE_ANSWER if out_of_calls else None,
            live=stream_answer and would_accept_answer(out_of_calls),
        )
        trace.steps += 1
        action = parse_action(text)

        if action is None:
            if out_of_calls or format_retries >= MAX_FORMAT_RETRIES:
                # Không còn cách bắt mô hình nói đúng định dạng: dùng nguyên văn làm câu trả lời.
                yield AgentResult(answer=text.strip(), forced=out_of_calls)
                return
            format_retries += 1
            scratchpad.append(
                "(Your previous reply was not a single valid JSON object. "
                'Reply with {"action": "tool", ...} or {"action": "answer", ...}.)'
            )
            continue

        action = _normalize(action)
        kind = str(action.get("action", "")).lower()
        if kind == "answer" or (kind == "" and "answer" in action):
            if not trace.tool_calls and not out_of_calls and format_retries < MAX_FORMAT_RETRIES:
                format_retries += 1
                scratchpad.append(
                    "(You have not called any tool yet. Call a tool such as search_chunks "
                    "or get_outline to check the course material before answering.)"
                )
                continue
            yield AgentResult(answer=str(action.get("answer", "")).strip(), forced=out_of_calls)
            return

        if kind != "tool":
            if format_retries >= MAX_FORMAT_RETRIES:
                yield AgentResult(answer=text.strip(), forced=out_of_calls)
                return
            format_retries += 1
            scratchpad.append('(Unknown "action". Use "tool" or "answer".)')
            continue

        if out_of_calls:
            # Đã nhắc ép trả lời mà vẫn đòi công cụ: lần sau không còn gì để đòi.
            scratchpad.append("(No tool calls left; answer now.)")
            if streamed:
                streamed = False
                yield {"event": "reset"}
            yield {"event": "step", "phase": "answering"}
            text = yield from ask(prompts.FORCE_ANSWER, live=stream_answer)
            trace.steps += 1
            final = parse_action(text)
            if final and "answer" in final:
                yield AgentResult(answer=str(final["answer"]).strip(), forced=True)
                return
            yield AgentResult(answer=text.strip(), forced=True)
            return

        tool = str(action.get("tool", "")).strip()
        raw_args = action.get("args") or {}
        step_no = len(trace.tool_calls) + 1
        header = f"Step {step_no}: {tool}({json.dumps(raw_args, ensure_ascii=False)})"
        call = ToolCallTrace(tool=tool, args=raw_args if isinstance(raw_args, dict) else {})
        trace.tool_calls.append(call)

        if tool not in names or tool not in TOOLS:
            call.ok, call.error = False, "unknown tool"
            scratchpad.append(f"{header}\nResult: error — unknown tool. Available: {', '.join(names)}")
            continue

        try:
            mcp_args = build_mcp_call(
                tool, raw_args, course_id=request.course_id, top_k=request.top_k, book=book
            )
        except ToolArgumentError as exc:
            call.ok, call.error = False, str(exc)
            scratchpad.append(f"{header}\nResult: error — {exc}")
            continue

        key = _call_key(tool, mcp_args)
        if key in seen_calls:
            call.ok, call.error = False, "repeated call"
            scratchpad.append(
                f"{header}\nResult: you already made this exact call; use the evidence you have."
            )
            continue
        seen_calls.add(key)

        yield {"event": "step", "phase": "searching", "tool": tool}
        before_count = len(book)
        started = time.monotonic()
        try:
            result = tools.call(TOOLS[tool].mcp_name, mcp_args)
            observation, found = format_observation(tool, result, book)
            call.found = found
        except MCPToolError as exc:
            call.ok, call.error = False, str(exc)[:200]
            observation, found = f"error — {str(exc)[:200]}", 0
        call.ms = int((time.monotonic() - started) * 1000)
        scratchpad.append(f"{header}\nResult:\n{observation}")
        if len(book) > before_count:
            yield {
                "event": "evidence",
                "citations": [c.model_dump() for c in book.all_citations()],
            }


def run_tool_loop(
    *,
    request: TutorRequest,
    model: TutorModel,
    tools: CourseTools,
    book: EvidenceBook,
    trace: TutorTrace,
    max_tool_calls: int,
) -> AgentResult:
    for item in iter_tool_loop(
        request=request,
        model=model,
        tools=tools,
        book=book,
        trace=trace,
        max_tool_calls=max_tool_calls,
    ):
        if isinstance(item, AgentResult):
            return item
    return AgentResult(answer="", forced=True)
