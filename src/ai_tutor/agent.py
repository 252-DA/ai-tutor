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
from dataclasses import dataclass
from typing import Any

from ai_runtime.errors import MCPToolError

from ai_tutor import prompts
from ai_tutor.evidence import EvidenceBook
from ai_tutor.models import ToolCallTrace, TutorRequest, TutorTrace
from ai_tutor.ports import CourseTools, TutorModel
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
    """Lấy object JSON đầu tiên trong câu trả lời (bỏ rào ```json nếu có)."""
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    for candidate in (raw, raw[raw.find("{") : raw.rfind("}") + 1] if "{" in raw else ""):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _call_key(tool: str, args: dict[str, Any]) -> str:
    return tool + json.dumps(args, sort_keys=True, ensure_ascii=False)


def run_tool_loop(
    *,
    request: TutorRequest,
    model: TutorModel,
    tools: CourseTools,
    book: EvidenceBook,
    trace: TutorTrace,
    max_tool_calls: int,
) -> AgentResult:
    names = toolset_for(request.mode)
    system = prompts.tool_system_prompt(request.mode, max_tool_calls)
    scratchpad: list[str] = []
    seen_calls: set[str] = set()
    format_retries = 0
    forced = False

    def ask(extra: str | None = None) -> str:
        prompt = prompts.tool_prompt(request, scratchpad)
        if extra:
            prompt += f"\n\n{extra}"
        reply = model.generate(prompt, system)
        trace.llm_calls += 1
        trace.input_tokens += reply.input_tokens
        trace.output_tokens += reply.output_tokens
        return reply.text

    while True:
        out_of_calls = len(trace.tool_calls) >= max_tool_calls
        text = ask(prompts.FORCE_ANSWER if out_of_calls else None)
        trace.steps += 1
        action = parse_action(text)

        if action is None:
            if out_of_calls or format_retries >= MAX_FORMAT_RETRIES:
                # Không còn cách bắt mô hình nói đúng định dạng: dùng nguyên văn làm câu trả lời.
                return AgentResult(answer=text.strip(), forced=out_of_calls)
            format_retries += 1
            scratchpad.append(
                "(Your previous reply was not a single valid JSON object. "
                'Reply with {"action": "tool", ...} or {"action": "answer", ...}.)'
            )
            continue

        kind = str(action.get("action", "")).lower()
        if kind == "answer" or (kind == "" and "answer" in action):
            return AgentResult(answer=str(action.get("answer", "")).strip(), forced=out_of_calls)

        if kind != "tool":
            if format_retries >= MAX_FORMAT_RETRIES:
                return AgentResult(answer=text.strip(), forced=out_of_calls)
            format_retries += 1
            scratchpad.append('(Unknown "action". Use "tool" or "answer".)')
            continue

        if out_of_calls:
            # Đã nhắc ép trả lời mà vẫn đòi công cụ: lần sau không còn gì để đòi.
            forced = True
            scratchpad.append("(No tool calls left; answer now.)")
            text = ask(prompts.FORCE_ANSWER)
            trace.steps += 1
            final = parse_action(text)
            if final and "answer" in final:
                return AgentResult(answer=str(final["answer"]).strip(), forced=True)
            return AgentResult(answer=text.strip(), forced=True)

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
