"""
Công cụ mô hình được dùng ở chế độ tools và graph.

Mô hình không bao giờ tự đặt ``course_id``: vòng lặp gắn course_id của yêu cầu
vào mọi lời gọi, nên một câu hỏi không thể tra khóa học khác. Mô hình chỉ thấy số
bằng chứng [n], không phải UUID, ở những chỗ nó trỏ lại một đoạn đã thấy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_tutor.evidence import EvidenceBook

OBSERVATION_CHARS = 6000
CHUNK_PREVIEW_CHARS = 900


@dataclass(frozen=True)
class ToolSpec:
    name: str          # tên mô hình dùng
    mcp_name: str      # tên công cụ MCP
    doc: str           # một dòng cho prompt
    example: str


TOOLS: dict[str, ToolSpec] = {
    "search_chunks": ToolSpec(
        "search_chunks", "search_course_chunks",
        "Semantic search over the course material. Optional document_id limits it to one document.",
        '{"query": "...", "document_id": "<uuid, optional>"}',
    ),
    "get_page": ToolSpec(
        "get_page", "get_document_page",
        "Every chunk on one page of a document (exact lookup). Give document_id and page, "
        "or evidence=<n> to read the page of evidence [n].",
        '{"document_id": "<uuid>", "page": 7}  or  {"evidence": 2}',
    ),
    "list_documents": ToolSpec(
        "list_documents", "list_course_documents",
        "List the course documents with ids, chapters and page counts.",
        '{"chapter_code": "<optional>"}',
    ),
    "get_outline": ToolSpec(
        "get_outline", "get_course_outline",
        "Chapters, learning outcomes (LO) and assessments from the syllabus; use it for "
        "'which chapter/LO covers X' or 'what is assessed' questions.",
        '{"chapter_code": "<optional>"}',
    ),
    "get_neighbors": ToolSpec(
        "get_neighbors", "get_neighbor_chunks",
        "The chunks just before and after evidence [n]; use when a chunk is cut off or only a title.",
        '{"evidence": 2, "window": 1}',
    ),
    "get_relations": ToolSpec(
        "get_relations", "get_concept_relations",
        "Typed relations (cause, part_of, contrast, prerequisite, example_of, defines, related) "
        "around a concept, each with a verbatim quote from the slides; use for why/how/compare "
        "questions. hops is 1 or 2.",
        '{"concept": "...", "hops": 1}',
    ),
}

TOOLSET_TOOLS = ("search_chunks", "get_page", "list_documents", "get_outline", "get_neighbors")
TOOLSET_GRAPH = TOOLSET_TOOLS + ("get_relations",)


def toolset_for(mode: str) -> tuple[str, ...]:
    return TOOLSET_GRAPH if mode == "graph" else TOOLSET_TOOLS


class ToolArgumentError(ValueError):
    """Mô hình gọi sai tham số; thông điệp được đưa lại cho nó để sửa."""


def _text(raw: dict, key: str, *, required: bool, limit: int = 500) -> str | None:
    value = raw.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ToolArgumentError(f"'{key}' is required")
        return None
    if not isinstance(value, (str, int)):
        raise ToolArgumentError(f"'{key}' must be a string")
    return str(value).strip()[:limit]


def _int(raw: dict, key: str, *, default: int | None, low: int, high: int) -> int | None:
    value = raw.get(key)
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ToolArgumentError(f"'{key}' must be an integer") from None
    return max(low, min(high, number))


def build_mcp_call(
    tool: str,
    raw: dict[str, Any],
    *,
    course_id: str,
    top_k: int,
    book: EvidenceBook,
) -> dict[str, Any]:
    """Tham số MCP đã kiểm tra cho một lời gọi của mô hình. Ném ToolArgumentError nếu sai."""
    if not isinstance(raw, dict):
        raise ToolArgumentError("args must be a JSON object")

    if tool == "search_chunks":
        args: dict[str, Any] = {
            "course_id": course_id,
            "query": _text(raw, "query", required=True, limit=500),
            "top_k": top_k,
        }
        document_id = _text(raw, "document_id", required=False, limit=100)
        if document_id:
            args["document_ids"] = [document_id]
        return args

    if tool == "get_page":
        evidence = _int(raw, "evidence", default=None, low=1, high=999)
        if evidence is not None:
            item = book.get(evidence)
            if item is None:
                raise ToolArgumentError(f"there is no evidence [{evidence}]")
            if item.page_number is None:
                raise ToolArgumentError(f"evidence [{evidence}] has no page number")
            return {
                "course_id": course_id,
                "document_id": item.document_id,
                "page": item.page_number,
            }
        return {
            "course_id": course_id,
            "document_id": _text(raw, "document_id", required=True, limit=100),
            "page": _int(raw, "page", default=None, low=1, high=9999)
            or _fail("'page' is required"),
        }

    if tool == "list_documents":
        args = {"course_id": course_id}
        chapter = _text(raw, "chapter_code", required=False, limit=20)
        if chapter:
            args["chapter_code"] = chapter
        return args

    if tool == "get_outline":
        args = {"course_id": course_id}
        chapter = _text(raw, "chapter_code", required=False, limit=20)
        if chapter:
            args["chapter_code"] = chapter
        return args

    if tool == "get_neighbors":
        evidence = _int(raw, "evidence", default=None, low=1, high=999)
        if evidence is None:
            raise ToolArgumentError("'evidence' is required")
        item = book.get(evidence)
        if item is None:
            raise ToolArgumentError(f"there is no evidence [{evidence}]")
        return {
            "chunk_id": item.chunk_id,
            "window": _int(raw, "window", default=1, low=1, high=3),
        }

    if tool == "get_relations":
        return {
            "course_id": course_id,
            "concept": _text(raw, "concept", required=True, limit=120),
            "hops": _int(raw, "hops", default=1, low=1, high=2),
            "limit": 15,
        }

    raise ToolArgumentError(f"unknown tool '{tool}'")


def _fail(message: str):
    raise ToolArgumentError(message)


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def chunk_line(item) -> str:
    where = " · ".join(
        part
        for part in (
            item.document_name,
            " > ".join(item.heading_path),
            f"tr. {item.page_number}" if item.page_number is not None else "",
        )
        if part
    )
    head = f"[{item.index}]" + (f" ({where})" if where else "")
    return f"{head}\n{_clip(item.text, CHUNK_PREVIEW_CHARS)}"


def add_chunks(book: EvidenceBook, rows: list[dict], *, name_key: str | None = None):
    items = []
    for row in rows:
        item = book.add(
            chunk_id=str(row.get("chunk_id")),
            document_id=str(row.get("document_id") or ""),
            heading_path=row.get("heading_path") or [],
            page_number=row.get("page_number"),
            text=str(row.get("content") or ""),
            document_name=row.get(name_key) if name_key else None,
        )
        if item is not None:
            items.append(item)
    return items


def format_observation(tool: str, result: dict[str, Any], book: EvidenceBook) -> tuple[str, int]:
    """(văn bản đưa lại cho mô hình, số kết quả). Bằng chứng mới được ghi vào sổ."""
    if tool == "search_chunks":
        items = add_chunks(book, result.get("results") or [], name_key="document_name")
        text = "\n\n".join(chunk_line(i) for i in items) or "No matching chunks."
        return _limit(text), len(items)

    if tool in ("get_page", "get_neighbors"):
        items = add_chunks(book, result.get("chunks") or [])
        text = "\n\n".join(chunk_line(i) for i in items) or "No chunks found."
        return _limit(text), len(items)

    if tool == "list_documents":
        docs = result.get("documents") or []
        lines = [
            f"- {d.get('title')} | chapter {d.get('chapter_code') or '-'} | "
            f"{d.get('role')} | {d.get('pages')} pages | document_id={d.get('document_id')}"
            for d in docs
        ]
        return _limit("\n".join(lines) or "No documents."), len(docs)

    if tool == "get_outline":
        lines = [f"Course {result.get('course', {}).get('code')}: {result.get('course', {}).get('title')}"]
        for chapter in result.get("chapters") or []:
            los = ", ".join(chapter.get("los") or []) or "-"
            topics = f" — {chapter['topics']}" if chapter.get("topics") else ""
            lines.append(f"- Chapter {chapter.get('code')}: {chapter.get('title')}{topics} (LO: {los})")
        for lo in result.get("learning_outcomes") or []:
            lines.append(f"- {lo.get('code')} [{lo.get('bloom_level') or '?'}]: {lo.get('statement')}")
        for a in result.get("assessments") or []:
            lines.append(f"- Assessment {a.get('code')}: {a.get('name')} ({a.get('category')}, weight {a.get('weight')})")
        return _limit("\n".join(lines)), len(result.get("chapters") or [])

    if tool == "get_relations":
        relations = result.get("relations") or []
        lines = []
        for rel in relations:
            item = book.add(
                chunk_id=str(rel.get("chunk_id")),
                document_id=str(rel.get("document_id") or ""),
                heading_path=[],
                page_number=rel.get("page_number"),
                text=str(rel.get("evidence_quote") or ""),
                quote=rel.get("evidence_quote"),
            )
            ref = f"[{item.index}]" if item else "[?]"
            page = f", tr. {rel['page_number']}" if rel.get("page_number") is not None else ""
            lines.append(
                f"{ref} {rel.get('source')} --{rel.get('relation')}--> {rel.get('target')} "
                f"(hop {rel.get('hop')}{page}) — \"{_clip(str(rel.get('evidence_quote')), 300)}\""
            )
        text = "\n".join(lines) or "No relations found for this concept."
        return _limit(text), len(relations)

    return "", 0


def _limit(text: str) -> str:
    return text if len(text) <= OBSERVATION_CHARS else text[:OBSERVATION_CHARS].rstrip() + "\n…(truncated)"
