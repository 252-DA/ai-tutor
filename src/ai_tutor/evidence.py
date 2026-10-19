"""
Sổ bằng chứng: mọi đoạn tài liệu mô hình được thấy đều mang một số [n] duy nhất
trong cả hội thoại một lượt hỏi. Câu trả lời trích [n]; sổ đổi [n] về đoạn nguồn
(tài liệu, trang, heading) để giao diện nhảy tới đúng chỗ.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_tutor.models import TutorCitation

_CODE_SPAN = re.compile(r"```.*?```|`[^`\n]+`", flags=re.DOTALL)
_MARKER = re.compile(r"(?<![A-Za-z0-9_])\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")


def _strip_prose_markers(segment: str) -> str:
    cleaned = _MARKER.sub("", segment)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    return re.sub(r" {2,}", " ", cleaned)


def strip_citation_markers(text: str) -> str:
    """Bỏ các ký hiệu trích dẫn [n] ngoài vùng code để đưa câu trả lời cũ vào lịch sử."""
    if not text:
        return ""
    parts: list[str] = []
    last = 0
    for match in _CODE_SPAN.finditer(text):
        parts.append(_strip_prose_markers(text[last : match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(_strip_prose_markers(text[last:]))
    return "".join(parts).strip()


@dataclass
class EvidenceItem:
    index: int
    chunk_id: str
    document_id: str
    heading_path: list[str]
    page_number: int | None
    text: str
    quote: str | None = None
    document_name: str | None = None

    def to_citation(self) -> TutorCitation:
        return TutorCitation(
            index=self.index,
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            heading_path=self.heading_path,
            page_number=self.page_number,
            quote=self.quote,
        )


@dataclass
class EvidenceBook:
    max_items: int = 40
    _items: list[EvidenceItem] = field(default_factory=list)
    _by_chunk: dict[str, EvidenceItem] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        *,
        chunk_id: str,
        document_id: str,
        heading_path: list[str] | tuple[str, ...] | None,
        page_number: int | None,
        text: str,
        quote: str | None = None,
        document_name: str | None = None,
    ) -> EvidenceItem | None:
        """Thêm một đoạn; cùng ``chunk_id`` thì dùng lại số cũ. None khi sổ đầy."""
        existing = self._by_chunk.get(chunk_id)
        if existing is not None:
            if quote and not existing.quote:
                existing.quote = quote
            if document_name and not existing.document_name:
                existing.document_name = document_name
            return existing
        if len(self._items) >= self.max_items:
            return None
        item = EvidenceItem(
            index=len(self._items) + 1,
            chunk_id=chunk_id,
            document_id=document_id,
            heading_path=list(heading_path or []),
            page_number=page_number,
            text=text,
            quote=quote,
            document_name=document_name,
        )
        self._items.append(item)
        self._by_chunk[chunk_id] = item
        return item

    def items(self) -> list[EvidenceItem]:
        return list(self._items)

    def get(self, index: int) -> EvidenceItem | None:
        return self._items[index - 1] if 1 <= index <= len(self._items) else None

    def cited_indexes(self, answer: str) -> list[int]:
        """Các số [n] có thật trong sổ, theo thứ tự xuất hiện đầu tiên."""
        seen: list[int] = []
        prose = _CODE_SPAN.sub(" ", answer or "")
        for match in _MARKER.finditer(prose):
            for raw in re.split(r"[,;]", match.group(1)):
                n = int(raw)
                if self.get(n) is not None and n not in seen:
                    seen.append(n)
        return seen

    def all_citations(self) -> list[TutorCitation]:
        return [item.to_citation() for item in self._items]

    def citations(self, answer: str) -> list[TutorCitation]:
        return [self._items[n - 1].to_citation() for n in self.cited_indexes(answer)]

