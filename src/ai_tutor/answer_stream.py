"""
Rút dần câu trả lời từ JSON mô hình đang sinh ở chế độ tools/graph.

Mô hình trả {"action": "answer", "answer": "..."}: để người học thấy chữ hiện dần, đọc
từng mẩu và giải mã giá trị chuỗi của khóa "answer" ở mức ngoài cùng ngay khi tới.
Chỉ tính object JSON đầu tiên (giống parse_action); mọi thứ trước dấu { đầu tiên
(rào ```json, lời dẫn) bị bỏ qua. Lời gọi công cụ không có khóa "answer" nên không
sinh ra chữ nào.
"""
from __future__ import annotations

_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class AnswerStream:
    def __init__(self) -> None:
        self._depth = 0
        self._finished = False
        self._in_string = False
        self._escape: str | None = None  # đang đọc dở chuỗi thoát sau dấu \
        self._high_surrogate: int | None = None
        self._key_chars: list[str] | None = None  # chỉ gom khi chuỗi là khóa ở mức ngoài cùng
        self._capturing = False  # đang ở trong giá trị của "answer"
        self._last_key: str | None = None
        self._expect_value = False  # đã qua dấu : ở mức ngoài cùng
        self.text = ""  # toàn bộ câu trả lời đã giải mã tới giờ

    @property
    def finished(self) -> bool:
        return self._finished

    def feed(self, chunk: str) -> str:
        """Nhận mẩu chữ mới của mô hình, trả phần câu trả lời mới giải mã được (có thể rỗng)."""
        out: list[str] = []
        for ch in chunk:
            if self._finished:
                break
            if self._in_string:
                self._string_char(ch, out)
            else:
                self._structural_char(ch)
        piece = "".join(out)
        self.text += piece
        return piece

    def _structural_char(self, ch: str) -> None:
        if self._depth == 0:
            if ch == "{":
                self._depth = 1
            return
        if ch == '"':
            self._in_string = True
            top = self._depth == 1
            self._capturing = top and self._expect_value and self._last_key == "answer"
            self._key_chars = [] if top and not self._expect_value else None
        elif ch == ":" and self._depth == 1:
            self._expect_value = True
        elif ch == "," and self._depth == 1:
            self._expect_value = False
            self._last_key = None
        elif ch in "{[":
            self._depth += 1
        elif ch in "}]":
            self._depth -= 1
            if self._depth == 0:
                self._finished = True

    def _string_char(self, ch: str, out: list[str]) -> None:
        if self._escape is not None:
            self._escape += ch
            if self._escape[0] == "u":
                if len(self._escape) < 5:
                    return
                try:
                    code = int(self._escape[1:], 16)
                except ValueError:
                    code = 0xFFFD
                self._escape = None
                self._emit_code(code, out)
            else:
                decoded = _SIMPLE_ESCAPES.get(self._escape, self._escape)
                self._escape = None
                self._emit(decoded, out)
            return
        if ch == "\\":
            self._escape = ""
            return
        if ch == '"':
            self._in_string = False
            if self._capturing:
                # Đủ câu trả lời; phần còn lại của object không cần nữa.
                self._capturing = False
                self._finished = True
            elif self._key_chars is not None:
                self._last_key = "".join(self._key_chars)
                self._key_chars = None
            return
        self._emit(ch, out)

    def _emit_code(self, code: int, out: list[str]) -> None:
        if 0xD800 <= code < 0xDC00:
            self._high_surrogate = code
            return
        if 0xDC00 <= code < 0xE000 and self._high_surrogate is not None:
            code = 0x10000 + ((self._high_surrogate - 0xD800) << 10) + (code - 0xDC00)
        self._high_surrogate = None
        self._emit(chr(code) if code < 0xD800 or code >= 0xE000 else "\ufffd", out)

    def _emit(self, text: str, out: list[str]) -> None:
        if self._capturing:
            out.append(text)
        elif self._key_chars is not None:
            self._key_chars.append(text)

