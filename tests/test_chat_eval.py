import csv
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "chat_eval.py"
spec = importlib.util.spec_from_file_location("chat_eval", SCRIPT)
chat_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat_eval)


class FakeTutor(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeTutor.seen.append(body)
        if "lỗi" in body["message"]:
            self.send_response(502)
            self.end_headers()
            return
        reply = {
            "answer": f"trả lời {body['mode']} [1]",
            "model": "fake",
            "citations": [{"index": 1, "chunk_id": "c", "document_id": "d", "heading_path": ["A", "B"], "page_number": 3, "quote": "trích"}],
            "trace": {"mode": body["mode"], "steps": 2, "tool_calls": [{"tool": "search_course_chunks"}], "input_tokens": 100, "output_tokens": 20, "latency_ms": 1500},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def start_server():
    server = HTTPServer(("127.0.0.1", 0), FakeTutor)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def write_questions(path: Path):
    rows = [
        {"id": "lookup-01", "type": "lookup", "question": "Câu một?", "reference": "đáp 1", "answerable": True},
        {"id": "absent-01", "type": "absent", "question": "Câu lỗi?", "reference": "không có", "answerable": False},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_seed_questions_are_well_formed():
    rows = chat_eval.read_jsonl(chat_eval.DEFAULT_QUESTIONS)
    assert len(rows) >= 30
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    kinds = {r["type"] for r in rows}
    assert kinds == {"lookup", "definition", "relational", "curriculum", "global", "absent"}
    for r in rows:
        assert r["question"].strip() and r["reference"].strip()
        assert r["answerable"] is (r["type"] != "absent")


def test_run_records_every_mode_and_errors_and_resumes(tmp_path):
    server = start_server()
    try:
        questions = tmp_path / "q.jsonl"
        write_questions(questions)
        out = tmp_path / "runs.jsonl"
        argv = [
            "run", "--questions", str(questions), "--tutor-url", f"http://127.0.0.1:{server.server_port}",
            "--course", "CO3137", "--out", str(out), "--resume",
        ]
        assert chat_eval.main(argv) == 0
        rows = chat_eval.read_jsonl(out)
        assert len(rows) == 6
        assert {r["mode"] for r in rows} == {"vector", "tools", "graph"}
        assert sum(r["status"] == "ERROR" for r in rows) == 3
        ok = next(r for r in rows if r["status"] == "OK")
        assert ok["answer"].endswith("[1]") and ok["trace"]["steps"] == 2
        assert FakeTutor.seen[0]["course_id"] == "CO3137"

        # Chạy lại với --resume: 3 lượt OK được bỏ qua, 3 lượt lỗi chạy lại.
        before = len(FakeTutor.seen)
        chat_eval.main(argv)
        assert len(FakeTutor.seen) - before == 3
    finally:
        server.shutdown()


def test_sheet_hides_mode_and_key_maps_back(tmp_path):
    rows = [
        {"id": f"q{i}", "type": "lookup", "question": f"hỏi {i}", "reference": "r", "answerable": True, "mode": m, "status": "OK",
         "answer": f"đáp {m}", "citations": [{"index": 1, "heading_path": ["H"], "page_number": 2, "quote": "q"}]}
        for i in range(3) for m in chat_eval.MODES
    ]
    sheet, key = chat_eval.build_sheet(rows, seed=1)
    assert len(sheet) == len(key) == 9
    assert all("mode" not in s for s in sheet)
    assert {k["mode"] for k in key} == set(chat_eval.MODES)
    mapped = {k["item"]: k for k in key}
    for s in sheet:
        assert mapped[s["item"]]["id"] and s["answer"].startswith("đáp")
    assert [s["item"] for s in sheet] == sorted(s["item"] for s in sheet)
    # Thứ tự trộn theo seed: cùng seed cho cùng kết quả.
    assert chat_eval.build_sheet(rows, seed=1)[1] == key


def test_summary_combines_ratings_and_kappa():
    rows = [
        {"id": "a", "type": "lookup", "mode": "vector", "status": "OK", "answerable": True, "citations": [{"index": 1}],
         "trace": {"steps": 1, "tool_calls": [], "input_tokens": 100, "output_tokens": 10, "latency_ms": 1000}},
        {"id": "a", "type": "lookup", "mode": "tools", "status": "OK", "answerable": True, "citations": [],
         "trace": {"steps": 3, "tool_calls": [1, 2], "input_tokens": 300, "output_tokens": 30, "latency_ms": 5000}},
        {"id": "b", "type": "lookup", "mode": "tools", "status": "ERROR", "answerable": True, "citations": [], "trace": {}},
    ]
    key = {"R1": {"item": "R1", "id": "a", "mode": "vector"}, "R2": {"item": "R2", "id": "a", "mode": "tools"}}
    rater1 = {"R1": {"item": "R1", "correct": "2", "grounded": "2", "citation_ok": "1"}, "R2": {"item": "R2", "correct": "1", "grounded": "0", "citation_ok": "0"}}
    rater2 = {"R1": {"item": "R1", "correct": "2", "grounded": "2", "citation_ok": "1"}, "R2": {"item": "R2", "correct": "2", "grounded": "0", "citation_ok": "0"}}
    summary = chat_eval.summarize(rows, key, [rater1, rater2])
    vector = summary["groups"]["all/vector"]
    tools = summary["groups"]["all/tools"]
    assert vector["n"] == 1 and vector["steps"] == 1 and vector["correct"] == 2
    assert tools["n"] == 2 and tools["errors"] == 1 and tools["tool_calls"] == 2
    assert tools["correct"] == 1.5 and tools["grounded"] == 0
    assert summary["kappa"]["correct"]["n"] == 2
    text = chat_eval.render_summary(summary)
    assert "all/vector" in text and "Cohen's kappa" in text


def test_cohen_kappa():
    assert chat_eval.cohen_kappa(["1", "1", "0", "0"], ["1", "1", "0", "0"]) == 1.0
    assert abs(chat_eval.cohen_kappa(["1", "1", "0", "0"], ["1", "0", "1", "0"])) < 1e-9
    assert chat_eval.cohen_kappa([], []) is None


def test_judge_parsing():
    assert chat_eval.parse_judge('xin chào {"supported": "partial", "reason": "một phần"} hết')["supported"] == "partial"
    assert chat_eval.parse_judge("không có json")["supported"] == "error"
    assert chat_eval.parse_judge('{"supported": "maybe"}')["supported"] == "error"


def test_sheet_csv_roundtrip(tmp_path):
    sheet, _ = chat_eval.build_sheet(
        [{"id": "q", "type": "lookup", "question": "hỏi", "reference": "r", "answerable": True, "mode": "vector", "status": "OK", "answer": "a", "citations": []}],
        seed=1,
    )
    path = tmp_path / "s.csv"
    chat_eval.write_csv(path, sheet)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        read = list(csv.DictReader(handle))
    assert read[0]["question"] == "hỏi" and read[0]["correct"] == ""


def test_format_citations_handles_outline_citations():
    outline = {"index": 2, "chunk_id": "outline:all", "document_id": "", "heading_path": ["Đề cương môn học"],
               "page_number": None, "quote": "Course X\n- Chapter 1: A", "kind": "outline"}
    assert chat_eval.format_citations([outline]) == "[2] Đề cương môn học: Course X - Chapter 1: A"
