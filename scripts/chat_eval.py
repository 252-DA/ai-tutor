#!/usr/bin/env python3
"""
Chạy bộ câu hỏi qua ba chế độ truy xuất của chatbot và lấy số liệu báo cáo.
Xem docs/chatbot-ba-che-do-ke-hoach.md, mục 9.

    run      hỏi tutor từng câu × từng chế độ, ghi JSONL (chỉ cần thư viện chuẩn)
    sheet    xuất CSV cho người chấm mù (ẩn chế độ) + file khóa
    judge    (tùy chọn) mô hình khác chấm sơ bộ độ bám nguồn
    summary  gộp số đo theo chế độ và kiểu câu hỏi; độ đồng thuận giữa hai người chấm
    import   (tùy chọn) nạp JSONL vào bảng chat_runs với source='eval'

Các lượt của một câu hỏi chạy liền nhau theo thứ tự đổi vòng giữa các chế độ, để
trôi dạt về thời gian/hạn mức của nhà cung cấp mô hình không dồn vào một chế độ.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

MODES = ("vector", "tools", "graph")
HERE = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = HERE / "eval" / "questions.seed.jsonl"


# ---------------------------------------------------------------- io


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- run


def ask_tutor(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/tutor/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_one(question: dict[str, Any], mode: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    row: dict[str, Any] = {
        "id": question["id"],
        "type": question["type"],
        "question": question["question"],
        "reference": question.get("reference", ""),
        "answerable": question.get("answerable", True),
        "evidence": question.get("evidence", []),
        "mode": mode,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        reply = ask_tutor(
            args.tutor_url,
            {"course_id": args.course, "message": question["question"], "mode": mode},
            args.timeout,
        )
        row.update(
            status="OK",
            answer=reply.get("answer", ""),
            model=reply.get("model"),
            citations=reply.get("citations", []),
            trace=reply.get("trace", {}),
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        row.update(status="ERROR", error=str(error)[:500], answer=None, citations=[], trace={})
    row["wall_ms"] = int((time.monotonic() - started) * 1000)
    return row


def cmd_run(args: argparse.Namespace) -> int:
    questions = read_jsonl(Path(args.questions))
    if args.types:
        wanted = set(args.types.split(","))
        questions = [q for q in questions if q["type"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    modes = [m for m in args.modes.split(",") if m]
    bad = [m for m in modes if m not in MODES]
    if bad:
        print(f"Chế độ không hợp lệ: {bad}", file=sys.stderr)
        return 2

    out = Path(args.out or HERE / "eval" / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    done: set[tuple[str, str]] = set()
    if args.resume and out.exists():
        done = {(r["id"], r["mode"]) for r in read_jsonl(out) if r.get("status") == "OK"}
    total = len(questions) * len(modes)
    count = 0
    for question in questions:
        for mode in modes:
            count += 1
            if (question["id"], mode) in done:
                continue
            row = run_one(question, mode, args)
            append_jsonl(out, row)
            steps = row.get("trace", {}).get("steps", "-")
            print(f"[{count}/{total}] {question['id']} {mode:6s} {row['status']} steps={steps} {row['wall_ms']}ms")
            if args.sleep:
                time.sleep(args.sleep)
    print(f"Đã ghi {out}")
    return 0


# ---------------------------------------------------------------- sheet

RATING_COLUMNS = ["correct", "grounded", "citation_ok", "note"]


def format_citations(citations: Iterable[dict[str, Any]]) -> str:
    parts = []
    for c in citations:
        where = " > ".join(c.get("heading_path") or []) or "?"
        page = f" tr.{c['page_number']}" if c.get("page_number") else ""
        quote = (c.get("quote") or "").replace("\n", " ")[:300]
        parts.append(f"[{c.get('index')}] {where}{page}: {quote}")
    return "\n".join(parts)


def build_sheet(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Mỗi lượt OK một dòng ẩn chế độ; file khóa nối `item` về (câu hỏi, chế độ)."""
    ok = [r for r in rows if r.get("status") == "OK"]
    order = list(range(len(ok)))
    random.Random(seed).shuffle(order)
    sheet, key = [], []
    for n, index in enumerate(order, start=1):
        r = ok[index]
        item = f"R{n:04d}"
        sheet.append(
            {
                "item": item,
                "question": r["question"],
                "reference": r.get("reference", ""),
                "answerable": "yes" if r.get("answerable", True) else "no (đúng là phải từ chối)",
                "answer": r.get("answer") or "",
                "citations": format_citations(r.get("citations", [])),
                **{col: "" for col in RATING_COLUMNS},
            }
        )
        key.append({"item": item, "id": r["id"], "type": r["type"], "mode": r["mode"]})
    return sheet, key


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["item"])
        writer.writeheader()
        writer.writerows(rows)


def cmd_sheet(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.runs))
    sheet, key = build_sheet(rows, args.seed)
    out = Path(args.out)
    write_csv(out, sheet)
    key_path = out.with_name(out.stem + ".key.csv")
    write_csv(key_path, key)
    print(f"{len(sheet)} dòng → {out}\nKhóa (đừng gửi người chấm) → {key_path}")
    return 0


# ---------------------------------------------------------------- judge

JUDGE_SYSTEM = (
    "Bạn là người kiểm tra độ bám nguồn của câu trả lời. Chỉ dùng các đoạn trích được đưa ra. "
    "Không dùng kiến thức bên ngoài. Chỉ trả về một JSON."
)


def judge_prompt(row: dict[str, Any]) -> str:
    return (
        f"Câu hỏi: {row['question']}\n\nCâu trả lời cần kiểm tra:\n{row.get('answer')}\n\n"
        f"Các đoạn trích mà câu trả lời viện dẫn:\n{format_citations(row.get('citations', [])) or '(không có)'}\n\n"
        'Mọi khẳng định thực tế trong câu trả lời có được các đoạn trích hỗ trợ không? '
        'Trả về {"supported": "yes" | "partial" | "no", "reason": "một câu"}. '
        '"yes": mọi khẳng định có căn cứ; "partial": một phần có căn cứ; "no": phần lớn không có căn cứ. '
        "Nếu câu trả lời chỉ nói rằng tài liệu không có thông tin và không khẳng định gì khác, trả về yes."
    )


def parse_judge(text: str) -> dict[str, str]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {"supported": "error", "reason": "không phải JSON"}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"supported": "error", "reason": "JSON hỏng"}
    value = str(data.get("supported", "")).lower()
    if value not in ("yes", "partial", "no"):
        return {"supported": "error", "reason": f"giá trị lạ: {value}"}
    return {"supported": value, "reason": str(data.get("reason", ""))[:300]}


def cmd_judge(args: argparse.Namespace) -> int:
    import os

    if args.profile:
        os.environ["LLM_PROFILE__QUIZ_VERIFICATION"] = args.profile
    sys.path.insert(0, str(HERE / "src"))
    from ai_runtime.catalog import QUIZ_VERIFICATION  # noqa: PLC0415

    from ai_tutor.adapters.ai_runtime_model import AIRuntimeTutorModel  # noqa: PLC0415
    from ai_tutor.config import get_settings  # noqa: PLC0415
    from ai_tutor.container import build_model_registry  # noqa: PLC0415

    resolved = build_model_registry(get_settings()).resolve(QUIZ_VERIFICATION)
    judge = AIRuntimeTutorModel(resolved.runtime())
    rows = read_jsonl(Path(args.runs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row.get("status") == "OK" and row.get("answerable", True):
                try:
                    row["judge"] = {**parse_judge(judge.generate(judge_prompt(row), JUDGE_SYSTEM).text), "model": judge.model_id}
                except Exception as error:  # noqa: BLE001 - một lượt hỏng không được dừng cả đợt chấm
                    row["judge"] = {"supported": "error", "reason": str(error)[:200], "model": judge.model_id}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Đã ghi {out}")
    return 0


# ---------------------------------------------------------------- summary


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def read_ratings(sheet_path: Path) -> dict[str, dict[str, str]]:
    with sheet_path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["item"]: row for row in csv.DictReader(handle)}


def to_number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def summarize(
    rows: list[dict[str, Any]],
    key: dict[str, dict[str, str]] | None = None,
    rater_sheets: list[dict[str, dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Gộp theo (chế độ) và (chế độ, kiểu). Điểm người chấm được tính trung bình giữa các người chấm."""
    by_item_rating: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    if key and rater_sheets:
        for sheet in rater_sheets:
            for item, rating in sheet.items():
                meta = key.get(item)
                if not meta:
                    continue
                scores = {c: to_number(rating.get(c)) for c in ("correct", "grounded", "citation_ok")}
                by_item_rating[(meta["id"], meta["mode"])].append({k: v for k, v in scores.items() if v is not None})

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[("all", r["mode"])].append(r)
        groups[(r["type"], r["mode"])].append(r)

    result: dict[str, Any] = {"groups": {}, "kappa": {}}
    for (kind, mode), items in sorted(groups.items()):
        ok = [r for r in items if r.get("status") == "OK"]
        traces = [r.get("trace", {}) for r in ok]
        latencies = [t.get("latency_ms", r.get("wall_ms", 0)) for r, t in zip(ok, traces)]
        entry: dict[str, Any] = {
            "n": len(items),
            "errors": len(items) - len(ok),
            "steps": round(statistics.mean(t.get("steps", 0) for t in traces), 2) if ok else None,
            "tool_calls": round(statistics.mean(len(t.get("tool_calls", [])) for t in traces), 2) if ok else None,
            "input_tokens": round(statistics.mean(t.get("input_tokens", 0) for t in traces)) if ok else None,
            "output_tokens": round(statistics.mean(t.get("output_tokens", 0) for t in traces)) if ok else None,
            "latency_median_s": round(statistics.median(latencies) / 1000, 1) if ok else None,
            "latency_p90_s": round(percentile(latencies, 0.9) / 1000, 1) if ok else None,
            "cited": round(sum(bool(r.get("citations")) for r in ok if r.get("answerable", True)) / max(1, sum(r.get("answerable", True) for r in ok)), 2) if ok else None,
        }
        judged = [r["judge"]["supported"] for r in ok if r.get("judge") and r["judge"]["supported"] != "error"]
        if judged:
            entry["judge_supported"] = round(sum({"yes": 1, "partial": 0.5, "no": 0}[v] for v in judged) / len(judged), 2)
        for column in ("correct", "grounded", "citation_ok"):
            values = [
                statistics.mean(s[column] for s in by_item_rating[(r["id"], r["mode"])] if column in s)
                for r in items
                if any(column in s for s in by_item_rating.get((r["id"], r["mode"]), []))
            ]
            if values:
                entry[column] = round(statistics.mean(values), 2)
        result["groups"][f"{kind}/{mode}"] = entry

    if key and rater_sheets and len(rater_sheets) >= 2:
        for column in ("correct", "grounded"):
            a, b = [], []
            for item, rating in rater_sheets[0].items():
                other = rater_sheets[1].get(item)
                if other and rating.get(column, "") != "" and other.get(column, "") != "":
                    a.append(rating[column])
                    b.append(other[column])
            result["kappa"][column] = {"n": len(a), "kappa": cohen_kappa(a, b)}
    return result


def render_summary(summary: dict[str, Any]) -> str:
    columns = [
        ("n", "n"),
        ("errors", "lỗi"),
        ("steps", "bước"),
        ("tool_calls", "gọi CC"),
        ("input_tokens", "tok vào"),
        ("output_tokens", "tok ra"),
        ("latency_median_s", "trễ giữa (s)"),
        ("latency_p90_s", "trễ p90 (s)"),
        ("cited", "có trích dẫn"),
        ("judge_supported", "judge"),
        ("correct", "đúng"),
        ("grounded", "bám nguồn"),
        ("citation_ok", "trích đúng"),
    ]
    present = [(k, label) for k, label in columns if any(k in e and e[k] is not None for e in summary["groups"].values())]
    lines = ["| nhóm | " + " | ".join(label for _, label in present) + " |", "|---|" + "---|" * len(present)]
    for name, entry in summary["groups"].items():
        cells = ["" if entry.get(k) is None else str(entry.get(k)) for k, _ in present]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for column, info in summary.get("kappa", {}).items():
        kappa = "n/a" if info["kappa"] is None else f"{info['kappa']:.2f}"
        lines.append(f"\nCohen's kappa ({column}): {kappa} trên {info['n']} mục")
    return "\n".join(lines)


def cmd_summary(args: argparse.Namespace) -> int:
    rows = read_jsonl(Path(args.runs))
    key = None
    sheets = None
    if args.ratings:
        if not args.key:
            print("--ratings cần --key", file=sys.stderr)
            return 2
        key = {r["item"]: r for r in csv.DictReader(Path(args.key).open(encoding="utf-8-sig", newline=""))}
        sheets = [read_ratings(Path(p)) for p in args.ratings]
    summary = summarize(rows, key, sheets)
    print(render_summary(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- import


def cmd_import(args: argparse.Namespace) -> int:
    try:
        import psycopg  # noqa: PLC0415
    except ImportError:
        print("Cần cài psycopg (pip install 'psycopg[binary]') để nạp vào chat_runs.", file=sys.stderr)
        return 2
    rows = read_jsonl(Path(args.runs))
    with psycopg.connect(args.db_url) as conn, conn.cursor() as cur:
        for r in rows:
            trace = r.get("trace", {})
            cur.execute(
                """INSERT INTO chat_runs (course_id, mode, source, question_type, question, answer, citations,
                       tool_calls, model, steps, input_tokens, output_tokens, latency_ms, status, error)
                   VALUES (%s, %s, 'eval', %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    args.course_uuid,
                    r["mode"],
                    r["type"],
                    r["question"],
                    r.get("answer"),
                    json.dumps(r.get("citations", []), ensure_ascii=False),
                    json.dumps(trace.get("tool_calls", []), ensure_ascii=False),
                    r.get("model"),
                    trace.get("steps", 0),
                    trace.get("input_tokens"),
                    trace.get("output_tokens"),
                    trace.get("latency_ms", r.get("wall_ms")),
                    r.get("status", "OK"),
                    r.get("error"),
                ),
            )
    print(f"Đã nạp {len(rows)} dòng vào chat_runs (source='eval').")
    return 0


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="chạy bộ câu hỏi qua các chế độ")
    run.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    run.add_argument("--tutor-url", default="http://localhost:8010")
    run.add_argument("--course", default="CO3137", help="UUID hoặc mã môn")
    run.add_argument("--modes", default=",".join(MODES))
    run.add_argument("--types", help="chỉ chạy các kiểu này, cách nhau bằng dấu phẩy")
    run.add_argument("--limit", type=int, help="chỉ chạy N câu đầu (chạy thử)")
    run.add_argument("--out")
    run.add_argument("--resume", action="store_true", help="bỏ qua cặp (câu, chế độ) đã có lượt OK trong --out")
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--sleep", type=float, default=0, help="nghỉ giữa các lượt (giây), để tránh hạn mức")
    run.set_defaults(func=cmd_run)

    sheet = sub.add_parser("sheet", help="xuất CSV cho người chấm mù")
    sheet.add_argument("--runs", required=True)
    sheet.add_argument("--out", required=True)
    sheet.add_argument("--seed", type=int, default=7)
    sheet.set_defaults(func=cmd_sheet)

    judge = sub.add_parser("judge", help="chấm sơ bộ độ bám nguồn bằng mô hình khác")
    judge.add_argument("--runs", required=True)
    judge.add_argument("--out", required=True)
    judge.add_argument("--profile", help="profile mô hình chấm (mặc định theo LLM_PROFILE__QUIZ_VERIFICATION)")
    judge.set_defaults(func=cmd_judge)

    summary = sub.add_parser("summary", help="tổng hợp số đo")
    summary.add_argument("--runs", required=True)
    summary.add_argument("--key", help="file .key.csv từ lệnh sheet")
    summary.add_argument("--ratings", nargs="+", help="một hoặc hai CSV đã chấm")
    summary.add_argument("--json")
    summary.set_defaults(func=cmd_summary)

    imp = sub.add_parser("import", help="nạp JSONL vào chat_runs")
    imp.add_argument("--runs", required=True)
    imp.add_argument("--db-url", required=True)
    imp.add_argument("--course-uuid", required=True)
    imp.set_defaults(func=cmd_import)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
