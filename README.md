# AI Tutor Service

Stateless, course-grounded chatbot API with three retrieval modes. Design and
rationale: `docs/chatbot-ba-che-do-ke-hoach.md`. Model-provider and MCP transport
are delegated to `ai-runtime-sdk`; retrieval tools live in `mcp-service`.

| `mode` | What it does |
|---|---|
| `vector` | One `search_course_chunks` call, then one answer call. Plain RAG, the control arm. |
| `tools` | The model calls tools in a JSON action loop (max 5): `search_course_chunks`, `get_document_page`, `list_course_documents`, `get_course_outline`, `get_neighbor_chunks`. |
| `graph` | Same as `tools` plus `get_concept_relations` (max 6 calls). Needs relations built from the admin page `/manage/chat`. |

All modes share the same model, answer rules and `[n]` citation format, so the
comparison differs only in how context is found. The mode is held by core-api
(`chat_settings`) and passed per request by the web BFF; this service has no
global switch.

## Run

```bash
uv run uvicorn ai_tutor.api:app --reload --port 8010
```

Configuration:

```bash
LLM_PROFILE__TUTOR=gemini-flash-lite       # Gemini 3.1 Flash-Lite (default)
GEMINI_API_KEY=...
LEARNING_CONTEXT_MCP_URL=http://localhost:8001/mcp
TUTOR_MAX_TOOL_CALLS_TOOLS=5
TUTOR_MAX_TOOL_CALLS_GRAPH=6
```

In compose it runs as the `ai-tutor` service on `:8010`; the web service reaches
it through `TUTOR_URL`.

## API

```bash
curl http://localhost:8010/v1/tutor/messages \
  -H 'content-type: application/json' \
  -d '{"course_id": "CO3137", "mode": "tools", "message": "Combiner là gì?"}'
```

Request: `course_id` (UUID or course code), `message`, optional `history`,
`mode` (default `vector`), `document_id`, `page_number`, `lo_code`, `top_k`.

Response: `answer`, `model`, `citations[]` (`index`, `chunk_id`, `document_id`,
`heading_path`, `page_number`, `quote`) and `trace` (`mode`, `steps`, `llm_calls`,
`retrieved`, `tool_calls[]`, token counts, `latency_ms`, `forced_answer`).

The course id is injected into every tool call by code; the model cannot pick
another course. Conversations are not persisted here: the web BFF writes every
turn to `chat_runs`.

## Evaluation

`scripts/chat_eval.py` runs a labelled question set through all three modes and
produces the numbers for the report. It needs only the standard library, except
`judge` (uses the tutor's model registry) and `import` (needs `psycopg`).

```bash
# 1. Ask every question in every mode (interleaved). --sleep avoids provider rate limits.
python scripts/chat_eval.py run --course CO3137 --tutor-url http://localhost:8010 \
    --out eval/runs/2026-10-xx.jsonl --sleep 2 --resume

# 2. Blind sheet for two raters: mode hidden, order shuffled. Keep the .key.csv private.
python scripts/chat_eval.py sheet --runs eval/runs/2026-10-xx.jsonl --out eval/sheets/round1.csv

# 3. Optional: a different model screens groundedness (not a replacement for raters).
python scripts/chat_eval.py judge --runs eval/runs/2026-10-xx.jsonl \
    --out eval/runs/2026-10-xx.judged.jsonl --profile gemini-flash

# 4. Per-mode and per-type table; give two rated sheets to get Cohen's kappa.
python scripts/chat_eval.py summary --runs eval/runs/2026-10-xx.jsonl \
    --key eval/sheets/round1.key.csv --ratings rater_a.csv rater_b.csv

# 5. Optional: load the runs into chat_runs (source='eval') to see them on /manage/chat.
python scripts/chat_eval.py import --runs eval/runs/2026-10-xx.jsonl \
    --db-url postgresql://... --course-uuid <course uuid>
```

Rater columns: `correct` (0 wrong, 1 partly, 2 right; for unanswerable questions 2
means it refused properly), `grounded` (0/1/2: claims supported by the cited
passages), `citation_ok` (0/1: cited page/passage really holds the claim), `note`.

`eval/questions.seed.jsonl` holds 37 starter questions (lookup, definition,
relational, curriculum, global, absent) written against the CO3137 documents.
The `reference` fields were drafted from the retrieved passages and must be
checked by a human before they are used as ground truth. The goal is 60-100
questions with at least 10 per type. Each question can make multiple model
calls, especially in tools/graph mode. Check the project's active model limits
in Google AI Studio before a full run; use `--resume` to continue across runs.

## Tests

```bash
uv run pytest
```
