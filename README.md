# AI Tutor Service

Minimal, stateless course-grounded tutor API. Tutor orchestration lives here;
model-provider and MCP transport are delegated to `ai-runtime-sdk`.

## Run

```bash
uv run uvicorn ai_tutor.api:app --reload --port 8010
```

Configuration:

```bash
LLM__PROVIDER=deepseek
LLM__MODEL=deepseek-chat
LLM__API_KEY=...
LEARNING_CONTEXT_MCP_URL=http://localhost:8001/mcp
```

Ask the tutor:

```bash
curl http://localhost:8010/v1/tutor/messages \
  -H 'content-type: application/json' \
  -d '{
    "course_id": "CO3115",
    "lo_code": "L.O.3.1",
    "message": "Giải thích khái niệm này bằng một ví dụ"
  }'
```

The first version intentionally does not persist conversations. A caller may
send recent messages in `history`; persistence and learner personalization can
be added without changing the model-provider layer.
