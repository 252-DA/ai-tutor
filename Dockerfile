# syntax=docker/dockerfile:1
# Build từ thư mục gốc meta-repo: docker compose build ai-tutor
FROM ghcr.io/astral-sh/uv:0.11.32 AS uv
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=uv /uv /uvx /bin/

# ai-runtime-sdk là phụ thuộc đường dẫn của ai-tutor (../ai-sdk).
COPY ai-sdk/pyproject.toml ai-sdk/uv.lock* ai-sdk/README.md /app/ai-sdk/
COPY ai-sdk/src/ /app/ai-sdk/src/
COPY ai-tutor/pyproject.toml ai-tutor/uv.lock* ai-tutor/README.md /app/ai-tutor/
COPY ai-tutor/src/ /app/ai-tutor/src/

WORKDIR /app/ai-tutor
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.10-slim

COPY --from=builder /app/ai-tutor/.venv /app/.venv

# Model chọn bằng LLM_PROFILE__TUTOR (tên profile trong catalog của ai-sdk).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8010

# python -m thay vì console-script: venv không relocatable giữa hai stage build.
CMD ["python", "-m", "uvicorn", "ai_tutor.api:app", "--host", "0.0.0.0", "--port", "8010"]
