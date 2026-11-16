from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    provider: Literal["gemini", "openai-compatible", "deepseek"] = "gemini"
    model: str = "gemini-3.1-flash-lite"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    timeout_seconds: float = 60
    max_retries: int = 2

    model_config = SettingsConfigDict(env_prefix="LLM__")


class LearningContextSettings(BaseSettings):
    url: str = "http://localhost:8001/mcp"
    timeout_seconds: float = 30

    model_config = SettingsConfigDict(env_prefix="LEARNING_CONTEXT_MCP_")


class TutorSettings(BaseSettings):
    # Số lời gọi công cụ tối đa mỗi câu hỏi. Chế độ graph có thêm một công cụ nên được thêm một lượt.
    max_tool_calls_tools: int = Field(default=5, ge=1, le=10)
    max_tool_calls_graph: int = Field(default=6, ge=1, le=10)

    model_config = SettingsConfigDict(env_prefix="TUTOR_")


class Settings(BaseSettings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    learning_context: LearningContextSettings = Field(
        default_factory=LearningContextSettings
    )
    tutor: TutorSettings = Field(default_factory=TutorSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
