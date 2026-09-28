import os
from functools import lru_cache

from ai_runtime import EnvConfigSource, ModelProfile, ModelRegistry
from ai_runtime.catalog import PROFILES, TUTOR
from ai_runtime.mcp import LearningContextClient, MCPToolClient

from ai_tutor.adapters.ai_runtime_model import AIRuntimeTutorModel
from ai_tutor.adapters.learning_context import MCPLearningContext
from ai_tutor.config import Settings, get_settings
from ai_tutor.service import TutorService


_LEGACY_PROFILE_NAME = "env-llm"
_LEGACY_ENV_VARS = ("LLM__PROVIDER", "LLM__MODEL", "LLM__BASE_URL")


def build_model_registry(settings: Settings) -> ModelRegistry:
    """
    LLM_PROFILE__TUTOR chọn model; LLM__* cũ vẫn thắng nếu còn được đặt.

    Giữ cùng thứ tự với worker (worker/container.py) để một biến có nghĩa như
    nhau ở mọi service.
    """
    legacy = ModelProfile(
        name=_LEGACY_PROFILE_NAME,
        provider=settings.llm.provider,
        model=settings.llm.model,
        key_env=("LLM__API_KEY",),
        base_url=settings.llm.base_url or None,
        temperature=settings.llm.temperature,
        timeout_seconds=settings.llm.timeout_seconds,
        max_retries=settings.llm.max_retries,
        internal=True,
    )
    env = dict(os.environ)
    if settings.llm.api_key:
        env[legacy.key_env[0]] = settings.llm.api_key
    prefer_legacy = any(os.environ.get(name, "").strip() for name in _LEGACY_ENV_VARS)

    return ModelRegistry(
        source=EnvConfigSource(
            env=env,
            fallback_profile=legacy.name if prefer_legacy else None,
        ),
        profiles={**PROFILES, legacy.name: legacy},
        env=env,
    )


def build_tutor_service(settings: Settings) -> TutorService:
    resolved = build_model_registry(settings).resolve(TUTOR)
    context_client = LearningContextClient(
        MCPToolClient(
            settings.learning_context.url,
            timeout_seconds=settings.learning_context.timeout_seconds,
        )
    )
    return TutorService(
        model=AIRuntimeTutorModel(resolved.runtime()),
        context=MCPLearningContext(context_client),
    )


@lru_cache
def get_tutor_service() -> TutorService:
    return build_tutor_service(get_settings())
