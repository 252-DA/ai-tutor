from functools import lru_cache

from ai_runtime import AIRuntime, ModelConfig, build_model_client
from ai_runtime.mcp import LearningContextClient, MCPToolClient

from ai_tutor.adapters.ai_runtime_model import AIRuntimeTutorModel
from ai_tutor.adapters.learning_context import MCPLearningContext
from ai_tutor.config import Settings, get_settings
from ai_tutor.service import TutorService


def build_tutor_service(settings: Settings) -> TutorService:
    model = build_model_client(
        ModelConfig(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        )
    )
    context_client = LearningContextClient(
        MCPToolClient(
            settings.learning_context.url,
            timeout_seconds=settings.learning_context.timeout_seconds,
        )
    )
    return TutorService(
        model=AIRuntimeTutorModel(AIRuntime(model)),
        context=MCPLearningContext(context_client),
    )


@lru_cache
def get_tutor_service() -> TutorService:
    return build_tutor_service(get_settings())
