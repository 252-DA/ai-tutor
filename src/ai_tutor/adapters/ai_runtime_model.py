from ai_runtime import AIRuntime

from ai_tutor.ports import ModelReply


class AIRuntimeTutorModel:
    def __init__(self, runtime: AIRuntime) -> None:
        self._runtime = runtime

    @property
    def model_id(self) -> str:
        return self._runtime.model_id

    def generate(self, prompt: str, system: str) -> ModelReply:
        result = self._runtime.generate_text(prompt=prompt, system=system)
        usage = result.usage
        return ModelReply(
            text=result.text,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
        )
