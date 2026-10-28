from collections.abc import Iterator

from ai_runtime import AIRuntime, GenerationResult

from ai_tutor.ports import ModelReply


def _reply(result: GenerationResult) -> ModelReply:
    usage = result.usage
    return ModelReply(
        text=result.text,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )


class AIRuntimeTutorModel:
    def __init__(self, runtime: AIRuntime) -> None:
        self._runtime = runtime

    @property
    def model_id(self) -> str:
        return self._runtime.model_id

    def generate(self, prompt: str, system: str) -> ModelReply:
        return _reply(self._runtime.generate_text(prompt=prompt, system=system))

    def stream(self, prompt: str, system: str) -> Iterator[str | ModelReply]:
        for item in self._runtime.stream_text(prompt=prompt, system=system):
            if isinstance(item, GenerationResult):
                yield _reply(item)
            else:
                yield item.text
