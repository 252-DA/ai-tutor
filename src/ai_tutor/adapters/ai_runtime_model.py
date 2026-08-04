from ai_runtime import AIRuntime


class AIRuntimeTutorModel:
    def __init__(self, runtime: AIRuntime) -> None:
        self._runtime = runtime

    @property
    def model_id(self) -> str:
        return self._runtime.model_id

    def answer(self, prompt: str, system: str) -> str:
        return self._runtime.generate_text(prompt=prompt, system=system).text
