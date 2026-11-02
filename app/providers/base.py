from typing import Protocol


class LLMProvider(Protocol):
    name: str
    model: str
    def generate(self, question: str, context: str, system_prompt: str = "") -> dict: ...
