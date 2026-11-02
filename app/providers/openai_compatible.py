import os
import time
from openai import OpenAI


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, model: str | None = None):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None)

    def generate(self, question: str, context: str, system_prompt: str = "") -> dict:
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt or "فقط بر اساس متن زمینه پاسخ بده."},
                      {"role": "user", "content": f"زمینه:\n{context}\n\nپرسش: {question}"}],
        )
        usage = response.usage
        return {"answer": response.choices[0].message.content or "", "model": self.model,
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
