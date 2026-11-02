import os
import time


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None):
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.client = genai.Client(api_key=key)

    def generate(self, question: str, context: str, system_prompt: str = "") -> dict:
        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=f"{system_prompt}\n\nزمینه:\n{context}\n\nپرسش: {question}",
        )
        usage = getattr(response, "usage_metadata", None)
        return {"answer": response.text or "", "model": self.model,
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
