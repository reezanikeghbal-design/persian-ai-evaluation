import time


class MockProvider:
    name = "mock"
    model = "mock-context-extractive-v1"

    def generate(self, question: str, context: str, system_prompt: str = "") -> dict:
        started = time.perf_counter()
        answer = context[:700].strip() if context.strip() else "اطلاعات کافی در اسناد موجود نیست."
        return {"answer": answer, "model": self.model, "input_tokens": None,
                "output_tokens": None, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
