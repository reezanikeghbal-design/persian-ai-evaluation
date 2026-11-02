from .mock import MockProvider


def get_provider(name: str, model: str | None = None):
    if name == "mock":
        provider = MockProvider()
        if model:
            provider.model = model
        return provider
    if name in {"openai", "openai-compatible"}:
        from .openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(model)
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(model)
    raise ValueError(f"Unsupported provider: {name}")
