class Model:
    """Protocol for language models."""

    model: str
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]]) -> dict:
        pass

    def get_usage_stats(self) -> dict[str, any]:
        pass


def get_model(model: str) -> Model:
    # Across all providers, **extended thinking is ON by default**:
    #   - plain `<model>`             → thinking on
    #   - `<model>-no-thinking`       → opt out (cheap / fast variant)
    #   - `<model>-thinking`          → explicit-on alias (redundant)
    if model == "passive":
        # Null-provider baseline: always emits no tool call → env synthesises
        # `wait_for_next_day`. $0 API cost. Use as `--models 'roaster_A:passive'`.
        from coffeebench.models.passive_model import PassiveModel

        return PassiveModel()
    if model == "heuristic_roaster":
        # Scripted state-machine baseline for the roaster role. $0 API cost.
        # Use as `--models 'roaster_A:heuristic_roaster'`.
        from coffeebench.models.heuristic_roaster_model import HeuristicRoasterModel

        return HeuristicRoasterModel()
    if model.startswith("gpt-"):
        from coffeebench.models.openai_model import OpenAIModel

        if model.endswith("-no-thinking"):
            base = model[: -len("-no-thinking")]
            return OpenAIModel(model=base, enable_thinking=False)
        if model.endswith("-thinking"):
            base = model[: -len("-thinking")]
            return OpenAIModel(model=base, enable_thinking=True)
        return OpenAIModel(model=model)
    elif model.startswith("claude-"):
        from coffeebench.models.anthropic_model import AnthropicModel

        if model.endswith("-no-thinking"):
            base = model[: -len("-no-thinking")]
            return AnthropicModel(model=base, enable_thinking=False)
        if model.endswith("-thinking"):
            base = model[: -len("-thinking")]
            return AnthropicModel(model=base, enable_thinking=True)
        return AnthropicModel(model=model)
    elif model.startswith("gemini-"):
        from coffeebench.models.gemini_model import GeminiModel

        if model.endswith("-no-thinking"):
            base = model[: -len("-no-thinking")]
            return GeminiModel(model=base, enable_thinking=False)
        if model.endswith("-thinking"):
            base = model[: -len("-thinking")]
            return GeminiModel(model=base, enable_thinking=True)
        return GeminiModel(model=model)
    elif "/" in model:
        # OpenRouter slugs are <org>/<model>, e.g. moonshotai/kimi-k2.6.
        from coffeebench.models.openrouter_model import OpenRouterModel

        if model.endswith("-no-thinking"):
            base = model[: -len("-no-thinking")]
            return OpenRouterModel(model=base, enable_thinking=False)
        if model.endswith("-thinking"):
            base = model[: -len("-thinking")]
            return OpenRouterModel(model=base, enable_thinking=True)
        return OpenRouterModel(model=model)
    else:
        raise ValueError(f"Unsupported model: {model}")


if __name__ == "__main__":
    model = get_model("gpt-5.5")
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = model.query(messages)
    print(response["content"])
    print(model.get_usage_stats())
