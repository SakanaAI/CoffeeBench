class Model:
    """Protocol for language models."""

    model: str
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]]) -> dict:
        pass

    def get_usage_stats(self) -> dict[str, any]:
        pass


# Scripted, zero-external-cost models. These never call a provider API, need no
# keys, and consume no provider tokens — their `query()` is a local, in-process
# invocation. Kept as a named set so the harness can (a) skip the ContextCompactor
# for them and (b) separate LOCAL scripted invocations from EXTERNAL API calls in
# usage reporting. See `coffeebench/models/scripted.py`.
SCRIPTED_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "passive",
        "heuristic_farmer",
        "heuristic_roaster",
        "heuristic_retailer",
    }
)


def is_scripted_model(model: str) -> bool:
    """True for models that run locally with zero external API cost."""
    return model in SCRIPTED_MODEL_NAMES


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
    if model in ("heuristic_farmer", "heuristic_roaster", "heuristic_retailer"):
        # Deterministic scripted role policies for the zero-LLM active economy.
        # Both agents of a role share the same policy class. $0 API cost.
        from coffeebench.models.scripted import (
            HeuristicFarmerModel,
            HeuristicRetailerModel,
            HeuristicRoasterModel,
        )

        return {
            "heuristic_farmer": HeuristicFarmerModel,
            "heuristic_roaster": HeuristicRoasterModel,
            "heuristic_retailer": HeuristicRetailerModel,
        }[model]()
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
