class Model:
    """Protocol for language models."""

    model: str
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]]) -> dict:
        pass

    def get_usage_stats(self) -> dict[str, any]:
        pass


VALID_EFFORTS = frozenset(
    {"off", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def get_model(model: str) -> Model:
    # Reasoning is controlled by a REQUIRED trailing `:<effort>` suffix on every
    # reasoning-capable model string (e.g. `gpt-5.5:high`, `claude-opus-4-8:xhigh`,
    # `z-ai/glm-5.2:max`, `claude-sonnet-4-6:off`). It is mandatory so a run never
    # silently inherits a provider default — those differ across providers
    # (Anthropic defaults thinking off, Gemini 3 defaults to high, …), which
    # would make cross-model comparison invalid. Valid values: off, minimal,
    # low, medium, high, xhigh, max — `off` means "no reasoning" (mapped to the
    # provider's floor where it can't be fully disabled). Each provider maps the
    # level onto its native control: OpenAI `reasoning.effort`, Anthropic
    # `output_config.effort` (adaptive thinking), Gemini `thinking_level`,
    # OpenRouter `reasoning_effort`. Temperature is always left at the provider
    # default (never sent).
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

    # Split off the required ":<effort>" suffix shared by every provider.
    base, effort = model.rsplit(":", 1) if ":" in model else (model, None)
    if effort is None:
        raise ValueError(
            f"model '{model}' is missing a required reasoning-effort suffix; "
            f"append one of {sorted(VALID_EFFORTS)} "
            f"(e.g. '{model}:high' or '{model}:off')"
        )
    if effort not in VALID_EFFORTS:
        raise ValueError(
            f"invalid reasoning effort '{effort}' in '{model}'; "
            f"valid values: {sorted(VALID_EFFORTS)}"
        )

    if base.startswith("gpt-"):
        from coffeebench.models.openai_model import OpenAIModel

        return OpenAIModel(model=base, effort=effort)
    elif base.startswith("claude-"):
        from coffeebench.models.anthropic_model import AnthropicModel

        return AnthropicModel(model=base, effort=effort)
    elif base.startswith("gemini-"):
        from coffeebench.models.gemini_model import GeminiModel

        return GeminiModel(model=base, effort=effort)
    elif "/" in base:
        # OpenRouter slugs are <org>/<model>, e.g. moonshotai/kimi-k2.6.
        # An optional `@<provider>` suffix pins routing to a single upstream
        # provider (e.g. `z-ai/glm-5.2@z-ai`); otherwise the official provider
        # is auto-pinned (see OpenRouterModel).
        from coffeebench.models.openrouter_model import OpenRouterModel

        provider = None
        slug = base
        if "@" in slug:
            slug, provider = slug.split("@", 1)
        return OpenRouterModel(model=slug, effort=effort, provider=provider)
    else:
        raise ValueError(f"Unsupported model: {model}")


if __name__ == "__main__":
    model = get_model("gpt-5.5:high")
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    response = model.query(messages)
    print(response["content"])
    print(model.get_usage_stats())
