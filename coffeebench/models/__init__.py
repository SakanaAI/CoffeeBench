class Model:
    """Protocol for language models."""

    model: str
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]]) -> dict:
        pass

    def get_usage_stats(self) -> dict[str, any]:
        pass


# $0 scripted baselines (no API calls; never compacted; checkpointable).
SCRIPTED_MODELS = (
    "passive",
    "heuristic_roaster",
    "heuristic_roaster_v2",
    "heuristic_farmer",
    "heuristic_retailer",
)

def is_scripted(model: str) -> bool:
    """True for a $0 scripted policy, optionally with a per-seat policy
    profile: `heuristic_farmer@adversarial` (see the model modules' PROFILES)."""
    return str(model).partition("@")[0] in SCRIPTED_MODELS


# Rule background market (`launch_sim.py --bg Rule`): role → scripted model
# that runs every non-focal agent of that role in-process.
RULE_BG_MODELS = {
    "farmer": "heuristic_farmer",
    "roaster": "heuristic_roaster_v2",
    "retailer": "heuristic_retailer",
}

VALID_EFFORTS = frozenset(
    {"off", "default", "minimal", "low", "medium", "high", "xhigh", "max"}
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
    # OpenRouter `reasoning_effort`. `default` sends no reasoning/thinking
    # control at all, so each provider runs at its own API default (Anthropic:
    # thinking off; OpenAI: effort medium; Gemini 3: thinking_level high;
    # OpenRouter: the model's default, reasoning on for Kimi/GLM/Qwen).
    # Temperature is always left at the provider default (never sent).
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
    if model == "heuristic_roaster_v2":
        # Stronger rule-based roaster baseline (lot costs, adaptive pricing,
        # festival ramp, both green grades). $0 API cost.
        from coffeebench.models.heuristic_roaster_v2_model import HeuristicRoasterV2Model

        return HeuristicRoasterV2Model()
    base_model, _, profile = model.partition("@")
    if base_model == "heuristic_farmer":
        # Rule-based background farmer (adaptive listing price, bounded
        # concessions, production toward a sales-based target). $0.
        # `heuristic_farmer@<profile>` selects a policy profile (PROFILES).
        from coffeebench.models.heuristic_farmer_model import PROFILES, HeuristicFarmerModel

        if profile and profile not in PROFILES:
            raise ValueError(f"unknown heuristic_farmer profile {profile!r}; valid: {sorted(PROFILES)}")
        return HeuristicFarmerModel(model=model, params=PROFILES.get(profile or "default"))
    if base_model == "heuristic_retailer":
        # Rule-based background retailer (buys cheapest roasted listings toward
        # a sales-based target stock, adaptive retail prices). $0.
        from coffeebench.models.heuristic_retailer_model import PROFILES, HeuristicRetailerModel

        if profile and profile not in PROFILES:
            raise ValueError(f"unknown heuristic_retailer profile {profile!r}; valid: {sorted(PROFILES)}")
        return HeuristicRetailerModel(model=model, params=PROFILES.get(profile or "default"))

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

    if effort == "default":
        # Every wrapper treats effort=None as "send no reasoning control".
        effort = None

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
