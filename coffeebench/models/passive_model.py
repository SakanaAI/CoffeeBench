"""PassiveModel — null-provider baseline that always emits no tool call."""

from __future__ import annotations

from typing import Any

from coffeebench.models.types import ModelResponse


class PassiveModel:
    """Conforms to the `Model` protocol but never calls a real API."""

    DEFAULT_MAX_INPUT_TOKENS = 1_000_000

    def __init__(self, model: str = "passive"):
        self.model = model
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        # Held for parity with Anthropic / Gemini wrappers that store the
        # system prompt out-of-band; harmless on this path.
        self.system_prompt = ""

    def query(self, messages, tools=None) -> ModelResponse:  # noqa: ARG002
        """Always return an empty response — no tool_calls means
        `Agent.step_apply` synthesises `wait_for_next_day`."""
        self.n_calls += 1
        return ModelResponse(
            content="(passive baseline: sleeping until next event)",
            thinking="",
            tool_calls=[],
            stop_reason="end_turn",
            cost=0.0,
            raw=None,
        )

    def get_usage_stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_calls": self.n_calls,
            "cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "last_input_tokens": 0,
        }

    def summarize(self, instructions: str, content: str, max_tokens: int = 4096) -> str:  # noqa: ARG002
        # ContextCompactor calls this when transcripts grow long. Passive
        # transcripts barely grow (no tool results, no thinking), so this
        # path should never fire — but if it does, return the content
        # unchanged so nothing breaks.
        return content
