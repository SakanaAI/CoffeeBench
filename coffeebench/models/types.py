"""Provider-agnostic types for the native tool-use harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSpec:
    """Provider-agnostic tool schema.

    Each provider wrapper renders this into its native tool spec
    (Anthropic: name/description/input_schema; OpenAI: function /
    type=function with parameters; Chat Completions: function block).
    """

    name: str
    description: str
    input_schema: dict  # JSON Schema for the tool's parameters

    @classmethod
    def from_function(cls, func) -> "ToolSpec":
        """Build a ToolSpec from a Python callable's signature + docstring."""
        from coffeebench.util import (
            _function_parameters_schema,
        )  # local import to avoid cycle

        schema = _function_parameters_schema(func)
        return cls(
            name=func.__name__,
            description=(func.__doc__ or "").strip(),
            input_schema=schema,
        )


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str  # provider-specific id; round-tripped to identify the matching tool_result
    name: str  # tool name (must match a ToolSpec.name registered for this turn)
    input: dict  # parsed arguments (already JSON-decoded)


@dataclass
class ModelResponse:
    """One assistant turn returned by a provider wrapper."""

    content: str = ""  # visible text (may be empty if model only emitted tool_use)
    thinking: str = ""  # extended-thinking summary if any (for trajectory display)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""  # provider's stop reason (e.g. "tool_use", "end_turn")
    cost: float = 0.0
    raw: Any = None  # opaque, provider-specific; required for replay fidelity
