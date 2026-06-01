"""ReAct-replacement agent harness using provider-native tool_use."""

import asyncio
import inspect
from datetime import datetime

from coffeebench.models import Model, get_model
from coffeebench.models.types import ModelResponse, ToolCall, ToolSpec


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class ContextOverflowError(TerminatingException):
    """Raised when the prompt has grown beyond the model's context window."""


class Agent:
    def __init__(
        self,
        model: Model,
        tools: list,
        instruct_prompt: str,
        initial_observation: str = "Observation: Day 0.",
        name: str = "wholesale",
        compactor=None,
    ):
        self.name = name
        self.tools: list = tools
        self.tools_by_name = {t.__name__: t for t in tools}
        self.tool_specs: list[ToolSpec] = [ToolSpec.from_function(t) for t in tools]
        print(
            f"[{name}] Registered {len(tools)} tools: {[t.name for t in self.tool_specs]}"
        )
        self.system_prompt: str = instruct_prompt
        self.user_prompt: str = initial_observation
        self.model = model
        self.messages: list[dict] = []
        self.compactor = compactor

    def add_message(self, role: str, content: str = "", **kwargs):
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            **kwargs,
        }
        self.messages.append(msg)

    def init(self):
        """Bootstrap the conversation. Anthropic and Gemini (native
        SDK) carry the system prompt out-of-band on
        `model.system_prompt`; OpenAI carries it as a `developer`
        message and OpenRouter as a `system` message."""
        m = self.model.model
        if m in ("passive", "heuristic_roaster"):
            # Scripted-policy models never read the system prompt; skip routing.
            pass
        elif m.startswith("claude-"):
            self.model.system_prompt = self.system_prompt
        elif m.startswith("gpt-"):
            self.add_message("developer", self.system_prompt)
        elif m.startswith("gemini-"):
            self.model.system_prompt = self.system_prompt
        elif "/" in m:
            # OpenRouter slugs (e.g. moonshotai/kimi-k2.6).
            self.add_message("system", self.system_prompt)
        else:
            raise ValueError(f"Unsupported model: {m}")
        self.add_message("user", self.user_prompt)

    def step_query(self) -> ModelResponse:
        """Phase 1 of step(): run compaction check + the LLM call. Does
        NOT mutate env state; the assistant turn is also NOT yet
        appended to history so a parallel batch of agents can run this
        phase concurrently from a thread pool. Caller must follow up
        with `step_apply(response)`.
        """
        if self.compactor is not None and self.compactor.should_compact():
            self.messages = self.compactor.compact(self.messages)
        return self._query_with_overflow_recovery()

    async def step_query_async(self) -> ModelResponse:
        """asyncio wrapper for `step_query`. Runs the (blocking) LLM
        round-trip in the default thread pool so multiple agents'
        queries can overlap on network I/O. The sync `step_query`
        method is unchanged so non-async callers still work."""
        return await asyncio.to_thread(self.step_query)

    def step_apply(self, response: ModelResponse) -> dict:
        """Phase 2 of step(): commit the assistant turn to history,
        execute the (single) tool_call, append its result. MUST be
        called serially with respect to other agents' apply phases —
        the dispatcher holds the marketplace lock around this call so
        tool effects on shared state (listings / offers / deals /
        truth ledger) land in a deterministic order.
        """
        # Persist the assistant turn — including the raw provider blocks —
        # so the next query() round-trips structured content faithfully
        # (e.g. Anthropic thinking-block signatures are required on
        # follow-up requests when the assistant turn included thinking).
        assistant_msg = {
            "role": "assistant",
            "content": response.content or "",
            "thinking": response.thinking or "",
            "tool_calls": list(response.tool_calls),
            "_raw": response.raw,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        self.messages.append(assistant_msg)

        # No tool call → model signalled it has nothing to do this turn.
        # Surface as `wait_for_next_day` so the env parks the agent.
        if not response.tool_calls:
            text = (response.content or "").strip()
            return {
                "action_name": "wait_for_next_day",
                "action_input": {},
                "observation": {
                    "status": "success",
                    "message": text or "(no action; sleeping until next event)",
                },
                "thought": response.thinking or text,
            }

        # Single-call-per-turn contract: execute only the first tool_call.
        # `disable_parallel_tool_use=True` is set on Anthropic; for other
        # providers we just ignore extras (rare in practice).
        call: ToolCall = response.tool_calls[0]
        result = self._execute_tool(call)
        # Append the tool result so the next query sees it. Each provider
        # wrapper translates {role:tool} into its native shape.
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": _stringify_tool_result(result),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        )

        return {
            "action_name": call.name,
            "action_input": call.input,
            "observation": result,
            "thought": response.thinking or response.content or "",
        }

    def step(self) -> dict:
        """Sync convenience wrapper: query → apply, in-process. Kept
        for non-async callers (smoke tests, single-agent debugging)."""
        response = self.step_query()
        return self.step_apply(response)

    # ---------- helpers ----------

    def _query_with_overflow_recovery(self) -> ModelResponse:
        try:
            return self.model.query(self.messages, tools=self.tool_specs)
        except ContextOverflowError:
            if self.compactor is None:
                raise
            print(
                f"[{self.name}] ContextOverflowError; "
                "forcing compaction and retrying once"
            )
            self.messages = self.compactor.compact(self.messages)
            return self.model.query(self.messages, tools=self.tool_specs)

    def _execute_tool(self, call: ToolCall) -> dict:
        """Resolve and invoke the tool. Tool exceptions are surfaced to
        the model as an error observation (not raised) so weaker models
        can self-correct instead of crashing the run."""
        tool = self.tools_by_name.get(call.name)
        if tool is None:
            return {
                "status": "error",
                "message": f"Unknown action '{call.name}'. Pick one of the registered tools.",
            }
        coerced = self._coerce_input(tool, call.input or {})
        try:
            return tool(**coerced)
        except Exception as e:
            return {
                "status": "error",
                "message": (
                    f"Tool call failed: {type(e).__name__}: {e}. "
                    "Check the tool's required parameters and types, then try again."
                ),
            }

    def _coerce_input(self, tool, action_input: dict) -> dict:
        """Provider tool_use already JSON-decodes args, but some routes
        (especially OpenRouter slugs and weaker models) emit numeric
        fields as strings. Coerce those here so the tool layer never
        sees `"5"` for an int param."""
        sig = inspect.signature(tool)
        coerced = {}
        for key, value in action_input.items():
            if key in sig.parameters and isinstance(value, str):
                annotation = sig.parameters[key].annotation
                target_types = getattr(annotation, "__args__", (annotation,))
                if float in target_types:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                elif int in target_types:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
            coerced[key] = value
        return coerced


def _stringify_tool_result(result) -> str:
    """Render a tool's return value as a string for the tool_result
    content. Dicts (the common case) become JSON; primitives are
    str()-formatted; lists become JSON."""
    import json as _json

    if isinstance(result, str):
        return result
    try:
        return _json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


if __name__ == "__main__":
    model = get_model("claude-sonnet-4-6")

    def check_balance(account_id: int) -> str:
        """Checks the balance of the given account ID."""
        return f"Balance for account {account_id} is $1000."

    def get_weather(location: str) -> str:
        """Gets the weather for the given location."""
        return f"{{'temp':67, 'unit':'F', 'location':'{location}'}}"

    agent = Agent(
        model=model,
        tools=[check_balance, get_weather],
        instruct_prompt="You are a helpful agent.",
        initial_observation="What's the weather in Paris?",
    )
    agent.init()
    print(agent.step())
