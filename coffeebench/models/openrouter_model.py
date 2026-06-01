"""OpenRouter wrapper — native tool-use via Chat Completions."""

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

from coffeebench.models._retry import call_with_retry
from coffeebench.models.types import ModelResponse, ToolCall, ToolSpec

load_dotenv()


class OpenRouterModel:
    DEFAULT_MAX_INPUT_TOKENS = 200_000

    def __init__(
        self, model: str = "moonshotai/kimi-k2.6", enable_thinking: bool = False
    ):
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            timeout=300.0,
        )
        self.max_tokens = 4096
        self.temperature = 0.0
        self._skip_temperature = False
        # When `enable_thinking=False` we don't send `reasoning_effort` at
        # all (consistent with the other providers' "thinking off" path).
        # Setting it to "low" / "medium" still routes the model into a
        # reasoning channel that consumes max_tokens silently. With
        # `_skip_reasoning_effort=True` the kwarg is omitted entirely.
        self._skip_reasoning_effort = not bool(enable_thinking)
        self.reasoning_effort = "high" if enable_thinking else "low"
        # Pricing per 1M tokens. OpenRouter exposes per-model rates at
        # /api/v1/models — values below match those snapshotted on
        # 2026-05-05. cached_input is the input_cache_read rate (which
        # is materially cheaper than the regular prompt rate for these
        # models, so cost-tracking depends on it).
        self.pricing = {
            "moonshotai/kimi-k2.6": {
                "input": 0.74,
                "cached_input": 0.14,
                "output": 3.49,
            },
            "z-ai/glm-5.1": {"input": 1.05, "cached_input": 0.525, "output": 3.50},
        }

    def _completion_cost(self, non_cached, cached, output) -> float:
        p = self.pricing[self.model]
        return (
            non_cached * p["input"] + cached * p["cached_input"] + output * p["output"]
        ) / 1_000_000

    @staticmethod
    def _tools_to_chatcompletions(tools: list[ToolSpec] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def _to_chat_messages(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": m.get("content", ""),
                    }
                )
                continue
            if role == "assistant":
                msg: dict = {"role": "assistant"}
                text = m.get("content") or ""
                msg["content"] = text or None
                tool_calls = m.get("tool_calls") or []
                if tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id if isinstance(tc, ToolCall) else tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc.name
                                if isinstance(tc, ToolCall)
                                else tc["name"],
                                "arguments": json.dumps(
                                    tc.input
                                    if isinstance(tc, ToolCall)
                                    else tc["input"]
                                ),
                            },
                        }
                        for tc in tool_calls
                    ]
                out.append(msg)
                continue
            out.append({"role": role, "content": m.get("content", "")})
        return out

    def query(
        self,
        messages: list[dict],
        tools: list[ToolSpec] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ModelResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": self._to_chat_messages(messages),
            "max_tokens": self.max_tokens,
        }
        if not self._skip_temperature:
            kwargs["temperature"] = self.temperature
        if not self._skip_reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        chat_tools = self._tools_to_chatcompletions(tools)
        if chat_tools:
            kwargs["tools"] = chat_tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            kwargs["parallel_tool_calls"] = False

        def _do_call():
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if "temperature" in msg and "temperature" in kwargs:
                    self._skip_temperature = True
                    kwargs.pop("temperature", None)
                    return self.client.chat.completions.create(**kwargs)
                if "reasoning" in msg and "reasoning_effort" in kwargs:
                    self._skip_reasoning_effort = True
                    kwargs.pop("reasoning_effort", None)
                    return self.client.chat.completions.create(**kwargs)
                if "parallel_tool_calls" in msg and "parallel_tool_calls" in kwargs:
                    kwargs.pop("parallel_tool_calls", None)
                    return self.client.chat.completions.create(**kwargs)
                raise
            # OpenRouter occasionally returns a malformed response with
            # `choices=None` (upstream provider returned an error body
            # the SDK parsed into a partial object). Treat as transient
            # so call_with_retry reschedules — its message contains
            # "service unavailable" which matches an existing keyword.
            if not getattr(resp, "choices", None):
                raise RuntimeError(
                    "openrouter response had no choices (service unavailable upstream)"
                )
            return resp

        response = call_with_retry(_do_call, label=f"openrouter:{self.model}")
        usage = response.usage
        prompt_tokens = int(usage.prompt_tokens or 0) if usage else 0
        completion_tokens = int(usage.completion_tokens or 0) if usage else 0
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        non_cached = prompt_tokens - cached
        cost = self._completion_cost(non_cached, cached, completion_tokens)
        self.n_calls += 1
        self.cost += cost
        self.last_input_tokens = prompt_tokens
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        print(
            f"[openrouter:{self.model}] in={prompt_tokens} cached={cached} out={completion_tokens}"
        )

        choice = response.choices[0]
        msg = choice.message
        content_text = msg.content or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = (
                    json.loads(tc.function.arguments) if tc.function.arguments else {}
                )
            except (ValueError, TypeError):
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        return ModelResponse(
            content=content_text,
            thinking="",
            tool_calls=tool_calls,
            stop_reason=getattr(choice, "finish_reason", "") or "",
            cost=cost,
            raw=None,
        )

    def get_usage_stats(self) -> dict:
        return {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "last_input_tokens": self.last_input_tokens,
        }

    def summarize(self, instructions: str, content: str, max_tokens: int = 8192) -> str:
        # Summarization is a low-creativity compression task; we deliberately
        # do NOT pass `reasoning_effort` here. With `reasoning_effort="high"`
        # on long inputs (~160K tokens) Kimi/GLM exhaust the max_tokens
        # budget on the reasoning channel and return empty `content`,
        # causing the ContextCompactor to skip compaction and the agent's
        # transcript to keep growing until provider context overflow.
        # max_tokens default bumped 4096 → 8192 for a bit more headroom.
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": content},
            ],
            "max_tokens": int(max_tokens),
        }
        if not self._skip_temperature:
            kwargs["temperature"] = self.temperature

        def _do_call():
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if "temperature" in msg and "temperature" in kwargs:
                    self._skip_temperature = True
                    kwargs.pop("temperature", None)
                    return self.client.chat.completions.create(**kwargs)
                raise

        response = call_with_retry(_do_call, label=f"openrouter:{self.model}:summarize")
        usage = response.usage
        prompt_tokens = int(usage.prompt_tokens or 0) if usage else 0
        completion_tokens = int(usage.completion_tokens or 0) if usage else 0
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        non_cached = prompt_tokens - cached
        cost = self._completion_cost(non_cached, cached, completion_tokens)
        self.n_calls += 1
        self.cost += cost
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        # Some OpenRouter models (notably Kimi-K2.6) silently route output
        # into the reasoning channel even when reasoning_effort isn't sent,
        # leaving message.content empty / None. Fall back to message.reasoning
        # before giving up — the compactor would otherwise log
        # "empty summary — skipping compaction" and the agent's transcript
        # keeps growing until provider context overflow.
        msg_obj = response.choices[0].message
        text = msg_obj.content or ""
        if not text:
            text = (getattr(msg_obj, "reasoning", None) or "") or ""
        print(
            f"[openrouter:{self.model}:summarize] in={prompt_tokens} "
            f"cached={cached} out={completion_tokens} content_chars={len(text)}"
        )
        return text


if __name__ == "__main__":
    m = OpenRouterModel()
    print(m.query([{"role": "user", "content": "Hello!"}]).content)
