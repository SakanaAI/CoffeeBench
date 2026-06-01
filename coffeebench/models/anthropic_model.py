"""Anthropic Claude wrapper — native tool-use harness."""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from coffeebench.models._retry import call_with_retry
from coffeebench.models.types import ModelResponse, ToolCall, ToolSpec

load_dotenv()


class AnthropicModel:
    DEFAULT_MAX_INPUT_TOKENS = 200_000

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 4096,
    ):
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.model = model
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=300.0)
        # max_tokens reserves room for the visible output. With legacy
        # `thinking={enabled, budget}`, Anthropic requires
        # max_tokens > budget — we size accordingly in query().
        self.max_tokens = 4096
        self.temperature = 0.0
        # Anthropic forces temperature=1.0 with extended thinking, and
        # newer models (Opus 4.7) deprecate the temperature param entirely.
        # Skip temperature whenever thinking is on.
        self._skip_temperature = bool(enable_thinking)
        self._skip_thinking = not bool(enable_thinking)
        self.thinking_budget_tokens = int(thinking_budget_tokens)
        self.system_prompt: str | None = None

        self.pricing = {
            "claude-sonnet-4-6": {
                "input": 3.00,
                "cache_write": 3.75,
                "cache_read": 0.30,
                "output": 15.00,
            },
            "claude-opus-4-7": {
                "input": 5.00,
                "cache_write": 6.25,
                "cache_read": 0.50,
                "output": 25.00,
            },
            "claude-haiku-4-5": {
                "input": 1.00,
                "cache_write": 1.25,
                "cache_read": 0.10,
                "output": 5.00,
            },
        }

    # ---------- pricing ----------

    def _completion_cost(
        self, input_tokens, output_tokens, cache_write_tokens=0, cache_read_tokens=0
    ) -> float:
        p = self.pricing[self.model]
        return (
            input_tokens * p["input"]
            + cache_write_tokens * p["cache_write"]
            + cache_read_tokens * p["cache_read"]
            + output_tokens * p["output"]
        ) / 1_000_000

    # ---------- internal → Anthropic message translation ----------

    def _to_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        """Translate the agent's internal history into Anthropic's
        messages format. Strict alternation is preserved by how agent.py
        appends turns; this method is a pure shape translator and never
        re-orders.
        """
        out: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                # Anthropic carries the system prompt as a separate kwarg,
                # never as a message; agent.py routes system → self.system_prompt.
                continue
            if role == "tool":
                # Tool results are sent back as a user-role message with a
                # `tool_result` content block keyed by tool_use_id.
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": m.get("content", ""),
                            }
                        ],
                    }
                )
                continue
            if role == "assistant":
                # Replay the full raw content blocks if we have them — this
                # preserves any thinking blocks Anthropic returned, which is
                # required for valid continuation when extended thinking
                # was active. Fallback: rebuild from text + tool_calls.
                raw = m.get("_raw")
                if raw is not None:
                    out.append({"role": "assistant", "content": raw})
                else:
                    blocks: list[dict] = []
                    text = m.get("content") or ""
                    if text:
                        blocks.append({"type": "text", "text": text})
                    for tc in m.get("tool_calls") or []:
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id if isinstance(tc, ToolCall) else tc["id"],
                                "name": tc.name
                                if isinstance(tc, ToolCall)
                                else tc["name"],
                                "input": tc.input
                                if isinstance(tc, ToolCall)
                                else tc["input"],
                            }
                        )
                    out.append(
                        {
                            "role": "assistant",
                            "content": blocks or [{"type": "text", "text": ""}],
                        }
                    )
                continue
            # role == "user"
            out.append({"role": "user", "content": m.get("content", "")})
        return out

    @staticmethod
    def _tools_to_anthropic(tools: list[ToolSpec] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    # ---------- query ----------

    def query(
        self,
        messages: list[dict],
        tools: list[ToolSpec] | None = None,
        tool_choice: dict | None = None,
    ) -> ModelResponse:
        formatted_messages = self._to_anthropic_messages(messages)
        # Cache-control breakpoint on the last user/assistant content
        # block boosts prompt-cache hits across consecutive turns.
        # Wrap the last message's text content in a list-of-blocks form
        # with cache_control attached. Tool-result messages are already
        # block-form; attach cache_control to the last block.
        if formatted_messages:
            last = formatted_messages[-1]
            content = last["content"]
            if isinstance(content, str):
                formatted_messages[-1] = {
                    "role": last["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            elif isinstance(content, list) and content:
                # Don't mutate the underlying list (it may be shared with
                # the agent's history).
                cloned = [dict(b) if isinstance(b, dict) else b for b in content]
                if isinstance(cloned[-1], dict):
                    cloned[-1]["cache_control"] = {"type": "ephemeral"}
                formatted_messages[-1] = {"role": last["role"], "content": cloned}

        # Reserve max_tokens for visible output + (legacy) thinking budget.
        is_opus_47 = self.model == "claude-opus-4-7"
        effective_max_tokens = self.max_tokens
        if not self._skip_thinking and not is_opus_47:
            effective_max_tokens = self.max_tokens + self.thinking_budget_tokens

        kwargs: dict = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": effective_max_tokens,
        }
        if not self._skip_temperature:
            kwargs["temperature"] = 1.0 if not self._skip_thinking else self.temperature
        if not self._skip_thinking:
            if is_opus_47:
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["extra_body"] = {"output_config": {"effort": "high"}}
            else:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget_tokens,
                }
        if self.system_prompt:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        anthro_tools = self._tools_to_anthropic(tools)
        if anthro_tools:
            kwargs["tools"] = anthro_tools
            # Default: let the model choose whether to call a tool. Agent
            # forces a tool call by passing tool_choice={"type":"any"} when
            # it wants to require one.
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            else:
                kwargs["tool_choice"] = {
                    "type": "auto",
                    "disable_parallel_tool_use": True,
                }

        def _do_call():
            try:
                return self.client.messages.create(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    "thinking" in msg
                    and (
                        "not supported" in msg
                        or "unrecognized" in msg
                        or "invalid" in msg
                    )
                    and "thinking" in kwargs
                ):
                    self._skip_thinking = True
                    kwargs.pop("thinking", None)
                    kwargs.pop("extra_body", None)
                    if not self._skip_temperature:
                        kwargs["temperature"] = self.temperature
                    kwargs["max_tokens"] = self.max_tokens
                    return self.client.messages.create(**kwargs)
                if (
                    "temperature" in msg
                    and (
                        "deprecated" in msg
                        or "not supported" in msg
                        or "invalid" in msg
                    )
                    and "temperature" in kwargs
                ):
                    self._skip_temperature = True
                    kwargs.pop("temperature", None)
                    return self.client.messages.create(**kwargs)
                raise

        response = call_with_retry(_do_call, label=f"anthropic:{self.model}")
        usage = response.usage

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_write_tokens = usage.cache_creation_input_tokens or 0
        cache_read_tokens = usage.cache_read_input_tokens or 0
        cost = self._completion_cost(
            input_tokens, output_tokens, cache_write_tokens, cache_read_tokens
        )
        self.n_calls += 1
        self.cost += cost
        prompt_total = int(input_tokens + cache_write_tokens + cache_read_tokens)
        self.last_input_tokens = prompt_total
        self.total_input_tokens += prompt_total
        self.total_output_tokens += int(output_tokens)
        print(
            f"[anthropic:{self.model}] in={input_tokens} cw={cache_write_tokens} "
            f"cr={cache_read_tokens} out={output_tokens} stop={getattr(response, 'stop_reason', '?')}"
        )

        # Parse content blocks → unified ModelResponse.
        content_text_parts: list[str] = []
        thinking_text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_blocks: list[dict] = []
        for b in response.content or []:
            btype = getattr(b, "type", None)
            if btype == "text":
                content_text_parts.append(getattr(b, "text", "") or "")
                raw_blocks.append({"type": "text", "text": b.text})
            elif btype == "thinking":
                thinking_text_parts.append(getattr(b, "thinking", "") or "")
                # Preserve the full thinking block — Anthropic requires
                # the original signature on replay for cache-friendly
                # multi-turn continuation.
                raw_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": getattr(b, "thinking", ""),
                        "signature": getattr(b, "signature", ""),
                    }
                )
            elif btype == "redacted_thinking":
                raw_blocks.append(
                    {
                        "type": "redacted_thinking",
                        "data": getattr(b, "data", ""),
                    }
                )
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=b.id,
                        name=b.name,
                        input=b.input or {},
                    )
                )
                raw_blocks.append(
                    {
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.input or {},
                    }
                )

        return ModelResponse(
            content="".join(content_text_parts),
            thinking="".join(thinking_text_parts),
            tool_calls=tool_calls,
            stop_reason=getattr(response, "stop_reason", "") or "",
            cost=cost,
            raw=raw_blocks,
        )

    # ---------- usage / summarize ----------

    def get_usage_stats(self) -> dict[str, float]:
        return {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "last_input_tokens": self.last_input_tokens,
        }

    def summarize(self, instructions: str, content: str, max_tokens: int = 4096) -> str:
        kwargs: dict = {
            "model": self.model,
            "system": instructions,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(max_tokens),
        }
        if not self._skip_temperature:
            kwargs["temperature"] = self.temperature

        def _do_call():
            try:
                return self.client.messages.create(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    "temperature" in msg
                    and (
                        "deprecated" in msg
                        or "not supported" in msg
                        or "invalid" in msg
                    )
                    and "temperature" in kwargs
                ):
                    self._skip_temperature = True
                    kwargs.pop("temperature", None)
                    return self.client.messages.create(**kwargs)
                raise

        response = call_with_retry(_do_call, label=f"anthropic:{self.model}:summarize")
        usage = response.usage
        input_tokens = int(usage.input_tokens or 0)
        output_tokens = int(usage.output_tokens or 0)
        cw = int(usage.cache_creation_input_tokens or 0)
        cr = int(usage.cache_read_input_tokens or 0)
        cost = self._completion_cost(input_tokens, output_tokens, cw, cr)
        self.n_calls += 1
        self.cost += cost
        self.total_input_tokens += input_tokens + cw + cr
        self.total_output_tokens += output_tokens
        if response.content:
            return "".join(
                getattr(b, "text", "") for b in response.content if hasattr(b, "text")
            )
        return ""


if __name__ == "__main__":
    m = AnthropicModel(model="claude-sonnet-4-6", enable_thinking=False)
    r = m.query([{"role": "user", "content": "Hello!"}])
    print(r.content)
    print(m.get_usage_stats())
