"""OpenAI wrapper — native tool-use via the Responses API."""

import os

from openai import OpenAI
from dotenv import load_dotenv

from coffeebench.models._retry import call_with_retry
from coffeebench.models.types import ModelResponse, ToolCall, ToolSpec

load_dotenv()


class OpenAIModel:
    DEFAULT_MAX_INPUT_TOKENS = 200_000

    def __init__(self, model: str = "gpt-5.5", enable_thinking: bool = False):
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=300.0)
        self._skip_temperature = False
        self._skip_reasoning = False
        self.reasoning_effort = "high" if enable_thinking else "low"
        self.max_tokens = 4096
        self.temperature = 0.0
        self.pricing = {
            "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
        }

    def _completion_cost(
        self, non_cached_input_tokens, cached_input_tokens, output_tokens
    ) -> float:
        p = self.pricing[self.model]
        return (
            non_cached_input_tokens * p["input"]
            + cached_input_tokens * p["cached_input"]
            + output_tokens * p["output"]
        ) / 1_000_000

    @staticmethod
    def _tools_to_openai(tools: list[ToolSpec] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in tools
        ]

    def _to_responses_input(self, messages: list[dict]) -> list[dict]:
        """Translate internal history into Responses API `input` items."""
        items: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role in ("system", "developer", "user"):
                items.append(
                    {
                        "role": "developer" if role == "system" else role,
                        "content": m.get("content", ""),
                    }
                )
                continue
            if role == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": m["tool_call_id"],
                        "output": m.get("content", ""),
                    }
                )
                continue
            if role == "assistant":
                raw = m.get("_raw")
                if raw is not None:
                    # _raw is a list of Responses output items; replay verbatim.
                    items.extend(raw)
                else:
                    text = m.get("content") or ""
                    if text:
                        items.append({"role": "assistant", "content": text})
                    for tc in m.get("tool_calls") or []:
                        import json as _json

                        items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.id
                                if isinstance(tc, ToolCall)
                                else tc["id"],
                                "name": tc.name
                                if isinstance(tc, ToolCall)
                                else tc["name"],
                                "arguments": _json.dumps(
                                    tc.input
                                    if isinstance(tc, ToolCall)
                                    else tc["input"]
                                ),
                            }
                        )
                continue
        return items

    def query(
        self,
        messages: list[dict],
        tools: list[ToolSpec] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ModelResponse:
        kwargs: dict = {
            "model": self.model,
            "input": self._to_responses_input(messages),
            "max_output_tokens": self.max_tokens,
        }
        if not self._skip_temperature:
            kwargs["temperature"] = self.temperature
        if not self._skip_reasoning:
            # `summary: "auto"` asks the Responses API to surface the
            # reasoning summary as a `reasoning` output item — required
            # for `response.thinking` to be non-empty. Without it the
            # reasoning is hidden inside the model and we'd have to
            # fall back to text content for the agent's `thought`.
            kwargs["reasoning"] = {
                "effort": self.reasoning_effort,
                "summary": "auto",
            }
        oa_tools = self._tools_to_openai(tools)
        if oa_tools:
            kwargs["tools"] = oa_tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            # parallel_tool_calls is supported on Responses; constrain to 1.
            kwargs["parallel_tool_calls"] = False

        def _do_call():
            try:
                return self.client.responses.create(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    "temperature" in msg
                    and any(
                        k in msg for k in ("not supported", "deprecated", "unsupported")
                    )
                    and "temperature" in kwargs
                ):
                    self._skip_temperature = True
                    kwargs.pop("temperature", None)
                    return self.client.responses.create(**kwargs)
                if (
                    "reasoning" in msg
                    and any(
                        k in msg
                        for k in ("not supported", "unrecognized", "unknown", "invalid")
                    )
                    and "reasoning" in kwargs
                ):
                    self._skip_reasoning = True
                    kwargs.pop("reasoning", None)
                    return self.client.responses.create(**kwargs)
                raise

        response = call_with_retry(_do_call, label=f"openai:{self.model}")

        usage = response.usage
        input_tokens = usage.input_tokens
        cached_tokens = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        non_cached = input_tokens - cached_tokens
        output_tokens = usage.output_tokens
        cost = self._completion_cost(non_cached, cached_tokens, output_tokens)
        self.n_calls += 1
        self.cost += cost
        self.last_input_tokens = int(input_tokens)
        self.total_input_tokens += int(input_tokens)
        self.total_output_tokens += int(output_tokens)
        print(
            f"[openai:{self.model}] in={input_tokens} cached={cached_tokens} "
            f"out={output_tokens} status={getattr(response, 'status', '?')}"
        )

        # Walk response.output to extract text, tool_calls, and reasoning summary.
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_items: list[dict] = []
        import json as _json

        for item in response.output or []:
            itype = getattr(item, "type", None)
            if itype == "message":
                # The assistant's user-visible reply; concatenate text segments.
                for part in getattr(item, "content", []) or []:
                    ptype = getattr(part, "type", None)
                    if ptype == "output_text":
                        content_parts.append(getattr(part, "text", "") or "")
                # Replay item: include id+content so Responses can stitch the
                # next round of input correctly.
                raw_items.append(_serialize_responses_item(item))
            elif itype == "function_call":
                args_raw = getattr(item, "arguments", "") or ""
                try:
                    args = _json.loads(args_raw) if args_raw else {}
                except (ValueError, TypeError):
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=getattr(item, "call_id", "") or "",
                        name=getattr(item, "name", "") or "",
                        input=args,
                    )
                )
                raw_items.append(_serialize_responses_item(item))
            elif itype == "reasoning":
                summary = getattr(item, "summary", None)
                if summary:
                    for s in summary:
                        thinking_parts.append(getattr(s, "text", "") or "")
                raw_items.append(_serialize_responses_item(item))

        return ModelResponse(
            content="".join(content_parts) or (response.output_text or ""),
            thinking="".join(thinking_parts),
            tool_calls=tool_calls,
            stop_reason=getattr(response, "status", "") or "",
            cost=cost,
            raw=raw_items,
        )

    def get_usage_stats(self) -> dict:
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
            "input": [
                {"role": "developer", "content": instructions},
                {"role": "user", "content": content},
            ],
            "max_output_tokens": int(max_tokens),
        }
        if not self._skip_temperature:
            kwargs["temperature"] = self.temperature
        if not self._skip_reasoning:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        def _do_call():
            try:
                return self.client.responses.create(**kwargs)
            except Exception as exc:
                msg = str(exc).lower()
                if "temperature" in msg and "temperature" in kwargs:
                    self._skip_temperature = True
                    kwargs.pop("temperature", None)
                    return self.client.responses.create(**kwargs)
                if "reasoning" in msg and "reasoning" in kwargs:
                    self._skip_reasoning = True
                    kwargs.pop("reasoning", None)
                    return self.client.responses.create(**kwargs)
                raise

        response = call_with_retry(_do_call, label=f"openai:{self.model}:summarize")
        usage = response.usage
        input_tokens = int(usage.input_tokens or 0)
        cached_tokens = int(
            getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        )
        output_tokens = int(usage.output_tokens or 0)
        cost = self._completion_cost(
            input_tokens - cached_tokens, cached_tokens, output_tokens
        )
        self.n_calls += 1
        self.cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        return response.output_text or ""


def _serialize_responses_item(item) -> dict:
    """Convert a Responses API item to a JSON-friendly dict for replay.

    The Responses API accepts the same item shapes back as input; we
    reflect the relevant fields verbatim. Unknown fields are dropped
    rather than passed through, since some are read-only and rejected
    on input.
    """
    itype = getattr(item, "type", None)
    if itype == "function_call":
        return {
            "type": "function_call",
            "id": getattr(item, "id", None),
            "call_id": getattr(item, "call_id", None),
            "name": getattr(item, "name", None),
            "arguments": getattr(item, "arguments", "") or "",
        }
    if itype == "message":
        # Normalize content parts.
        parts = []
        for p in getattr(item, "content", []) or []:
            if getattr(p, "type", None) == "output_text":
                parts.append(
                    {"type": "output_text", "text": getattr(p, "text", "") or ""}
                )
        return {
            "type": "message",
            "id": getattr(item, "id", None),
            "role": getattr(item, "role", "assistant"),
            "content": parts,
        }
    if itype == "reasoning":
        # The Responses API allows reasoning items to be replayed; preserve
        # id and summary text only — internal state isn't user-addressable.
        summary = []
        for s in getattr(item, "summary", []) or []:
            summary.append(
                {"type": "summary_text", "text": getattr(s, "text", "") or ""}
            )
        return {
            "type": "reasoning",
            "id": getattr(item, "id", None),
            "summary": summary,
        }
    # Fallback: best-effort dump.
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return {"type": itype}


if __name__ == "__main__":
    m = OpenAIModel()
    print(m.query([{"role": "user", "content": "Hello!"}]).content)
