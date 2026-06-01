"""Gemini wrapper — native tool-use via Google's `google-genai` SDK."""

from __future__ import annotations

import os
import uuid
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

from coffeebench.models._retry import call_with_retry
from coffeebench.models.types import ModelResponse, ToolCall, ToolSpec

load_dotenv()


class GeminiModel:
    DEFAULT_MAX_INPUT_TOKENS = 200_000

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        enable_thinking: bool = False,
        thinking_budget: int = -1,  # -1 = dynamic / "let the model decide"
    ):
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.model = model
        # 5-minute per-call HTTP timeout. Without this, the google-genai
        # SDK defaults to no timeout and a stuck connection can wedge a
        # long-horizon run for hours (observed on Gemini 3.1 preview
        # during this bench: 9.5h hang on Day 14 with no retry firing).
        # 300s is comfortably above normal Gemini latency (~10-30s with
        # thinking) so it doesn't bite legitimate slow calls; on timeout
        # the request raises and `_retry.call_with_retry` reschedules.
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=genai_types.HttpOptions(timeout=300_000),  # ms
        )
        self.max_tokens = 4096
        self.temperature = 0.0
        self.enable_thinking = bool(enable_thinking)
        # `-1` is genai's sentinel for "dynamic" thinking budget;
        # anything > 0 caps the thoughts token budget at that value.
        # NOTE: Gemini 3.x is thinking-required and rejects
        # `thinking_budget=0` ("Budget 0 is invalid"); when the user
        # opts out of thinking we still send -1 (dynamic) but with
        # `include_thoughts=False` so the response.thinking is empty
        # and the model spends minimal latency on chain-of-thought.
        self.thinking_budget = int(thinking_budget)
        self.system_prompt: str | None = None
        self.pricing = {
            "gemini-3-pro-preview": {
                "input": 2.00,
                "cached_input": 0.20,
                "output": 12.00,
            },
            "gemini-3.1-pro-preview": {
                "input": 2.00,
                "cached_input": 0.20,
                "output": 12.00,
            },
            "gemini-2.5-pro": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        }

    # ---------- pricing ----------

    def _completion_cost(
        self,
        non_cached_input_tokens: int,
        cached_input_tokens: int,
        completion_tokens: int,
    ) -> float:
        p = self.pricing[self.model]
        return (
            non_cached_input_tokens * p["input"]
            + cached_input_tokens * p["cached_input"]
            + completion_tokens * p["output"]
        ) / 1_000_000

    # ---------- internal → genai Contents translation ----------

    @staticmethod
    def _tools_to_genai(tools: list[ToolSpec] | None) -> list[genai_types.Tool] | None:
        if not tools:
            return None
        decls = [
            genai_types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.input_schema,
            )
            for t in tools
        ]
        return [genai_types.Tool(function_declarations=decls)]

    def _to_genai_contents(self, messages: list[dict]) -> list[genai_types.Content]:
        out: list[genai_types.Content] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                # system carried out-of-band on the config; skip here.
                continue
            if role == "user":
                out.append(
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=str(m.get("content", "")))],
                    )
                )
                continue
            if role == "tool":
                # function_response back to the model. genai pairs by
                # function NAME; the call id is for our own
                # bookkeeping (agent harness pairing).
                name = m.get("name") or ""
                content = m.get("content", "")
                out.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part.from_function_response(
                                name=str(name),
                                response={"output": content},
                            )
                        ],
                    )
                )
                continue
            if role == "assistant":
                parts: list[genai_types.Part] = []
                raw = m.get("_raw") or {}
                thought_blocks = (
                    raw.get("thought_parts") if isinstance(raw, dict) else None
                )
                # Replay any thinking parts the model emitted on this
                # turn. They keep their thought_signature too, which
                # genai requires on continuation turns.
                if thought_blocks:
                    for tb in thought_blocks:
                        kwargs: dict[str, Any] = {"text": tb.get("text", "")}
                        sig = tb.get("thought_signature")
                        if sig:
                            kwargs["thought_signature"] = sig
                        kwargs["thought"] = True
                        parts.append(genai_types.Part(**kwargs))
                text = m.get("content") or ""
                if text:
                    parts.append(genai_types.Part(text=text))
                tool_calls = m.get("tool_calls") or []
                signatures = (
                    raw.get("thought_signatures", {}) if isinstance(raw, dict) else {}
                )
                for tc in tool_calls:
                    tc_id = tc.id if isinstance(tc, ToolCall) else tc["id"]
                    tc_name = tc.name if isinstance(tc, ToolCall) else tc["name"]
                    tc_input = tc.input if isinstance(tc, ToolCall) else tc["input"]
                    fc = genai_types.FunctionCall(
                        name=str(tc_name),
                        args=dict(tc_input or {}),
                    )
                    sig = signatures.get(tc_id)
                    part = genai_types.Part(function_call=fc)
                    if sig:
                        # Attach signature on the part so the API
                        # accepts the assistant turn on replay.
                        part.thought_signature = sig
                    parts.append(part)
                if not parts:
                    # Empty assistant turns are illegal — emit a
                    # placeholder text part. (Should not occur in
                    # normal play; defensive.)
                    parts.append(genai_types.Part(text=""))
                out.append(genai_types.Content(role="model", parts=parts))
                continue
            if role == "developer":
                # OpenAI-style developer role; map to system fragment
                # appended to the system_instruction equivalent.
                self.system_prompt = (
                    (self.system_prompt or "") + "\n" + str(m.get("content", ""))
                ).strip() or self.system_prompt
                continue
            # Unknown role — skip.
        return out

    # ---------- query ----------

    def query(
        self,
        messages: list[dict],
        tools: list[ToolSpec] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ModelResponse:
        contents = self._to_genai_contents(messages)
        genai_tools = self._tools_to_genai(tools)

        cfg_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "thinking_config": genai_types.ThinkingConfig(
                include_thoughts=self.enable_thinking,
                thinking_budget=self.thinking_budget,
            ),
        }
        if self.system_prompt:
            cfg_kwargs["system_instruction"] = self.system_prompt
        if genai_tools:
            cfg_kwargs["tools"] = genai_tools
            mode = "any" if tool_choice == "any" else "auto"
            cfg_kwargs["tool_config"] = genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(mode=mode),
            )
        config = genai_types.GenerateContentConfig(**cfg_kwargs)

        def _do_call():
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

        response = call_with_retry(_do_call, label=f"gemini:{self.model}")

        # ---------- usage / cost ----------
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
        thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        # Add thoughts to output token total — they're billed as output.
        total_output = completion_tokens + thoughts_tokens
        non_cached = prompt_tokens - cached
        cost = self._completion_cost(non_cached, cached, total_output)
        self.n_calls += 1
        self.cost += cost
        self.last_input_tokens = prompt_tokens
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += total_output
        print(
            f"[gemini:{self.model}] in={prompt_tokens} cached={cached} "
            f"out={completion_tokens} thoughts={thoughts_tokens}"
        )

        # ---------- parse response parts ----------
        thinking_text_parts: list[str] = []
        thought_blocks_for_raw: list[dict] = []
        content_text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thought_signatures: dict[str, str] = {}

        candidate = response.candidates[0] if response.candidates else None
        finish_reason = ""
        if candidate is not None:
            finish_reason = str(getattr(candidate, "finish_reason", "") or "")
            content_obj = getattr(candidate, "content", None)
            for part in getattr(content_obj, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                is_thought = bool(getattr(part, "thought", False))
                text = getattr(part, "text", None)
                sig = getattr(part, "thought_signature", None)
                if fc is not None:
                    tc_id = f"{fc.name}_{uuid.uuid4().hex[:8]}"
                    tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            name=fc.name,
                            input=dict(fc.args or {}),
                        )
                    )
                    if sig:
                        thought_signatures[tc_id] = sig
                elif is_thought:
                    thinking_text_parts.append(text or "")
                    block: dict[str, Any] = {"text": text or ""}
                    if sig:
                        block["thought_signature"] = sig
                    thought_blocks_for_raw.append(block)
                elif text:
                    content_text_parts.append(text)

        raw_payload: dict[str, Any] = {}
        if thought_signatures:
            raw_payload["thought_signatures"] = thought_signatures
        if thought_blocks_for_raw:
            raw_payload["thought_parts"] = thought_blocks_for_raw

        return ModelResponse(
            content="".join(content_text_parts),
            thinking="".join(thinking_text_parts),
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            cost=cost,
            raw=raw_payload or None,
        )

    # ---------- usage ----------
    def get_usage_stats(self) -> dict:
        return {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "last_input_tokens": self.last_input_tokens,
        }

    # ---------- summarize (used by ContextCompactor) ----------
    def summarize(
        self,
        instructions: str,
        content: str,
        max_tokens: int = 4096,
    ) -> str:
        cfg = genai_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=int(max_tokens),
            system_instruction=instructions,
            # Gemini 3.x requires thinking mode (budget != 0) — passing
            # budget=0 returns 400. Use dynamic budget with thoughts
            # hidden from the response so the summary stays terse.
            thinking_config=genai_types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=-1,
            ),
        )
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=str(content))],
            )
        ]

        def _do_call():
            return self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=cfg,
            )

        response = call_with_retry(_do_call, label=f"gemini:{self.model}:summarize")
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
        non_cached = prompt_tokens - cached
        cost = self._completion_cost(non_cached, cached, completion_tokens)
        self.n_calls += 1
        self.cost += cost
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None:
            return ""
        for part in getattr(candidate.content, "parts", None) or []:
            t = getattr(part, "text", None)
            if t:
                return t
        return ""


if __name__ == "__main__":
    m = GeminiModel()
    print(m.query([{"role": "user", "content": "Hello!"}]).content)
