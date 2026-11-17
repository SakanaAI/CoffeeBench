"""$0 test of CoffeeTerminus' context budget (no LLM calls, no containers):
summary calls get SUMMARY_MAX_TOKENS and walk down the effort ladder on output overflow,
a failed summarization truncates the oldest turns to the budget, and the token
count is calibrated on provider-reported prompt tokens.

    PYTHONPATH=. ~/.local/share/uv/tools/harbor/bin/python scripts/test_context_budget.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from harbor.llms.base import LLMResponse, OutputLengthExceededError
from harbor.llms.chat import Chat
from harbor.models.metric import UsageInfo

from coffeebench.harbor_agent import (
    EFFORT_LADDER,
    SUMMARY_MAX_TOKENS,
    REASONING_OFF,
    TRUNCATION_NOTE,
    CoffeeTerminus,
)


class FakeLLM:
    def __init__(self, fail_first: int = 0, limit: int = 200_000):
        self.calls: list[dict] = []
        self.fail_first = fail_first
        self.limit = limit

    async def call(self, prompt, message_history=None, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail_first:
            raise OutputLengthExceededError("hit max_tokens limit")
        return LLMResponse(content="summary", usage=UsageInfo(prompt_tokens=1000, completion_tokens=10,
                                                              cache_tokens=0, cost_usd=0.0))

    def get_model_context_limit(self):
        return self.limit


def make_agent(llm, effort="max"):
    a = CoffeeTerminus(logs_dir=Path(tempfile.mkdtemp()), model_name="openrouter/test/model",
                       llm_call_kwargs={"max_tokens": 32768,
                                        "extra_body": {"provider": {"order": ["x"]},
                                                       **({"reasoning": {"effort": effort}} if effort else {})}})
    a._llm = llm
    return a


def turns(n, size=4000):
    msgs = [{"role": "user", "content": "INSTRUCTION"}]
    for i in range(n):
        msgs += [{"role": "assistant", "content": f"a{i} " + "x " * size},
                 {"role": "user", "content": f"u{i} " + "y " * size}]
    msgs.append({"role": "assistant", "content": "last"})
    return msgs


async def main():
    # 1. summary kwargs: bigger cap; retries walk down the effort ladder; base kwargs restored.
    llm = FakeLLM(fail_first=5)
    a = make_agent(llm)
    r, _ = await a._run_subagent(prompt="p", message_history=[], steps=[], session_id="s",
                                 agent_name="n", filename_suffix="summary", summary_text="t")
    assert r.content == "summary"
    assert [c["max_tokens"] for c in llm.calls] == [SUMMARY_MAX_TOKENS] * 6
    assert [c["extra_body"]["reasoning"] for c in llm.calls] == [{"effort": e} for e in EFFORT_LADDER] + [REASONING_OFF]
    assert a._llm_call_kwargs["max_tokens"] == 32768
    assert a._llm_call_kwargs["extra_body"]["reasoning"] == {"effort": "max"}
    print("ok: summary call kwargs + effort ladder max..low, off")

    # 1a. Gemini (focal effort high): high -> medium -> low, then the error propagates.
    llm = FakeLLM(fail_first=99)
    a = make_agent(llm, effort="high")
    try:
        await a._run_subagent(prompt="p", message_history=[], steps=[], session_id="s",
                              agent_name="n", filename_suffix="summary", summary_text="t")
        raise AssertionError("expected OutputLengthExceededError")
    except OutputLengthExceededError:
        pass
    assert [c["extra_body"]["reasoning"] for c in llm.calls] == [
        {"effort": "high"}, {"effort": "medium"}, {"effort": "low"}, REASONING_OFF]
    print("ok: ladder from high")

    # 1b. no effort control sent (bg / MiMo): API default, then reasoning off, then the error propagates.
    llm = FakeLLM(fail_first=5)
    a = make_agent(llm, effort=None)
    try:
        await a._run_subagent(prompt="p", message_history=[], steps=[], session_id="s",
                              agent_name="n", filename_suffix="summary", summary_text="t")
        raise AssertionError("expected OutputLengthExceededError")
    except OutputLengthExceededError:
        pass
    assert "reasoning" not in llm.calls[0]["extra_body"]
    assert [c["extra_body"].get("reasoning") for c in llm.calls] == [None, REASONING_OFF]
    print("ok: no effort control -> default, off")

    # 2. failed summarization -> truncation to <= TRUNCATE_TARGET_FRAC of budget.
    llm = FakeLLM(fail_first=99, limit=50_000)
    a = make_agent(llm)
    chat = Chat(llm)
    chat._messages = turns(30)
    before = a._count_total_tokens(chat)
    assert before > 50_000
    out = await a._check_proactive_summarization(chat, "INSTRUCTION", session=None)
    after = a._count_total_tokens(chat)
    m = chat.messages
    assert out is None and a._n_truncations == 1
    assert m[0]["content"].startswith("INSTRUCTION") and TRUNCATION_NOTE in m[0]["content"]
    assert m[1]["role"] == "assistant" and m[2]["role"] == "user" and m[-1]["content"] == "last"
    assert all(m[i]["role"] != m[i + 1]["role"] for i in range(len(m) - 1)), "roles alternate"
    assert after <= 25_000, after
    print(f"ok: truncation {before} -> {after} tokens, {len(m)} messages")

    # 2a. after an earlier summarization the hand-off (summary) turn is kept (v2.1).
    llm = FakeLLM(fail_first=99, limit=50_000)
    a = make_agent(llm)
    chat = Chat(llm)
    body = turns(30)  # [instruction, a0, u1, a1, ...]; post-summary layout = upstream terminus-2's
    chat._messages = body[:1] + [{"role": "user", "content": "You are picking up a CoffeeBench session ... SUMMARY"},
                                 {"role": "assistant", "content": "questions"}] + body[2:]
    await a._check_proactive_summarization(chat, "INSTRUCTION", session=None)
    m = chat.messages
    assert "SUMMARY" in m[1]["content"] and m[2]["content"] == "questions" and m[3]["content"] == TRUNCATION_NOTE
    assert all(m[i]["role"] != m[i + 1]["role"] for i in range(1, len(m) - 1)), "roles alternate after the head"
    print("ok: truncation keeps the summary hand-off turn")

    # 2c. transient-error classification (v2.1).
    from coffeebench.harbor_agent import _is_transient_llm_error as tr
    class E(Exception):
        pass
    assert tr(E("litellm.RateLimitError: 429 temporarily rate-limited")) and tr(E("502 Bad Gateway"))
    assert not tr(E("prompt is too long: 142,900 tokens > limit")) and not tr(E("context 1,104,293 tokens"))
    print("ok: transient classification")

    # 2b. under budget -> no summarization attempt at all.
    llm = FakeLLM(limit=10_000_000)
    a = make_agent(llm)
    chat = Chat(llm)
    chat._messages = turns(3)
    assert await a._check_proactive_summarization(chat, "I", session=None) is None and not llm.calls
    print("ok: no-op under budget")

    # 3. calibration: provider reports 2x the litellm estimate -> counts double.
    a = make_agent(FakeLLM())
    msgs = turns(5)
    est = a._estimate_tokens(msgs)
    a._calibrate_tokens(msgs, LLMResponse(content="", usage=UsageInfo(
        prompt_tokens=2 * est, completion_tokens=1, cache_tokens=0, cost_usd=0.0)))
    assert abs(a._estimate_tokens(msgs) - 2 * est) <= 2, (a._estimate_tokens(msgs), est)
    a._calibrate_tokens(msgs, LLMResponse(content="", usage=UsageInfo(
        prompt_tokens=1, completion_tokens=1, cache_tokens=0, cost_usd=0.0)))  # implausible: ignored
    assert abs(a._tok_ratio - 2.0) < 0.01
    print("ok: calibration on reported prompt tokens")


asyncio.run(main())
print("ALL OK")
