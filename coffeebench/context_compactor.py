"""Context-window compaction for long-running agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from coffeebench.models import Model


COMPACTION_SYSTEM_PROMPT = """\
You are summarizing a transcript of an LLM agent's turns in a multi-day business simulation. Your task is to compress this transcript into a factual memory note that the agent will read to continue operating without the full transcript.

Rules — read carefully:
1. Output ONLY facts present in the transcript. Do not invent or extrapolate.
2. Use NO evaluative language. Avoid words like "wisely", "smartly", "mistakenly", "should have", "good decision", "regret", "the right call". Just record what happened.
3. Stay neutral. Do not flag patterns or judge counterparties.
4. Preserve all numbers exactly: prices, quantities, dates (day numbers), agent IDs, invoice IDs.

Output format — use these section headers verbatim. Omit a section entirely if it has no content.

## Counterparties & deals
For each completed deal: who (buyer/seller IDs), item, qty, unit price, day accepted, payment terms (NET-X), delivery status.

## Open AR / AP
Outstanding receivables and payables not yet settled, with invoice ID, counterparty, balance, due day, overdue flag if past due.

## Messages exchanged
Digest of incoming/outgoing DMs: counterparty, day, one-line gist. List the most recent ~10 if many.

## Observed market patterns
Concrete observations from the transcript supported by numbers: e.g., "day 45 roasted_coffee_kg sold 18 kg at $22/kg vs ~10 kg typical" or "competing retailer priced roasted_coffee_kg at $20/kg on day 60". Only include if numbers are present in the transcript.

## Pending actions
Listings posted, offers made/received, production calls in flight that have not yet resolved.

## Last self-observed state
Most recent self-reported cash, inventory snapshot, AR/AP totals from the morning observation.
"""

COMPACTION_USER_PROMPT_TEMPLATE = """\
Compress the following transcript into the factual memory format described in your instructions. The transcript covers turns the agent took in a middle range of the run; turns before and after this range remain visible to the agent in their original form.

TRANSCRIPT:
{transcript}
"""


def _render_transcript(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


@dataclass
class CompactionEvent:
    """Diagnostic record of a single compaction. Exposed via
    `compactor.events` for run-end reporting."""

    triggered_at_tokens: int
    middle_start: int
    middle_end: int
    middle_msg_count: int
    summary_chars: int
    extras: dict[str, Any] = field(default_factory=dict)


class ContextCompactor:
    """Threshold-driven, summary-based context compaction.

    Wired into Agent.step(): after each model call, if the most-recent
    `model.last_input_tokens` exceeds `threshold_ratio * model.max_input_tokens`,
    `compact(messages)` is invoked and the agent's message buffer is swapped
    for the compacted version before the next step.

    Anthropic's `last_input_tokens` includes cache-read + cache-write
    tokens, which is intentional — the cache contributes to the prompt
    length the API actually sees.
    """

    def __init__(
        self,
        model: Model,
        *,
        threshold_ratio: float = 0.80,
        keep_first_n: int = 5,
        keep_last_n: int = 20,
        agent_name: str = "agent",
    ) -> None:
        self.model = model
        self.threshold_ratio = float(threshold_ratio)
        self.keep_first_n = int(keep_first_n)
        self.keep_last_n = int(keep_last_n)
        self.agent_name = agent_name
        self.events: list[CompactionEvent] = []

    @property
    def threshold_tokens(self) -> int:
        max_in = int(getattr(self.model, "max_input_tokens", 0) or 0)
        if max_in <= 0:
            return 0
        return int(max_in * self.threshold_ratio)

    def should_compact(self) -> bool:
        thr = self.threshold_tokens
        if thr <= 0:
            return False
        return int(getattr(self.model, "last_input_tokens", 0) or 0) >= thr

    def compact(self, messages: list[dict]) -> list[dict]:
        """Return a new messages list with the middle range replaced by a
        single synthetic user-role summary message. The input list is not
        mutated. If no safe truncation point exists (e.g. transcript is
        too short, or summarization fails), returns the input unchanged.
        """
        if not messages:
            return messages

        # Skip leading system/developer message (OpenAI/Gemini/OpenRouter
        # carry the system prompt as messages[0]; Anthropic stores it on
        # `model.system_prompt` and never mixes it into `messages`).
        conv_start = 0
        if messages[0].get("role") in {"system", "developer"}:
            conv_start = 1

        n = len(messages)

        # Boundary rule for the native tool_use harness: an assistant turn
        # carrying tool_calls MUST be immediately followed by the matching
        # tool_result(s) — Anthropic and OpenAI Responses both reject
        # orphan tool_use blocks. So head_end must NOT land right after
        # an assistant turn with unresolved tool_calls; everything else
        # is safe (`tool` result completes a cycle; clean assistant or
        # user-observation are natural turn boundaries).
        def _is_clean_head(idx: int) -> bool:
            if idx <= conv_start:
                return False
            prev = messages[idx - 1]
            role = prev.get("role")
            if role == "tool":
                return True
            if role == "assistant":
                return not (prev.get("tool_calls") or [])
            if role in ("user", "system", "developer"):
                return True
            return False

        target_head_end = min(conv_start + self.keep_first_n, n)
        head_end = target_head_end
        while head_end > conv_start and not _is_clean_head(head_end):
            head_end -= 1

        # tail_start must be an assistant turn — the synthetic bridge we
        # insert above it (see below) ends with a user message, and
        # user→assistant is the only universally-valid alternation. A
        # `tool` start would orphan its tool_result.
        def _is_clean_tail(idx: int) -> bool:
            if idx >= n or idx <= 0:
                return False
            return messages[idx].get("role") == "assistant"

        target_tail_start = max(n - self.keep_last_n, head_end)
        tail_start = target_tail_start
        while tail_start < n and not _is_clean_tail(tail_start):
            tail_start += 1
        # If forward walk reached the end (no assistant turn found in
        # the trailing window), fall back to a backward walk from the
        # target — keeps slightly more than `keep_last_n` messages but
        # still cuts the long middle. Without this fallback, agents
        # whose recent tail happens to end with tool-result + injected-
        # observation pairs would silently skip compaction every turn.
        if tail_start >= n:
            tail_start = target_tail_start
            while tail_start > head_end and not _is_clean_tail(tail_start):
                tail_start -= 1

        if head_end <= conv_start or tail_start >= n or tail_start <= head_end:
            # No valid middle to compact. Log so silent skips don't
            # masquerade as "everything's fine"; if this fires every
            # turn the agent's `last_input_tokens` will keep growing
            # past the threshold without any compaction happening.
            tail_roles = [messages[i].get("role") for i in range(max(0, n - 25), n)]
            print(
                f"[compactor:{self.agent_name}] no valid middle: "
                f"head_end={head_end}, tail_start={tail_start}, "
                f"conv_start={conv_start}, n={n}; "
                f"keep_first_n={self.keep_first_n}, keep_last_n={self.keep_last_n}; "
                f"trigger={int(getattr(self.model, 'last_input_tokens', 0) or 0)} tok; "
                f"last_25_roles={tail_roles}"
            )
            return messages

        middle = messages[head_end:tail_start]
        if not middle:
            return messages

        transcript = _render_transcript(middle)
        user_prompt = COMPACTION_USER_PROMPT_TEMPLATE.format(transcript=transcript)

        summarize = getattr(self.model, "summarize", None)
        if not callable(summarize):
            print(
                f"[compactor:{self.agent_name}] model {getattr(self.model, 'model', '?')} "
                f"has no summarize() — skipping compaction"
            )
            return messages

        try:
            summary_text = summarize(COMPACTION_SYSTEM_PROMPT, user_prompt)
        except Exception as exc:  # noqa: BLE001 — fall back to no-op on any failure
            print(
                f"[compactor:{self.agent_name}] summarize() failed "
                f"({type(exc).__name__}: {exc!s:.200s}); skipping compaction"
            )
            return messages

        if not summary_text or not summary_text.strip():
            print(f"[compactor:{self.agent_name}] empty summary — skipping compaction")
            return messages

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        synthetic = {
            "role": "user",
            "content": (
                "<context_summary>\n"
                f"{summary_text.strip()}\n"
                "</context_summary>\n\n"
                "[The block above is a system-generated factual summary of "
                "your earlier turns, produced because the running transcript "
                "exceeded the context window. Treat its contents as your own "
                "memory of those events. The next message resumes the live "
                "simulation.]"
            ),
            "timestamp": ts,
            "compaction": True,
            "compacted_msg_range": [head_end, tail_start],
        }

        # Strict user/assistant alternation: if the message before the
        # cut acts as a user turn at the API level (tool_result is
        # serialized as user-role for Anthropic; explicit user obs is
        # already user-role), insert a tiny assistant ack between it
        # and the synthetic user summary so we don't emit two
        # consecutive user turns.
        prev_role = messages[head_end - 1].get("role") if head_end > 0 else None
        bridge: list[dict] = []
        if prev_role in ("tool", "user"):
            bridge.append(
                {
                    "role": "assistant",
                    "content": "(context summary follows)",
                    "thinking": "",
                    "tool_calls": [],
                    "_raw": None,
                    "timestamp": ts,
                    "compaction_bridge": True,
                }
            )
        bridge.append(synthetic)

        ev = CompactionEvent(
            triggered_at_tokens=int(getattr(self.model, "last_input_tokens", 0) or 0),
            middle_start=head_end,
            middle_end=tail_start,
            middle_msg_count=len(middle),
            summary_chars=len(summary_text),
        )
        self.events.append(ev)
        print(
            f"[compactor:{self.agent_name}] compacted {len(middle)} msgs "
            f"(idx {head_end}..{tail_start}) into {len(summary_text)} chars "
            f"(trigger: {ev.triggered_at_tokens} tok, threshold: {self.threshold_tokens})"
        )

        return messages[:head_end] + bridge + messages[tail_start:]
