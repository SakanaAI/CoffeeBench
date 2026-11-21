"""$0 end-to-end test of CoffeeTerminus: the real terminus-2 loop (tmux, parser,
blocking `coffee` commands, task_complete guard, telemetry) with the LLM call
replaced by a fixed script. Never contacts an LLM provider.

    uv run python harbor_adapter/launch_sim.py --config X.toml --scripted --network relay ...

Script per episode: `coffee start` → a premature task_complete (must be refused
by the guard) → a free shell command → guard tests → alternate
view_trial_balance / [shell; wait_for_next_day; shell] → once the terminal
shows SIMULATION OVER, task_complete twice (terminus asks for confirmation).
Every LLM input (history + prompt) is logged to
<agent logs>/scripted_prompts.jsonl (harbor_adapter/smoke_resume_exact.py
compares them between uninterrupted and restarted runs).
"""

from __future__ import annotations

import json
import os
import re

from harbor.llms.base import LLMResponse

from coffeebench.harbor_agent import CoffeeTerminus

END_LINE = re.compile(r"^SIMULATION OVER — set task_complete true$", re.M)


def _resp(keystrokes: list[str], complete: bool = False, note: str = "") -> LLMResponse:
    body = {
        "analysis": f"scripted: {note}",
        "plan": "scripted policy",
        "commands": [{"keystrokes": k, "duration": 1.0} for k in keystrokes],
        "task_complete": complete,
    }
    return LLMResponse(content=json.dumps(body), model_name="scripted")


class RateLimitError(Exception):
    """Stand-in with the same class name as litellm's (classified by name)."""


class ScriptedCoffeeTerminus(CoffeeTerminus):
    """Stateless script: the step is derived from the Chat history (number of
    earlier assistant turns), so an exactly resumed trial continues the same
    script. Every call's full input (history + prompt) is appended to
    <agent logs>/scripted_prompts.jsonl (exact-resume comparison)."""

    @staticmethod
    def name() -> str:
        return "scripted-coffee-terminus"

    async def _query_llm(self, chat, prompt, original_instruction="", session=None):
        return await self._with_llm_retry(lambda: self._scripted_call(chat, prompt), delay=0.5)

    async def _scripted_call(self, chat, prompt):
        # Fault injection for the $0 robustness tests (env vars, off by default):
        #   SCRIPTED_FLAKY_AGENT=<aid>: two transient 429s per call on steps 3-4 (must be retried).
        #   SCRIPTED_DIE_AGENT=<aid>, SCRIPTED_DIE_STEP=<n>: permanent error at step n (trial dies ->
        #   launcher must crash-restart). Only on the first attempt of the run (marker file).
        aid = self._coffee.get("agent", "")
        step = sum(1 for m in chat.messages if m.get("role") == "assistant")
        if os.environ.get("SCRIPTED_FLAKY_AGENT") == aid and step in (3, 4):
            self._flaky_n = getattr(self, "_flaky_n", 0) + 1
            if self._flaky_n % 3:
                raise RateLimitError("429 qwen is temporarily rate-limited upstream (injected)")
        marker = self.logs_dir.parents[3] / f".died_{aid}"
        if (os.environ.get("SCRIPTED_DIE_AGENT") == aid and step == int(os.environ.get("SCRIPTED_DIE_STEP", "5"))
                and not marker.exists()):
            marker.touch()
            raise RuntimeError("permanent provider failure (injected)")
        resp = self._script(chat, prompt)
        with open(self.logs_dir / "scripted_prompts.jsonl", "a") as f:
            f.write(json.dumps({"step": _step(chat), "messages": chat.messages, "prompt": prompt}) + "\n")
        # Mimic Chat.chat's history update (no LLM call).
        chat.messages.extend([{"role": "user", "content": prompt},
                              {"role": "assistant", "content": resp.content}])
        return resp

    def _script(self, chat, prompt: str) -> LLMResponse:
        step = _step(chat)
        # Only the CLI's own end line counts (the instruction and the guard
        # warning quote the phrase too).
        seen = [prompt] + [m["content"] for m in chat.messages[1:] if m.get("role") == "user"]
        if step > 0 and any(END_LINE.search(t) for t in seen):
            return _resp([], complete=True, note="sim over")
        if step == 0:
            return _resp(["coffee start\n"], note="start")
        if step == 1:
            return _resp([], complete=True, note="premature completion (guard test)")
        if step == 2:
            return _resp(["echo free-shell >> /workspace/note.txt; cat /workspace/note.txt\n"], note="shell")
        if step == 3:
            # Unbalanced quotes: must be refused (not typed), with the harness note.
            return _resp(["coffee send_message --recipient roaster_B --title 'x' --body 'I'll sell'\n",
                          "coffee view_dashboard\n"], note="unparseable (guard test)")
        if step == 4:
            # Leave bash at its `>` continuation prompt, then a coffee call must still run.
            return _resp(["echo 'unterminated\n", "coffee view_trial_balance\n"], note="continuation reset test")
        if step == 5:
            # Heredoc JSON with `$` and an apostrophe.
            return _resp(["coffee send_message - <<'EOF'\n"
                          '{"recipient": "roaster_B", "title": "t", "body": "I\'ll sell at $12/kg"}\n'
                          "EOF\n"], note="heredoc")
        if step in (6, 7, 8):
            # Models copy the echoed blocking suffix; it must be stripped and
            # must not desynchronise later blocking calls.
            return _resp(["coffee view_dashboard; tmux wait -S done; tmux wait -S done\n",
                          "coffee view_trial_balance; tmux wait -S done\n"], note="copied tmux wait")
        if step == 9 and os.environ.get("SCRIPTED_MECH"):
            # Mechanics check (test_rule_market --only harbor): role tools an
            # LLM seat would use with [mechanics] on.
            role = self._coffee.get("agent", "").split("_")[0]
            cmds = {"roaster": ["coffee buy_roaster_upgrade\n", "coffee view_market_aggregate --days_back 1\n"],
                    "farmer": ["coffee produce_item --item_id green_coffee_kg --quantity 10\n"],
                    "retailer": ["coffee view_market_aggregate --days_back 1\n"]}.get(role, [])
            return _resp(cmds or ["coffee view_trial_balance\n"], note="mechanics tools")
        if step % 2:
            return _resp(["coffee view_trial_balance\n"], note="view_trial_balance")
        # Shell work before AND after the blocking call in one batch, with the
        # workspace note visible (persistence across a restart shows up here).
        return _resp([f"echo step{step} >> /workspace/note.txt; tail -n 3 /workspace/note.txt\n",
                      "coffee wait_for_next_day\n",
                      "wc -l < /workspace/note.txt\n"], note="wait_for_next_day")


def _step(chat) -> int:
    return sum(1 for m in chat.messages if m.get("role") == "assistant")
