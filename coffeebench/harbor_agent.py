"""CoffeeTerminus — Harbor terminus-2 tuned for the `coffee` CLI (DESIGN §0).

Loaded by Harbor via `import_path: coffeebench.harbor_agent:CoffeeTerminus`.
This module imports Harbor at module level, so only the Harbor host process
imports it; nothing else in `coffeebench` does (the core package does not
depend on Harbor). Verified against harbor v0.24.0
(`agents/terminus_2/terminus_2.py`, `tmux_session.py`).

Overrides vs stock terminus-2:
- `_execute_commands`: keystrokes starting with `coffee` run in blocking mode
  (`tmux wait`, up to `coffee_timeout_sec`, default 7200 s) instead of the
  60 s-capped non-blocking mode, so waiting at the turn barrier costs no LLM
  calls. Before each episode: `tmux clear-history` + reset the incremental
  buffer (keeps capture-pane cheap over thousands of turns).
- tmux capture uses `-J` (join wrapped lines) so long JSON lines stay intact.
- `_limit_output_length`: 10 KB → `max_output_bytes` (default 100 KB; a
  `view_offers` observation can reach ~51 KB).
- `_handle_llm_interaction`: refuses `task_complete` while the simulation is
  still running for this agent (WARNINGS feedback), and POSTs telemetry
  (cost, tokens, analysis) to the env server after every LLM call.
- Prompt template (`coffeebench/harbor_templates/coffee-terminus-json.txt`,
  JSON parser only): terminus-json-plain with the same response spec, but a
  long-running-business framing, `coffee` commands blocking (duration
  ignored) instead of the duration/polling advice, and task_complete only
  after SIMULATION OVER. The completion confirmation and the 3-step
  summarization/handoff prompts (`_summarize`, copied from v0.24.0 with only
  the prompt texts changed) ask about the business state instead of a
  coding task.
- Exact resume (DESIGN §6): before every blocking `coffee` command it writes
  `$COFFEE_STATE_DIR/<agent>/call_<seq>.json` (Chat messages, the rest of the
  command batch, pane before the command, the episode's task_complete /
  feedback / pending-completion state) and a call marker for the CLI. When a
  `coffee` command returns `segment_end` (run paused) the trial ends at once,
  without another LLM call. In a resumed run, `run()` asks the server which
  call was parked at the checkpoint, restores that snapshot (Chat, tmux
  screen), re-sends the parked command in resume mode (the CLI prints the
  continuation of the parked call) plus the rest of the batch, builds the
  observation exactly as terminus-2 would, and continues the normal loop. No
  digest is shown then; without a snapshot it falls back to a fresh start
  with the server's digest. Cost/token counters restart per trial (the
  launcher sums segments).

The agent id comes from the task's `[environment].env` (`COFFEE_AGENT_ID`);
the server is reached over host TCP `COFFEE_HOST_URL` (host env, set by
launch_sim; the containers themselves use a Unix socket and have no network)
with the harness-scope token `COFFEE_HARNESS_TOKEN_<AID>` (host env only), so
telemetry / status / resume can only ever concern this agent.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import subprocess
import time
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2, Terminus2Options
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
)
from harbor.utils.env import resolve_env_vars

from coffeebench.server.agent_state import call_seq, call_sub, snapshot_path

import builtins as _builtins
import errno as _errno

import harbor.agents.terminus_2.terminus_2 as _t2_module


def _nofollow_open(file, mode="r", *args, **kwargs):
    """`open` for Terminus2's host-side log writes (trajectory*.json in the
    trial's agent dir, which Harbor bind-mounts read-write into the container
    as /logs/agent). v2.1 security fix: an agent could plant a symlink there
    (e.g. trajectory.json -> ../../../../../cluster/pbs_job.sh) and make the host
    overwrite arbitrary files. Writes never follow a symlink: a planted link
    is removed (the link itself, not its target) and the file is created."""
    if not any(c in mode for c in "wax+"):
        return _builtins.open(file, mode, *args, **kwargs)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_APPEND if "a" in mode else (os.O_EXCL if "x" in mode else os.O_TRUNC)
    if "+" in mode:
        flags = (flags & ~os.O_WRONLY) | os.O_RDWR
    for _ in range(2):
        try:
            fd = os.open(file, flags, 0o644)
            break
        except OSError as exc:
            if exc.errno != _errno.ELOOP or not os.path.islink(file):
                raise
            os.unlink(file)
    else:  # pragma: no cover
        raise OSError(_errno.ELOOP, "symlink re-created during write", str(file))
    return os.fdopen(fd, mode, *args, **kwargs)


_t2_module.open = _nofollow_open  # module-global shadow of the builtin for terminus_2 only

TASK_COMPLETE_WARNING = (
    "WARNINGS: task_complete was ignored — the CoffeeBench simulation is still "
    "running for you. Keep operating your business with `coffee <tool>` "
    "commands (use `coffee wait_for_next_day` to end your day). Mark the task "
    "complete only after a `coffee` command prints 'SIMULATION OVER'."
)

TEMPLATE_DIR = Path(__file__).resolve().parent / "harbor_templates"
JSON_TEMPLATE = TEMPLATE_DIR / "coffee-terminus-json.txt"
# mini-swe-agent action format (parser_name "mini_swe"): a short note + exactly one ```bash block per turn.
# Own wording on purpose: on 2026-10-27/28 OpenRouter content-filtered (0 tokens) both the verbatim terminus
# and mini-swe-agent templates for Claude Sonnet/Opus 5.5, and a "THOUGHT:" label for Opus 5.5.
MINI_SWE_TEMPLATE = TEMPLATE_DIR / "coffee-mini-swe.txt"
MINI_SWE_SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MINI_SWE_CMD_TIMEOUT_SEC = 300.0

SUMMARY_PROMPT = """You are about to hand off your CoffeeBench session to another AI agent, who will continue running your business. Please provide a comprehensive summary of the session so far.

Original instructions: {original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Financial position** - cash; receivables and payables (invoice ids, counterparties, amounts, due days, late fees).
2. **Inventory** - lots with item, quantity, unit cost and expiry day; inventory-cap headroom; pending production / roasts.
3. **Open market state** - your open listings, pending offers (received and made) and deals in flight, with ids, prices, quantities, payment terms and delivery days.
4. **Commitments and agreements** - everything agreed, promised or requested in messages, with counterparties and message / thread ids, and whether it has been fulfilled.
5. **Price and market observations** - listing and deal prices, consumer sales and market-aggregate data, counterparty behaviour.
6. **Current plan** - what you intend to do next and why.
7. **Workspace** - notes, scripts and other files in /workspace and what they contain.
8. **Current status** - the current simulated day and time, and your last `coffee` command and its result.

Be comprehensive and detailed, with exact ids and numbers. The next agent needs to understand everything that has happened so far in order to continue."""

QUESTION_PROMPT = """You are picking up a CoffeeBench session from a previous AI agent and will continue running the same business:

**Original instructions:** {original_instruction}

**Summary from Previous Agent:**
{summary}

**Current Terminal Screen:**
{current_screen}

Please begin by asking several questions (at least five, more if necessary)
about the current state of the business (finances, inventory, open listings /
offers / deals, commitments made in messages, market observations, plan,
/workspace files) that are not answered in the summary from the prior agent.
After you ask these questions you will be on your own, so ask everything you
need to know."""

HANDOFF_PROMPT = (
    "Here are the answers the other agent provided.\n\n{answers}\n\n"
    "Continue running the business from where the previous agent left off."
    " You can no longer ask questions. Please follow the spec to interact with "
    "the terminal."
)

MAX_REFUSED_COMPLETIONS = 20
END_LINE = "SIMULATION OVER — set task_complete true"
CALL_FILE = "/tmp/.coffee_call"
SNAPSHOT_VERSION = 1

# Context budget (DESIGN §4). The summary/handoff calls get the largest output
# cap shared by every official endpoint (Gemini 3.8 Flash: 65,536): at max
# effort the 32,768-token turn cap was used up by reasoning before the summary
# was written, so summarization failed and the context kept growing past the
# budget. A summary call that still runs out of output is retried down the
# effort ladder from the turn's own effort (e.g. max → xhigh → high → medium →
# low) and finally with reasoning off; without effort control (API default):
# default → off. Models with mandatory reasoning reject "off" (400) — then the
# oldest turns are dropped instead (_truncate_context).
SUMMARY_MAX_TOKENS = 65536
EFFORT_LADDER = ("max", "xhigh", "high", "medium", "low")
REASONING_OFF = {"enabled": False}
TRUNCATE_TARGET_FRAC = 0.5  # truncation keeps the newest turns up to this share of the budget
TRUNCATION_NOTE = (
    "[harness] Your conversation history reached the context limit and a summary "
    "could not be produced, so the oldest turns were removed. Your /workspace "
    "files are unchanged, and the `coffee` view tools show the current state of "
    "your business."
)
TRUNCATION_ACK = "Understood. I will continue running the business from the current state."


class SegmentPaused(Exception):
    """A `coffee` command returned segment_end: the run is paused; end the
    trial now (the resumed run continues from the snapshot of this call)."""


# Models copy the `; tmux wait -S done` suffix they see echoed in the pane into
# their own commands (16 agents in the seed-0 matrix). Extra signals on the
# shared `done` channel desynchronised harbor's blocking waits and caused 2 h
# hangs. So: strip any typed `tmux wait`, and block on a fresh channel per call.
# v2.1: only an actual `tmux wait -S|-L|-U <channel>` invocation, not the words in message text.
_TMUX_WAIT_RE = re.compile(r"\s*(?:;|&&)?\s*tmux\s+wait(?:-for)?\s+-[SLU]\s+[^\s;]+")


def _strip_tmux_wait(keystrokes: str) -> str:
    """Remove `tmux wait …` fragments the model typed itself, keeping the
    trailing newline that makes the line execute."""
    nl = keystrokes.endswith("\n")
    body = _TMUX_WAIT_RE.sub("", keystrokes.rstrip("\n")).rstrip()
    return body + ("\n" if nl else "")


_QUOTED_HEREDOC_RE = re.compile(r"<<-?\s*(['\"])(\w+)\1")
_UNQUOTED_HEREDOC_RE = re.compile(r"<<-?\s*([A-Za-z_]\w*)")


def _protect_dollar_amounts(keystrokes: str) -> str:
    """Escape `$<digit>` wherever bash would expand it (outside single quotes
    and outside quoted-heredoc bodies), so "@ $2.40" reaches the server
    verbatim instead of "@ .40". Models keep typing prices inside double
    quotes despite the CLI guide (seed-0: up to 158 of 171 DMs mangled)."""
    out: list[str] = []
    lines = keystrokes.split("\n")
    in_single = in_double = False
    heredoc_end: str | None = None
    unquoted_end: str | None = None
    for line in lines:
        if heredoc_end is not None:  # quoted heredoc body: literal
            out.append(line)
            if line.strip() == heredoc_end:
                heredoc_end = None
            continue
        if unquoted_end is not None:  # unquoted heredoc body: $ expands, quotes do not matter (v2.1)
            if line.strip() == unquoted_end:
                unquoted_end = None
                out.append(line)
            else:
                out.append(re.sub(r"(?<!\\)\$(?=\d)", r"\\$", line))
            continue
        buf = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and not in_single:
                buf.append(line[i:i + 2])
                i += 2
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "$" and not in_single and i + 1 < len(line) and line[i + 1].isdigit():
                buf.append("\\")
            buf.append(ch)
            i += 1
        out.append("".join(buf))
        m = _QUOTED_HEREDOC_RE.search(line)
        if m and not in_single:
            heredoc_end = m.group(2)
        elif not in_single and not in_double:
            mu = _UNQUOTED_HEREDOC_RE.search(line)
            if mu:
                unquoted_end = mu.group(1)
    return "\n".join(out)


def _group_multiline(keystrokes: str) -> str:
    """Wrap a multi-line `coffee` command (e.g. JSON via a quoted heredoc) in
    `{ ...\\n}` so tmux_session's blocking suffix `; tmux wait -S done` lands
    after the closing brace instead of on the heredoc terminator line, where
    bash would no longer recognise `EOF` and the call would hang."""
    body = keystrokes.rstrip("\n")
    if "\n" not in body or not keystrokes.endswith("\n"):
        return keystrokes
    return "{ " + body + "\n}\n"


_UNPARSEABLE_NOTE = (
    "\n[harness] NOT EXECUTED (nothing was sent to the shell): bash cannot parse "
    "this command, most likely unbalanced quotes (an apostrophe inside a "
    "single-quoted argument) or an unterminated heredoc:\n{command}\n"
    "Pass free text as JSON on stdin with a quoted heredoc, e.g.\n"
    "coffee send_message - <<'EOF'\n"
    '{{"recipient": "roaster_A", "title": "Offer", "body": "I\'ll sell 30 kg at $12/kg"}}\n'
    "EOF\n"
    "Commands after it in the same response were not run either.\n"
)


def _bash_parses(keystrokes: str) -> bool:
    """True if bash can parse the command (`bash -n`, nothing is executed).
    Unparseable or incomplete input (bad quoting, unterminated heredoc) must
    not be sent in blocking mode."""
    try:
        proc = subprocess.run(
            ["bash", "-n"], input=_group_multiline(keystrokes), text=True,
            capture_output=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True  # can't check: keep the normal blocking path
    return proc.returncode == 0


class CoffeeTerminusOptions(Terminus2Options):
    """Harbor validates agent kwargs against `options_model` (preflight)."""

    coffee_timeout_sec: float = Field(
        default=7200.0, description="Max wall time of one blocking `coffee` command."
    )
    max_output_bytes: int = Field(
        default=100_000, description="Terminal output cap per observation."
    )


# Provider hiccups (OpenRouter 429 "temporarily rate-limited upstream", 5xx,
# connection drops) must not end a trial: terminus-2 gives up after litellm's
# own short retries, and a dead background agent then idles for the rest of the
# run. Retry with backoff for up to LLM_RETRY_BUDGET_SEC (below the env's
# 30-min wake deadline, so the sim keeps moving meanwhile).
_TRANSIENT_LLM_ERRORS = (
    "RateLimitError", "ServiceUnavailableError", "APIConnectionError",
    "InternalServerError", "Timeout", "BadGatewayError", "APIError",
)
LLM_RETRY_BUDGET_SEC = 1500.0


_TRANSIENT_CODE_RE = re.compile(r"(?<![\d,.])(?:429|500|502|503|504)(?![\d,.])")


def _is_transient_llm_error(exc: BaseException) -> bool:
    """Retry provider hiccups only. v2.1: HTTP codes are matched as whole
    numbers (a context-length 400 quoting e.g. "142,900 tokens" contains the
    substring 429 and used to be retried for 25 minutes); context-length and
    bad-request errors are never transient."""
    name = type(exc).__name__
    if name == "EmptyLLMResponseError":
        return True
    if name in ("ContextLengthExceededError", "BadRequestError", "OutputLengthExceededError"):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 429, 500, 502, 503, 504)
    text = str(exc).lower()
    return any(k in name for k in _TRANSIENT_LLM_ERRORS) or bool(_TRANSIENT_CODE_RE.search(text)) or any(
        k in text for k in ("rate-limit", "rate limit", "temporarily", "overloaded")
    )


class MiniSweParser:
    """mini-swe-agent action format: a short free-text note followed by exactly one fenced bash block
    (one command, possibly multi-line, e.g. a heredoc). Returns terminus-2's ParseResult so the
    rest of the harness (blocking `coffee` calls, context budget, exact resume) is unchanged."""

    _BLOCK = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)\n?```", re.S)

    def parse_response(self, response: str):
        from harbor.agents.terminus_2.terminus_json_plain_parser import ParsedCommand, ParseResult

        blocks = self._BLOCK.findall(response or "")
        thought = (response or "").split("```", 1)[0].strip()
        thought = re.sub(r"^\s*(?:THOUGHT|Note):\s*", "", thought)
        if len(blocks) != 1:
            return ParseResult([], False, (
                f"Found {len(blocks)} bash code blocks; your response must contain EXACTLY ONE bash code "
                "block (```bash ... ```) with one command."), "", thought, "")
        cmd = blocks[0].strip()
        if not cmd:
            return ParseResult([], False, "The bash code block is empty.", "", thought, "")
        if cmd == MINI_SWE_SUBMIT:
            return ParseResult([], True, "", "", thought, "")
        return ParseResult([ParsedCommand(keystrokes=cmd + "\n", duration=1.0)], False, "", "", thought, "")


class EmptyLLMResponseError(Exception):
    """A provider returned no content and no tokens (seen 2026-10-27: an outage made every
    call return an empty message with 0 prompt / 0 completion tokens). Treated as a
    transient error (retried with backoff) instead of a normal, empty agent turn."""


def _is_empty_response(response) -> bool:
    if (getattr(response, "content", "") or "").strip() or (getattr(response, "reasoning_content", "") or "").strip():
        return False
    usage = getattr(response, "usage", None)
    toks = 0
    if usage is not None:
        toks = sum(int(getattr(usage, k, 0) or 0) for k in ("prompt_tokens", "completion_tokens"))
    return toks == 0


class CoffeeTerminus(Terminus2):
    options_model = CoffeeTerminusOptions
    options: CoffeeTerminusOptions

    def __init__(self, *args, **kwargs):
        self._max_output_bytes = 100_000  # used during super().__init__
        self._coffee: dict[str, str] = {}
        self._n_refused_completions = 0
        self._call_seq = 0
        # Counter -> a fresh tmux wait channel for every coffee call (unique within a
        # container; deterministic and snapshotted so exact resume sees identical screens).
        self._chan_n = 0
        self._episode_ctx: dict[str, Any] = {}
        self._state_dir: Path | None = None
        # Provider-reported / litellm-estimated prompt tokens of the last turn:
        # the context budget counts the model's own tokenizer, not tiktoken's.
        self._tok_ratio = 1.0
        self._n_truncations = 0
        super().__init__(*args, **kwargs)
        self._coffee_timeout_sec = float(self.options.coffee_timeout_sec)
        self._max_output_bytes = int(self.options.max_output_bytes)

    @staticmethod
    def name() -> str:
        return "coffee-terminus"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        env = resolve_env_vars(dict(environment.task_env_config.env or {}))
        agent = env.get("COFFEE_AGENT_ID", "")
        self._coffee = {
            "url": (os.environ.get("COFFEE_HOST_URL") or env.get("COFFEE_URL", "")).rstrip("/"),
            # Harness-scope token (telemetry / status / resume), host env only;
            # the container's agent token cannot reach those endpoints.
            "token": os.environ.get(f"COFFEE_HARNESS_TOKEN_{agent.upper()}")
            or env.get("COFFEE_TOKEN", ""),
            "agent": agent,
        }
        if os.environ.get("COFFEE_STATE_DIR") and self._coffee["agent"]:
            self._state_dir = Path(os.environ["COFFEE_STATE_DIR"])
        await super().setup(environment)
        # Join wrapped lines (`capture-pane -J`): compact JSON lines are longer
        # than the 160-col pane and would otherwise be split mid-token.
        orig_capture = self._session._tmux_capture_pane
        self._session._tmux_capture_pane = lambda capture_entire=False: orig_capture(
            capture_entire
        ).replace("capture-pane -p", "capture-pane -p -J", 1)

    # ----- prompts -----

    def _get_parser(self):
        if self.options.parser_name == "mini_swe":
            return MiniSweParser()
        return super()._get_parser()

    def _get_prompt_template_path(self) -> Path:
        if self.options.parser_name == "json":
            return JSON_TEMPLATE
        if self.options.parser_name == "mini_swe":
            return MINI_SWE_TEMPLATE
        return super()._get_prompt_template_path()

    def _get_error_response_type(self) -> str:
        if self.options.parser_name == "mini_swe":
            return "response"
        return super()._get_error_response_type()

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        if self.options.parser_name == "mini_swe":
            return (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure your part of the simulation has ended? It has ended only if a "
                "`coffee` command printed 'SIMULATION OVER'; after this you cannot act any more. "
                f"If so, reply with the command `{MINI_SWE_SUBMIT}` again."
            )
        if self.options.parser_name != "json":
            return super()._get_completion_confirmation_message(terminal_output)
        return (
            f"Current terminal state:\n{terminal_output}\n\n"
            "Are you sure you want to mark the task as complete? Your part of the "
            "simulation has ended only if a `coffee` command printed 'SIMULATION "
            "OVER'; after this you cannot act any more. If so, include "
            '"task_complete": true in your JSON response again.'
        )

    async def _summarize(
        self, chat: Chat, original_instruction: str, session: TmuxSession
    ) -> tuple[str, list[SubagentTrajectoryRef] | None]:
        """Terminus2._summarize (v0.24.0) with business-state prompt texts
        (SUMMARY_PROMPT / QUESTION_PROMPT / HANDOFF_PROMPT); the 3-step
        structure, message histories and trajectory bookkeeping are unchanged."""
        if len(chat.messages) == 0:
            return original_instruction, None

        self._summarization_count += 1
        subagent_trajectory_refs = []

        # ===== SUBAGENT 1: Summary Generation =====
        summary_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-summary"
        )
        steps_to_include = 1 + (len(chat.messages) - 1) // 2
        summary_steps, step_id_counter = self._prepare_copied_trajectory_steps(
            steps_to_include
        )
        summary_prompt = SUMMARY_PROMPT.format(original_instruction=original_instruction)
        summary_response, summary_trajectory_ref = await self._run_subagent(
            prompt=summary_prompt,
            message_history=chat.messages,
            steps=summary_steps,
            session_id=summary_session_id,
            agent_name="terminus-2-summarization-summary",
            filename_suffix="summary",
            summary_text=f"Context summarization {self._summarization_count}: Step 1 - Summary generation",
            subagent_name_for_logging="summary generation LLM call",
        )
        subagent_trajectory_refs.append(summary_trajectory_ref)

        # ===== SUBAGENT 2: Question Asking =====
        current_screen = await session.capture_pane(capture_entire=False)
        questions_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-questions"
        )
        questions_steps = []
        question_prompt = QUESTION_PROMPT.format(
            original_instruction=original_instruction,
            summary=summary_response.content,
            current_screen=current_screen,
        )
        questions_response, questions_trajectory_ref = await self._run_subagent(
            prompt=question_prompt,
            message_history=[],
            steps=questions_steps,
            session_id=questions_session_id,
            agent_name="terminus-2-summarization-questions",
            filename_suffix="questions",
            summary_text=f"Context summarization {self._summarization_count}: Step 2 - Question asking",
            subagent_name_for_logging="questions subagent",
        )
        model_questions = questions_response.content
        subagent_trajectory_refs.append(questions_trajectory_ref)

        # ===== SUBAGENT 3: Answer Providing =====
        answers_session_id = (
            f"{self._session_id}-summarization-{self._summarization_count}-answers"
        )
        answers_steps, step_id_counter = self._prepare_copied_trajectory_steps(
            steps_to_include
        )
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=summary_prompt,
                is_copied_context=True,
            )
        )
        step_id_counter += 1
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=summary_response.model_name or self._model_name,
                message=summary_response.content,
                reasoning_content=summary_response.reasoning_content,
                is_copied_context=True,
                extra={
                    "note": "Copied from summary subagent - metrics already recorded there"
                },
            )
        )
        step_id_counter += 1
        answer_request_prompt = (
            "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n"
            + model_questions
        )
        answers_message_history = chat.messages + [
            {"role": "user", "content": summary_prompt},
            {"role": "assistant", "content": summary_response.content},
        ]
        answers_response, answers_trajectory_ref = await self._run_subagent(
            prompt=answer_request_prompt,
            message_history=answers_message_history,
            steps=answers_steps,
            session_id=answers_session_id,
            agent_name="terminus-2-summarization-answers",
            filename_suffix="answers",
            summary_text=f"Context summarization {self._summarization_count}: Step 3 - Answer providing",
            subagent_name_for_logging="answers subagent",
        )
        subagent_trajectory_refs.append(answers_trajectory_ref)

        chat._messages = [
            chat.messages[0],
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": model_questions},
        ]
        chat.reset_response_chain()
        handoff_prompt = HANDOFF_PROMPT.format(answers=answers_response.content)
        return handoff_prompt, subagent_trajectory_refs

    # ----- context budget -----

    def _estimate_tokens(self, messages: list[dict]) -> int:
        from litellm.utils import token_counter

        return int(token_counter(model=self._model_name, messages=messages) * self._tok_ratio)

    def _count_total_tokens(self, chat: Chat) -> int:
        """Context size in the model's own tokens (litellm estimate scaled by
        the provider-reported ratio of the last turn); drives the proactive
        summarization check."""
        return self._estimate_tokens(chat.messages)

    def _calibrate_tokens(self, sent: list[dict], response) -> None:
        usage = getattr(response, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        if pt <= 0 or not sent:
            return
        from litellm.utils import token_counter

        est = token_counter(model=self._model_name, messages=sent)
        if est > 0 and 0.3 <= pt / est <= 4.0:  # ignore implausible usage reports
            self._tok_ratio = pt / est

    async def _check_proactive_summarization(self, chat, original_instruction, session):
        """Terminus2's check, but a failed summarization no longer leaves the
        context over budget (and retried on every turn): drop the oldest turns."""
        limit = self._llm.get_model_context_limit()
        if limit - self._count_total_tokens(chat) >= self.options.proactive_summarization_threshold:
            return None
        self.logger.debug("Proactively summarizing (context budget)")
        try:
            return await self._summarize(chat, original_instruction, session)
        except Exception as exc:  # noqa: BLE001 — any failure → deterministic fallback
            self.logger.error(f"Summarization failed, truncating oldest turns: {str(exc)[:300]}")
            self._truncate_context(chat, limit)
            return None

    def _truncate_context(self, chat: Chat, limit: int) -> None:
        """Keep the instruction (message 0) — and, after an earlier summarization,
        the summary hand-off turn (messages 1-2) — plus the newest whole turns that
        fit in TRUNCATE_TARGET_FRAC of the budget, with a note on the dropped
        turns. Roles stay strictly alternating (v2.1: the note no longer follows
        the instruction as a second user message, and the summary is kept)."""
        msgs = chat.messages
        summarized = (len(msgs) > 3 and msgs[1].get("role") == "user"
                      and str(msgs[1].get("content", "")).startswith("You are picking up a CoffeeBench session"))
        n_head = 3 if summarized else 1
        head = [dict(m) for m in msgs[:n_head]]
        if head[-1].get("role") == "user":
            head[-1]["content"] = f"{head[-1].get('content', '')}\n\n{TRUNCATION_NOTE}"
            head.append({"role": "assistant", "content": TRUNCATION_ACK})
        else:
            head += [{"role": "user", "content": TRUNCATION_NOTE},
                     {"role": "assistant", "content": TRUNCATION_ACK}]
        budget = int(limit * TRUNCATE_TARGET_FRAC) - self._estimate_tokens(head)
        # Cut only before a user message (whole turns); always keep the last turn.
        starts = [i for i in range(n_head, len(msgs)) if msgs[i].get("role") == "user"]
        start, used, end = len(msgs), 0, len(msgs)
        for i in reversed(starts):
            cost = self._estimate_tokens(msgs[i:end])
            if used + cost > budget and start < len(msgs):
                break
            used, start, end = used + cost, i, i
        chat._messages = head + msgs[start:]
        chat.reset_response_chain()
        self._n_truncations += 1

    def _summary_call_kwargs(self) -> list[dict]:
        """Kwargs for each summary attempt: the turn's own effort first, then
        every lower rung of EFFORT_LADDER, then reasoning off."""
        kw = json.loads(json.dumps(self._llm_call_kwargs))
        kw["max_tokens"] = SUMMARY_MAX_TOKENS
        extra = kw.get("extra_body")
        if isinstance(extra, str):
            extra = json.loads(extra)
            kw["extra_body"] = extra
        if not isinstance(extra, dict):
            extra = kw["extra_body"] = {}
        effort = (extra.get("reasoning") or {}).get("effort")
        rungs: list[dict | None] = [None]  # None = the turn's own setting
        if effort in EFFORT_LADDER:
            rungs += [{"effort": e} for e in EFFORT_LADDER[EFFORT_LADDER.index(effort) + 1:]]
        rungs.append(REASONING_OFF)
        tries = []
        for r in rungs:
            k = json.loads(json.dumps(kw))
            if r is not None:
                k["extra_body"]["reasoning"] = r
            tries.append(k)
        return tries

    async def _run_subagent(self, *args, **kwargs):
        """Summary / question / answer calls: SUMMARY_MAX_TOKENS, transient-error
        retries, and retries down the effort ladder while the output runs out."""
        from harbor.llms.base import OutputLengthExceededError
        from litellm.exceptions import BadRequestError

        steps = kwargs.get("steps") if "steps" in kwargs else args[2]
        steps0 = list(steps)
        base = self._llm_call_kwargs
        tries = self._summary_call_kwargs()
        async def attempt():
            steps[:] = steps0  # _run_subagent appends its prompt step before calling
            return await super(CoffeeTerminus, self)._run_subagent(*args, **kwargs)

        try:
            for i, kw in enumerate(tries):
                self._llm_call_kwargs = kw
                try:
                    return await self._with_llm_retry(attempt)
                except (OutputLengthExceededError, BadRequestError) as exc:
                    # BadRequest: an endpoint may reject one rung of the ladder.
                    if i == len(tries) - 1:
                        raise
                    nxt = tries[i + 1]["extra_body"].get("reasoning")
                    self.logger.warning(f"summary call failed ({type(exc).__name__}); retrying (reasoning {nxt})")
        finally:
            self._llm_call_kwargs = base

    # ----- terminal -----

    def _limit_output_length(self, output: str, max_bytes: int | None = None) -> str:
        return super()._limit_output_length(
            output, max_bytes=max_bytes or self._max_output_bytes
        )

    async def _reset_scrollback(self, session: TmuxSession) -> None:
        try:
            await session.environment.exec(
                f"tmux clear-history -t {session._session_name}", user=session._user
            )
            session._previous_buffer = await session.capture_pane(capture_entire=True)
        except Exception as exc:  # noqa: BLE001 — cosmetic; never fail the trial
            self.logger.debug(f"clear-history failed: {exc}")

    async def _send_coffee_blocking(self, session: TmuxSession, keystrokes: str,
                                    timeout_sec: float | None = None) -> None:
        """Type a coffee command and block until it exits, signalled on a fresh
        tmux wait channel (never harbor's shared `done`). A stale or copied
        signal on an old channel can then neither release nor stall a later
        call. Raises TimeoutError after coffee_timeout_sec."""
        self._chan_n += 1
        chan = f"cb{self._chan_n}"
        line = _group_multiline(keystrokes).rstrip("\n")
        # v2.1: the signal goes after a closing brace on its own line, so a trailing
        # `# comment` or `&` on the command can no longer swallow it (2 h hang).
        await session.send_keys([f"{{ {line}\n}}; tmux wait -S {chan}", "Enter"], block=False)
        limit = self._coffee_timeout_sec if timeout_sec is None else timeout_sec
        result = await session.environment.exec(f"timeout {limit:.0f}s tmux wait {chan}", user=session._user)
        if result.return_code != 0:
            raise TimeoutError(f"command timed out after {limit:.0f} s")

    async def _reset_continuation(self, session: TmuxSession) -> None:
        """If bash is waiting at its continuation prompt (an earlier shell
        command left a quote or heredoc open), send C-c so the next blocking
        `coffee` command starts at a fresh prompt instead of being swallowed."""
        try:
            pane = await session.capture_pane()
        except Exception as exc:  # noqa: BLE001 — best effort
            self.logger.debug(f"capture-pane failed: {exc}")
            return
        lines = [ln for ln in pane.splitlines() if ln.strip()]
        if lines and lines[-1].lstrip().startswith(">"):
            await session.send_keys(["C-c"], block=False, min_timeout_sec=0.5)

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
        resume_call: str | None = None,
    ) -> tuple[bool, str]:
        """`resume_call` (exact resume): the screen/buffer were restored from a
        snapshot; commands[0] is the parked `coffee` command, re-sent with the
        CLI in resume mode for that call id."""
        if resume_call is None:
            await self._reset_scrollback(session)
        for idx, command in enumerate(commands):
            if "tmux" in command.keystrokes:
                command = dataclasses.replace(command, keystrokes=_strip_tmux_wait(command.keystrokes))
            is_coffee = command.keystrokes.lstrip().startswith("coffee")
            mini = self.options.parser_name == "mini_swe"
            if is_coffee and "$" in command.keystrokes:
                command = dataclasses.replace(command, keystrokes=_protect_dollar_amounts(command.keystrokes))
            timeout_sec = (self._coffee_timeout_sec if is_coffee
                           else MINI_SWE_CMD_TIMEOUT_SEC if mini else command.duration_sec)
            if (is_coffee or mini) and not _bash_parses(command.keystrokes):
                # Never type a line bash cannot parse: with unbalanced quotes
                # bash sits at its `>` continuation prompt and swallows every
                # later command (seen in the cluster pilot), and in blocking mode
                # the `; tmux wait -S done` suffix never fires. Skip it and the
                # rest of this batch, and tell the model why.
                output = self._limit_output_length(await session.get_incremental_output())
                return False, output + _UNPARSEABLE_NOTE.format(command=command.keystrokes.strip())
            try:
                if is_coffee:
                    if resume_call is not None and idx == 0:
                        await self._write_marker(session, call_seq(resume_call), call_sub(resume_call))
                    else:
                        await self._reset_continuation(session)
                        if self._state_dir is not None:
                            self._call_seq += 1
                            await self._snapshot(commands[idx:], session)
                            await self._write_marker(session, self._call_seq, None)
                    await self._send_coffee_blocking(session, command.keystrokes)
                    await self._clear_marker(session)
                    if await self._segment_paused(session):
                        raise SegmentPaused()
                elif mini:
                    # mini-swe-agent semantics: every command runs to completion before the next turn.
                    await self._send_coffee_blocking(session, command.keystrokes, timeout_sec=MINI_SWE_CMD_TIMEOUT_SEC)
                else:
                    await session.send_keys(
                        command.keystrokes,
                        block=False,
                        min_timeout_sec=command.duration_sec,
                    )
            except TimeoutError:
                if mini and not is_coffee:
                    await session.send_keys(["C-c"], block=False, min_timeout_sec=0.5)  # stop the runaway command
                return True, self._timeout_template.format(
                    timeout_sec=timeout_sec,
                    command=command.keystrokes,
                    terminal_state=self._limit_output_length(
                        await session.get_incremental_output()
                    ),
                )
        return False, self._limit_output_length(await session.get_incremental_output())

    # ----- LLM turn -----

    async def _query_llm(self, chat, prompt, original_instruction="", session=None):
        sent = list(chat.messages) + [{"role": "user", "content": prompt}]
        n_before = len(chat.messages)

        async def attempt():
            response = await super(CoffeeTerminus, self)._query_llm(chat, prompt, original_instruction, session)
            if _is_empty_response(response):
                # Drop the empty exchange so the retry resends the same prompt.
                del chat.messages[n_before:]
                chat.reset_response_chain()
                raise EmptyLLMResponseError("empty LLM response (no content, 0 tokens)")
            return response

        response = await self._with_llm_retry(attempt)
        if chat.messages[: len(sent) - 1] == sent[:-1]:  # not rewritten by a fallback
            self._calibrate_tokens(sent, response)
        return response

    async def _with_llm_retry(self, call, delay: float = 15.0):
        """Run `call()` (an LLM request), retrying transient provider errors."""
        start = time.monotonic()
        while True:
            try:
                return await call()
            except Exception as exc:  # noqa: BLE001 — classified below
                if not _is_transient_llm_error(exc) or time.monotonic() - start > LLM_RETRY_BUDGET_SEC:
                    raise
                self.logger.warning(f"transient LLM error, retrying in {delay:.0f}s: {str(exc)[:200]}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 300.0)


    async def _handle_llm_interaction(self, chat, prompt, original_instruction="", session=None):
        (
            commands,
            is_task_complete,
            feedback,
            analysis,
            plan,
            llm_response,
        ) = await super()._handle_llm_interaction(chat, prompt, original_instruction, session)
        await self._post_telemetry(chat, analysis)
        if is_task_complete and not await self._sim_done():
            self._n_refused_completions += 1
            is_task_complete = False
            feedback = f"{feedback}\n{TASK_COMPLETE_WARNING}" if feedback else TASK_COMPLETE_WARNING
            # v2.1: never end the trial early (that looked like a crash and rolled
            # every agent back to the last checkpoint). After repeated refusals the
            # harness ends the agent's day for it, so the simulation keeps moving.
            if self._n_refused_completions > MAX_REFUSED_COMPLETIONS:
                commands = [Command(keystrokes="coffee wait_for_next_day\n", duration_sec=1.0)]
        self._episode_ctx = {"is_task_complete": bool(is_task_complete), "feedback": feedback or ""}
        return commands, is_task_complete, feedback, analysis, plan, llm_response

    # ----- exact resume -----

    async def _write_marker(self, session: TmuxSession, seq: int | None, resume_sub: int | None) -> None:
        if seq is None:
            return
        body = json.dumps({"seq": seq, "sub": 0, "resume": resume_sub})
        await session.environment.exec(f"printf '%s' '{body}' > {CALL_FILE}", user=session._user)

    async def _clear_marker(self, session: TmuxSession) -> None:
        if self._state_dir is not None:
            await session.environment.exec(f"rm -f {CALL_FILE}", user=session._user)

    async def _snapshot(self, rest: list[Command], session: TmuxSession) -> None:
        """Everything needed to continue this episode identically after a
        restart, taken right before the blocking `coffee` command rest[0]."""
        assert self._state_dir is not None
        data = {
            "version": SNAPSHOT_VERSION,
            "agent": self._coffee["agent"],
            "seq": self._call_seq,
            "chan_n": self._chan_n,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "messages": self._chat.messages if self._chat is not None else [],
            "commands": [{"keystrokes": c.keystrokes, "duration_sec": c.duration_sec} for c in rest],
            "pane_pre": await session.capture_pane(capture_entire=True),
            "prev_buffer": session._previous_buffer,
            "is_task_complete": self._episode_ctx.get("is_task_complete", False),
            "feedback": self._episode_ctx.get("feedback", ""),
            "pending_completion": self._pending_completion,
            "summarization_count": self._summarization_count,
            "n_refused_completions": self._n_refused_completions,
            "tok_ratio": self._tok_ratio,
            "n_truncations": self._n_truncations,
        }
        path = Path(snapshot_path(str(self._state_dir), self._coffee["agent"], self._call_seq))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        os.replace(tmp, path)

    async def _segment_paused(self, session: TmuxSession) -> bool:
        if not self._coffee.get("url"):
            return False
        try:
            tail = await session.capture_pane()
        except Exception:  # noqa: BLE001
            return False
        if END_LINE not in tail or '"status":"segment_end"' not in tail:
            return False
        status = await asyncio.to_thread(self._http, "GET", "/status")
        return bool(status) and status.get("sim_state") == "segment_end"

    async def _restore_screen(self, session: TmuxSession, pane: str) -> None:
        """Make the fresh tmux pane show exactly the snapshot's pane (incl. the
        prompt the parked command was typed at): readline clear-screen, drop
        the history, then write the text straight to the pane's tty (no
        command echo). Readline's cursor (after the prompt) matches."""
        env, user, name = session.environment, session._user, session._session_name
        res = await env.exec(f"tmux display-message -p -t {name} '#{{pane_tty}}'", user=user)
        tty = (res.stdout or "").strip()
        if not tty.startswith("/dev/"):
            raise RuntimeError(f"cannot find pane tty: {res.stdout!r} {res.stderr!r}")
        await session.send_keys(["C-l"], block=False, min_timeout_sec=0.5)
        await env.exec(f"tmux clear-history -t {name}", user=user)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\r\033[2K" + pane.rstrip("\n"))
            local = f.name
        try:
            await env.upload_file(local, "/tmp/.coffee_screen")
        finally:
            os.unlink(local)
        await env.exec(f"cat /tmp/.coffee_screen > {tty}; rm -f /tmp/.coffee_screen", user=user)
        await asyncio.sleep(0.3)

    async def _resume_snapshot(self) -> dict | None:
        info = await asyncio.to_thread(self._http, "GET", "/resume_info")
        if not info or not info.get("resumed"):
            return None
        call = info.get("pending_call")
        snap = None
        seq = call_seq(call)
        if self._state_dir is not None and seq is not None:
            path = Path(snapshot_path(str(self._state_dir), self._coffee["agent"], seq))
            try:
                snap = json.loads(path.read_text())
                snap["call"] = call
            except (OSError, ValueError) as exc:
                self.logger.warning(f"exact resume: snapshot {path} unusable: {exc}")
        mode = "exact" if snap is not None else "fallback"
        await asyncio.to_thread(self._http, "POST", "/resume_ack",
                                {"agent": self._coffee["agent"], "mode": mode})
        self.logger.info(f"resume at day {info.get('day')}: call={call} mode={mode}")
        return snap

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        snap = await self._resume_snapshot()
        try:
            if snap is None:
                await super().run(instruction, environment, context)
            else:
                await self._run_resumed(snap, instruction, context)
        except SegmentPaused:
            self.logger.info("segment_end: run paused; trial ends without another LLM call")

    async def _run_resumed(self, snap: dict, instruction: str, context: AgentContext) -> None:
        self._reset_per_run_state()
        self._chat = Chat(self._llm, interleaved_thinking=self.options.interleaved_thinking)
        self._chat._messages = list(snap["messages"])
        self._context = context
        self._summarization_count = int(snap.get("summarization_count") or 0)
        self._pending_completion = bool(snap.get("pending_completion"))
        self._n_refused_completions = int(snap.get("n_refused_completions") or 0)
        self._tok_ratio = float(snap.get("tok_ratio") or 1.0)
        self._n_truncations = int(snap.get("n_truncations") or 0)
        self._call_seq = int(snap["seq"])
        self._chan_n = int(snap.get("chan_n", 0))
        self._trajectory_steps.append(Step(
            step_id=1, timestamp=datetime.now(timezone.utc).isoformat(), source="system",
            message=f"Exact resume from snapshot call {snap['call']} "
                    f"({len(snap['messages'])} chat messages restored).",
        ))
        try:
            prompt = await self._resume_tail(snap)
            if prompt is not None:
                await self._run_agent_loop(
                    initial_prompt=prompt, chat=self._chat, original_instruction=instruction
                )
        finally:
            self._finalize_context(context)

    async def _resume_tail(self, snap: dict) -> str | None:
        """Finish the snapshot's episode as terminus-2's loop would have:
        run the rest of the batch (parked command in resume mode) and build
        the observation. Returns the next prompt (None = task confirmed done)."""
        session = self._session
        await self._restore_screen(session, snap["pane_pre"])
        session._previous_buffer = snap["prev_buffer"]
        cmds = [Command(keystrokes=c["keystrokes"], duration_sec=c["duration_sec"])
                for c in snap["commands"]]
        _, terminal_output = await self._execute_commands(cmds, session, resume_call=snap["call"])
        is_task_complete, feedback = bool(snap["is_task_complete"]), snap.get("feedback") or ""
        was_pending = self._pending_completion
        if is_task_complete:
            if self._pending_completion:
                observation = terminal_output
            else:
                self._pending_completion = True
                observation = self._get_completion_confirmation_message(terminal_output)
        else:
            self._pending_completion = False
            if feedback and "WARNINGS:" in feedback:
                observation = (
                    f"Previous response had warnings:\n{feedback}\n\n"
                    f"{self._limit_output_length(terminal_output)}"
                )
            else:
                observation = self._limit_output_length(terminal_output)
        self._trajectory_steps.append(Step(
            step_id=len(self._trajectory_steps) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(), source="system",
            message="Resumed command batch: " + " | ".join(c.keystrokes.strip()[:80] for c in cmds),
            observation=Observation(results=[ObservationResult(content=observation)]),
        ))
        self._dump_trajectory()
        if is_task_complete and was_pending:
            return None
        return observation

    def _finalize_context(self, context: AgentContext) -> None:
        """Same bookkeeping as the `finally` of Terminus2.run (v0.24.0)."""
        context.rollout_details = self._chat.rollout_details + self._subagent_rollout_details
        context.n_input_tokens = (
            self._chat.total_input_tokens + self._subagent_metrics.total_prompt_tokens
        )
        context.n_output_tokens = (
            self._chat.total_output_tokens + self._subagent_metrics.total_completion_tokens
        )
        context.n_cache_tokens = (
            self._chat.total_cache_tokens + self._subagent_metrics.total_cached_tokens
        )
        total_cost = self._chat.total_cost + self._subagent_metrics.total_cost_usd
        context.cost_usd = total_cost if total_cost > 0 else None
        context.metadata = {
            "n_episodes": self._n_episodes,
            "api_request_times_msec": self._api_request_times,
            "summarization_count": self._summarization_count,
            "context_truncations": self._n_truncations,
            "exact_resume": True,
        }
        if self.options.store_all_messages:
            context.metadata["all_messages"] = self._chat.messages
        self._dump_trajectory()

    # ----- server I/O (host side, stdlib HTTP in a worker thread) -----

    def _http(self, method: str, path: str, body: dict | None = None) -> dict | None:
        if not self._coffee.get("url"):
            return None
        req = urllib.request.Request(
            self._coffee["url"] + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Authorization": f"Bearer {self._coffee['token']}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode() or "{}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self.logger.debug(f"coffee server {method} {path} failed: {exc}")
            return None

    async def _sim_done(self) -> bool:
        """True when the sim is over for this agent (or the server is gone —
        never trap the agent in a loop it cannot finish)."""
        agent = self._coffee.get("agent", "")
        status = await asyncio.to_thread(self._http, "GET", f"/status?agent={agent}")
        if status is None:
            return True
        return bool(status.get("agent_done", True))

    async def _post_telemetry(self, chat, analysis: str) -> None:
        sub = self._subagent_metrics
        data: dict[str, Any] = {
            "agent": self._coffee.get("agent", ""),
            "cost_so_far": float(chat.total_cost + sub.total_cost_usd),
            "input_tokens": int(chat.total_input_tokens + sub.total_prompt_tokens),
            "output_tokens": int(chat.total_output_tokens + sub.total_completion_tokens),
            "cache_tokens": int(chat.total_cache_tokens + sub.total_cached_tokens),
            "n_calls": int(self._n_episodes),
            "summarizations": int(self._summarization_count),
            "context_truncations": int(self._n_truncations),
            "tok_ratio": round(self._tok_ratio, 3),
            "analysis": (analysis or "")[:4000],
        }
        await asyncio.to_thread(self._http, "POST", "/telemetry", data)
