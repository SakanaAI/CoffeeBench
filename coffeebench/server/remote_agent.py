"""RemoteAgent — `Agent` stand-in whose decisions arrive over HTTP.

`Environment.run()` drives it exactly like an in-process `Agent`:

    init()                 → queue the initial observation
    add_message("user", …) → queue a wake observation
    step_query_async()     → flush (previous tool result + queued observations)
                             to the waiting CLI ticket, then await the next
                             action until the wake deadline
    step_apply(response)   → execute the tool via `Agent._execute_tool`

Ticket protocol (see `coffeebench.server.app`): every CLI call that waits for
the simulation gets a ticket; the ticket resolves with the payload flushed at
the agent's NEXT query (or with a terminal status: sim_over / bankrupt /
segment_end / aborted). At most one unresolved ticket per agent → 409 on
concurrent calls.

All methods run on the single asyncio loop shared by uvicorn and `env.run()`,
so no locking is needed.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any

from coffeebench.agent import Agent
from coffeebench.event_loop import format_min
from coffeebench.models.types import ModelResponse, ToolCall, ToolSpec

END_MESSAGE = "SIMULATION OVER — set task_complete true"
TERMINAL_STATUSES = ("sim_over", "bankrupt", "segment_end", "aborted")


class Conflict(Exception):
    """A second concurrent call while the agent's ticket is unresolved."""


class RemoteModelStub:
    """Duck-types `coffeebench.models.Model` for env bookkeeping; usage
    numbers are filled in from `POST /telemetry` (Harbor agent side)."""

    def __init__(self, label: str):
        self.model = label
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.system_prompt = ""

    def get_usage_stats(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_calls": self.n_calls,
            "cost": self.cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "last_input_tokens": self.last_input_tokens,
        }


class RemoteAgent(Agent):
    def __init__(
        self,
        name: str,
        tools: list,
        instruct_prompt: str,
        initial_observation: str,
        model_label: str = "remote",
        deadline_sec: float = 1800.0,
    ):
        # Skip Agent.__init__ (needs a real Model); keep the attributes
        # Environment and the inherited helpers rely on.
        self.name = name
        self.tools = tools
        self.tools_by_name = {t.__name__: t for t in tools}
        self.tool_specs = [ToolSpec.from_function(t) for t in tools]
        self.system_prompt = instruct_prompt
        self.user_prompt = initial_observation
        self.model = RemoteModelStub(f"remote({model_label})")
        self.messages: list[dict] = []
        self.compactor = None
        self.deadline_sec = float(deadline_sec)
        self.env = None  # set by the server (turn_timeout events, clock)

        self.registered = False
        self.ended: str | None = None
        self.n_turns = 0
        self.n_timeouts = 0
        self.n_conflicts = 0
        self.telemetry: dict[str, Any] = {}

        # Outbox, flushed at the next query / end.
        self._pending_result: dict | None = None
        self._pending_obs: list[str] = []
        self._pending_notes: list[str] = []
        # Tickets.
        self._ticket_seq = 0
        self._open_ticket: int | None = None
        self._open_future: asyncio.Future | None = None
        self._results: dict[int, dict] = {}
        self._last_payload: dict | None = None
        # Current query (env waiting for an action).
        self._awaiting: asyncio.Future | None = None
        self._undelivered = False
        self._deadline_at = 0.0
        self._last_thought = ""
        self._call_seq_id = 0
        # Resume (segment k>1): digest delivered once, with the first payload.
        self.resume_digest: str | None = None
        # Resume: tell the CLI (once) to download + extract its saved /workspace
        # (True, or "quiet" under exact resume: no note for the model).
        self.restore_workspace: bool | str = False
        # Exact resume (coffeebench.server.agent_state): id of the CLI call
        # parked on the open ticket, and the one restored from a checkpoint.
        self.pending_call: str | None = None
        self.resume_call: str | None = None
        # Workspace upload at checkpoints (requested through /wait).
        self.ws_request: int | None = None
        self._ws_done: asyncio.Event | None = None
        self._ws_wake: asyncio.Event | None = None
        self._ws_dropped: list[str] = []
        self.last_poll_at = 0.0
        # Usage accumulated in earlier segments; telemetry reports per-trial
        # totals, so the stub adds these on top.
        self._usage_base: dict[str, float] = {}

    # ----- checkpoint / resume (coffeebench.server.checkpoint) -----

    _USAGE_KEYS = ("cost", "n_calls", "total_input_tokens", "total_output_tokens")

    def checkpoint_state(self) -> dict:
        """State that must survive a segment boundary. The outbox holds the
        last tool result + observations not yet delivered to the CLI (at a
        segment end they are ALSO sent with the segment_end payload; the
        resumed agent gets them again with its first payload)."""
        return {
            "kind": "remote",
            "messages": self.messages,
            "outbox": {
                "result": self._pending_result,
                "observations": list(self._pending_obs),
                "notes": list(self._pending_notes),
            },
            "counters": {
                "n_turns": self.n_turns,
                "n_timeouts": self.n_timeouts,
                "n_conflicts": self.n_conflicts,
                "call_seq_id": self._call_seq_id,
                "ticket_seq": self._ticket_seq,
                "open_ticket": self._open_ticket,
            },
            "pending_call": self.pending_call if self._open_ticket is not None else None,
            "ws_dropped": list(self._ws_dropped),
            "usage": {k: getattr(self.model, k) for k in self._USAGE_KEYS},
        }

    def restore_checkpoint_state(self, state: dict) -> None:
        if state.get("kind") != "remote":
            raise ValueError(
                f"{self.name}: checkpoint holds a {state.get('kind')!r} agent, "
                "not a remote one (the remote set must not change across segments)"
            )
        self.messages = state["messages"]
        box = state["outbox"]
        self._pending_result = box["result"]
        self._pending_obs = list(box["observations"])
        self._pending_notes = list(box["notes"])
        c = state["counters"]
        self.n_turns = int(c["n_turns"])
        self.n_timeouts = int(c["n_timeouts"])
        self.n_conflicts = int(c["n_conflicts"])
        self._call_seq_id = int(c["call_seq_id"])
        # Ticket numbers continue as without the restart: the resumed parked
        # call re-registers and gets its original ticket number back.
        open_ticket = c.get("open_ticket")
        self._ticket_seq = int(open_ticket) - 1 if open_ticket else int(c.get("ticket_seq") or 0)
        self.resume_call = state.get("pending_call")
        self._ws_dropped = list(state.get("ws_dropped") or [])
        self._usage_base = dict(state["usage"])
        for k, v in self._usage_base.items():
            setattr(self.model, k, v)

    # ----- Agent interface used by Environment -----

    def init(self) -> None:
        self.add_message("user", self.user_prompt)

    def add_message(self, role: str, content: str = "", **kwargs) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                **kwargs,
            }
        )
        if role == "user":
            self._pending_obs.append(content)

    async def step_query_async(self) -> ModelResponse:
        loop = asyncio.get_running_loop()
        if self.ended:
            return ModelResponse(content=f"(agent ended: {self.ended})")
        self._awaiting = loop.create_future()
        self._deadline_at = loop.time() + self.deadline_sec
        self._undelivered = not self._flush()
        try:
            while not self._awaiting.done():
                remaining = self._deadline_at - loop.time()
                if remaining <= 0:
                    return self._timeout()
                await asyncio.wait({self._awaiting}, timeout=remaining)
            tool, args = self._awaiting.result()
        finally:
            self._awaiting = None
        self._call_seq_id += 1
        return ModelResponse(
            content="",
            thinking=self._last_thought,
            tool_calls=[ToolCall(id=f"cli_{self._call_seq_id:05d}", name=tool, input=args)],
            stop_reason="tool_use",
        )

    def step_query(self) -> ModelResponse:  # pragma: no cover - sync path unused
        raise RuntimeError("RemoteAgent only supports step_query_async")

    def step_apply(self, response: ModelResponse) -> dict:
        thought = response.thinking or ""
        self._last_thought = ""
        if not response.tool_calls:
            # Wake-deadline timeout (or ended agent): idle for the day.
            result = {
                "status": "turn_timeout",
                "message": (
                    f"No action within {int(self.deadline_sec)} s of your "
                    "observation; your turn was treated as wait_for_next_day."
                ),
            }
            self.messages.append({"role": "assistant", "content": response.content})
            self._pending_result = {
                "tool": "wait_for_next_day",
                "args": {},
                "result": result,
            }
            return {
                "action_name": "wait_for_next_day",
                "action_input": {},
                "observation": result,
                "thought": response.content,
            }
        call = response.tool_calls[0]
        result = self._execute_tool(call)
        self.n_turns += 1
        self.messages.append(
            {
                "role": "assistant",
                "content": thought,
                "tool_calls": [call],
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(result, default=str),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            }
        )
        self._pending_result = {"tool": call.name, "args": call.input, "result": result}
        return {
            "action_name": call.name,
            "action_input": call.input,
            "observation": result,
            "thought": thought,
        }

    def on_end(self, reason: str) -> None:
        """Terminal status (sim_over / bankrupt / segment_end / aborted).
        Idempotent; the first reason wins."""
        if self.ended:
            return
        self.ended = reason
        if self._open_ticket is not None:
            self._resolve(self._build_payload(status=reason))

    # ----- HTTP-facing API (called by app.py) -----

    def register(self, call: str | None = None) -> int:
        """`coffee start` (also the exact-resume continuation). Returns a
        ticket resolving with the first observation (or the current state if
        already started)."""
        ticket = self._register()
        if ticket == self._open_ticket:
            self.pending_call = call
        return ticket

    def _register(self) -> int:
        self.registered = True
        if self._open_ticket is not None:
            return self._open_ticket
        if self.ended:
            return self._new_resolved(self._build_payload(status=self.ended))
        if self._awaiting is not None and self._undelivered:
            return self._deliver_now(note=None)
        if self._last_payload is not None:
            payload = dict(self._last_payload)
            payload["notes"] = list(payload.get("notes", [])) + [
                "Already started; this is your latest observation again. "
                "Act with `coffee <tool>`."
            ]
            return self._new_resolved(payload)
        return self._new_open()

    def submit(self, tool: str, args: dict, call: str | None = None) -> int:
        """`coffee <tool> …`. Returns a ticket; raises Conflict (409)."""
        ticket = self._submit(tool, args)
        if ticket == self._open_ticket:
            self.pending_call = call
        return ticket

    def _submit(self, tool: str, args: dict) -> int:
        if self.ended:
            return self._new_resolved(self._build_payload(status=self.ended))
        if self._open_ticket is not None:
            self.n_conflicts += 1
            raise Conflict(
                f"ticket {self._open_ticket} is still pending; run `coffee wait` "
                "and never run two `coffee` commands concurrently"
            )
        if self._awaiting is not None and not self._awaiting.done():
            if self._undelivered:
                # The agent has not seen its current observation (it timed
                # out earlier and is only now acting): show it, don't execute.
                return self._deliver_now(
                    note=f"Your action `{tool}` was NOT executed: your previous "
                    "turn timed out and new observations arrived. Read them and act again."
                )
            ticket = self._new_open()
            self._awaiting.set_result((tool, dict(args or {})))
            return ticket
        # Not being queried (parked after a timeout): hold until next wake.
        self._pending_notes.append(
            f"Your action `{tool}` was NOT executed: it was not your turn "
            "(your previous turn had timed out, or the simulation had not "
            "reached you yet). Read the observations and act again."
        )
        return self._new_open()

    async def wait_ticket(self, ticket: int | None, timeout: float, call: str | None = None) -> dict:
        if ticket is None:
            ticket = self._open_ticket if self._open_ticket is not None else self._ticket_seq
        if ticket == self._open_ticket:
            self.pending_call = call  # the latest CLI call parked on it
        self.last_poll_at = time.time()
        if ticket in self._results:
            return self._results[ticket]
        if ticket != self._open_ticket or self._open_future is None:
            return {"status": "unknown_ticket", "ticket": ticket}
        fut = self._open_future
        if self.ws_request is None:
            self._ws_wake = asyncio.Event()
            waker = asyncio.ensure_future(self._ws_wake.wait())
            try:
                await asyncio.wait({fut, waker}, timeout=max(0.0, timeout),
                                   return_when=asyncio.FIRST_COMPLETED)
            finally:
                waker.cancel()
            self.last_poll_at = time.time()
        if fut.done():
            return fut.result()
        if self.ws_request is not None:
            return {"status": "pending", "ticket": ticket, "upload_workspace": self.ws_request}
        return {"status": "pending", "ticket": ticket}

    def request_workspace(self, day: int) -> asyncio.Event:
        """Checkpoint at `day`: the parked CLI's next /wait answer asks it to
        upload /workspace (a running long-poll is woken)."""
        self.ws_request = day
        self._ws_done = asyncio.Event()
        if self._ws_wake is not None:
            self._ws_wake.set()
        return self._ws_done

    def workspace_uploaded(self, dropped: list[str]) -> None:
        """Upload received. A changed over-cap drop list is reported to the
        agent with its next observation (deterministic: part of the outbox)."""
        self.ws_request = None
        if self._ws_done is not None:
            self._ws_done.set()
        if dropped and dropped != self._ws_dropped:
            shown = ", ".join(dropped[:10]) + (f" (+{len(dropped) - 10} more)" if len(dropped) > 10 else "")
            self._pending_notes.append(
                "Your /workspace exceeds the 20 MB cap: these files would NOT survive a pause "
                f"of the run: {shown}"
            )
        self._ws_dropped = list(dropped)

    def is_done(self) -> bool:
        return self.ended is not None

    def note_telemetry(self, data: dict) -> None:
        self.telemetry = dict(data)
        m, base = self.model, self._usage_base
        for attr, key, cast in (
            ("cost", "cost_so_far", float),
            ("n_calls", "n_calls", int),
            ("total_input_tokens", "input_tokens", int),
            ("total_output_tokens", "output_tokens", int),
        ):
            if data.get(key):
                setattr(m, attr, cast(base.get(attr, 0)) + cast(data[key]))
        m.last_input_tokens = int(data.get("last_input_tokens") or m.last_input_tokens)
        if data.get("analysis"):
            self._last_thought = str(data["analysis"])[:4000]

    def state(self) -> str:
        if self.ended:
            return self.ended
        if not self.registered:
            return "unregistered"
        if self._awaiting is not None:
            return "awaiting_action" if not self._undelivered else "observation_pending"
        return "waiting_for_turn"

    # ----- internals -----

    def _timeout(self) -> ModelResponse:
        self.n_timeouts += 1
        self._undelivered = False
        if self.env is not None:
            self.env._emit(
                "turn_timeout",
                agent_id=self.name,
                day=self.env.time_manager.get_current_day(),
                at=self.env.time_manager.get_virtual_min(),
                deadline_sec=self.deadline_sec,
            )
        print(f"[server] {self.name}: turn_timeout after {self.deadline_sec:.0f}s")
        return ModelResponse(
            content=f"(turn_timeout: no action within {int(self.deadline_sec)} s)"
        )

    def _now_str(self) -> str | None:
        if self.env is None:
            return None
        return format_min(self.env.time_manager.get_virtual_min())

    def _build_payload(self, status: str = "ok") -> dict:
        payload: dict[str, Any] = {"status": status, "time": self._now_str()}
        if self.resume_digest:
            payload["digest"] = self.resume_digest
            self.resume_digest = None
        if self.restore_workspace:
            payload["workspace_restore"] = self.restore_workspace
            self.restore_workspace = False
        if self._pending_result is not None:
            payload["tool"] = self._pending_result["tool"]
            payload["result"] = self._pending_result["result"]
        payload["observations"] = list(self._pending_obs)
        payload["notes"] = list(self._pending_notes)
        if status in TERMINAL_STATUSES:
            payload["message"] = END_MESSAGE
        self._pending_result = None
        self._pending_obs = []
        self._pending_notes = []
        return payload

    def _flush(self) -> bool:
        """Deliver the outbox to the open ticket. False if no CLI waiting."""
        if self._open_ticket is None:
            return False
        self._resolve(self._build_payload())
        return True

    def _deliver_now(self, note: str | None) -> int:
        payload = self._build_payload()
        if note:
            payload["notes"].append(note)
        self._undelivered = False
        loop = asyncio.get_running_loop()
        self._deadline_at = loop.time() + self.deadline_sec
        return self._new_resolved(payload)

    def _new_open(self) -> int:
        self._ticket_seq += 1
        self._open_ticket = self._ticket_seq
        self._open_future = asyncio.get_running_loop().create_future()
        return self._open_ticket

    def _new_resolved(self, payload: dict) -> int:
        self._ticket_seq += 1
        self._store(self._ticket_seq, payload)
        return self._ticket_seq

    def _resolve(self, payload: dict) -> None:
        ticket, fut = self._open_ticket, self._open_future
        self._open_ticket, self._open_future = None, None
        self.pending_call = None
        self._store(ticket, payload)
        if fut is not None and not fut.done():
            fut.set_result(self._results[ticket])

    def _store(self, ticket: int, payload: dict) -> None:
        payload = dict(payload)
        payload["ticket"] = ticket
        payload["agent"] = self.name
        self._results[ticket] = payload
        if payload.get("status") == "ok":
            self._last_payload = payload
        for old in [t for t in self._results if t < ticket - 8]:
            del self._results[old]
