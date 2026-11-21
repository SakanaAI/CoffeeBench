"""Checkpoint / resume of a CoffeeBench simulation (DESIGN §6).

A checkpoint is taken by `Environment.checkpoint_hook` at the start of a
MORNING_OPEN bucket, before any of that day's mechanics: every agent is
parked between turns and no LLM call is in flight. It holds

    env             the pickled Environment (agents and hook excluded —
                    see `Environment._UNPICKLED`; Marketplace drops its RLock,
                    EventLogger keeps only path/offset/elapsed)
    agents          per-agent state: transcript (`messages`), scripted-model
                    state (PassiveModel / HeuristicRoasterModel internals) or,
                    for a RemoteAgent, its undelivered outbox + counters
    events_offset   byte size of run.events.jsonl at the checkpoint

Resume = build the run exactly as the original (`build_run(...,
open_events=False)` → fresh agents bound to fresh BusinessApps), then
`restore()` swaps in the unpickled env, rebinds every agent's tools to the
restored BusinessApps, restores the agent state, and reopens the events file
truncated to the checkpoint offset. `env.run(resume=True)` then continues
from the stored MORNING_OPEN; the trajectory is identical to an
uninterrupted run (modulo wall-clock timestamps).

Agents are never pickled: their tools are bound methods of BusinessApps and
LLM-backed models hold network clients. An in-process LLM `Agent` therefore
cannot be checkpointed (the Harbor server never has one).
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Callable

from coffeebench.event_logger import EventLogger
from coffeebench.models.heuristic_farmer_model import HeuristicFarmerModel
from coffeebench.models.heuristic_retailer_model import HeuristicRetailerModel
from coffeebench.models.heuristic_roaster_model import HeuristicRoasterModel
from coffeebench.models.heuristic_roaster_v2_model import HeuristicRoasterV2Model
from coffeebench.models.passive_model import PassiveModel

CKPT_VERSION = 1
SCRIPTED_MODEL_TYPES = (
    PassiveModel,
    HeuristicRoasterModel,
    HeuristicRoasterV2Model,
    HeuristicFarmerModel,
    HeuristicRetailerModel,
)


def agent_state(agent) -> dict:
    if hasattr(agent, "checkpoint_state"):  # RemoteAgent
        return agent.checkpoint_state()
    model = getattr(agent, "model", None)
    # `checkpointable = True` opts a scripted test policy in (state = vars()).
    if not isinstance(model, SCRIPTED_MODEL_TYPES) and not getattr(
        model, "checkpointable", False
    ):
        raise NotImplementedError(
            f"{agent.name}: in-process LLM agents ({type(model).__name__}) "
            "cannot be checkpointed; run them through the Harbor server."
        )
    return {"kind": "scripted", "messages": agent.messages, "model": vars(model)}


def restore_agent_state(agent, state: dict) -> None:
    if hasattr(agent, "restore_checkpoint_state"):
        agent.restore_checkpoint_state(state)
        return
    if state.get("kind") != "scripted":
        raise ValueError(f"{agent.name}: checkpoint holds a {state.get('kind')!r} agent")
    agent.messages = state["messages"]
    agent.model.__dict__.update(state["model"])


def snapshot(env, day: int, extra: dict | None = None) -> dict:
    lg = env.event_logger
    return {
        "version": CKPT_VERSION,
        "day": int(day),
        "max_days": env.max_days,
        "env": env,
        "agents": {aid: agent_state(a) for aid, a in env.agents.items()},
        "events_offset": lg.offset() if lg is not None else None,
        "events_elapsed_ms": lg.elapsed_ms() if lg is not None else 0,
        "extra": dict(extra or {}),
    }


def save(env, day: int, path: str, extra: dict | None = None) -> str:
    """Pickle a checkpoint atomically (tmp file + rename)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = pickle.dumps(snapshot(env, day, extra), protocol=pickle.HIGHEST_PROTOCOL)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load(path: str) -> dict:
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    if not isinstance(ckpt, dict) or ckpt.get("version") != CKPT_VERSION:
        raise ValueError(f"{path}: not a v{CKPT_VERSION} CoffeeBench checkpoint")
    return ckpt


def restore(fresh_env, ckpt: dict, events_path: str | None):
    """Return the checkpointed env wired to `fresh_env`'s agents (built by
    the resuming process with the same config/seed/models)."""
    env = ckpt["env"]
    if env.max_days != fresh_env.max_days:
        raise ValueError(
            f"max_days mismatch: checkpoint {env.max_days} vs run {fresh_env.max_days}"
        )
    if list(fresh_env.agents) != list(ckpt["agents"]):
        raise ValueError(
            f"agent set mismatch: checkpoint {list(ckpt['agents'])} vs run "
            f"{list(fresh_env.agents)}"
        )
    env.agents = fresh_env.agents
    env.verbose = fresh_env.verbose
    for aid, agent in env.agents.items():
        ba = env.business_apps[aid]
        agent.tools = [getattr(ba, t.__name__) for t in agent.tools]
        agent.tools_by_name = {t.__name__: t for t in agent.tools}
        restore_agent_state(agent, ckpt["agents"][aid])
    if events_path is not None and ckpt.get("events_offset") is not None:
        env.event_logger = EventLogger(
            events_path,
            truncate_to=ckpt["events_offset"],
            elapsed_ms=int(ckpt.get("events_elapsed_ms") or 0),
        )
    else:
        env.event_logger = None
    env._resume_day = int(ckpt["day"])
    env.stopped_at_day = None
    return env


def make_hook(
    ckpt_dir: str | None,
    stop_day: int | None = None,
    extra: Callable[[Any, int], dict] | None = None,
) -> Callable[[Any, int], bool]:
    """Hook for `env.checkpoint_hook`: write `<ckpt_dir>/day_NNN.pkl` at the
    start of every day; stop (return True) at `stop_day`."""

    def hook(env, day: int) -> bool:
        if ckpt_dir:
            save(env, day, os.path.join(ckpt_dir, f"day_{day:03d}.pkl"),
                 extra(env, day) if extra else None)
        return stop_day is not None and day >= stop_day

    return hook
