"""Load / resume / branch a CoffeeBench run from a versioned checkpoint.

`Environment.save_checkpoint` writes the on-disk format; this module reads it
back, reconstructs a fresh `Environment` (with brand-new model clients, agent
objects, marketplace, event loop and RNG — nothing shared with the process that
wrote the checkpoint), overlays the saved economic state, reconnects all runtime
callbacks, and validates the result.

The same checkpoint file can be loaded any number of times into independent
environments (in one process or across processes). Each load produces its own
mutable state; branches never share model clients, agents, message lists, the
marketplace, the event loop, RNG, or (later) Valto state.

Public API:

    env = load_checkpoint(path, model_overrides={"roaster_A": "gpt-5.5"},
                          branch_name="gpt_variant", events_path="…jsonl")
    env.run()  # continues from the checkpoint's day through max_days
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from coffeebench.environment import CHECKPOINT_SCHEMA_VERSION, Environment
from coffeebench.event_logger import EventLogger
from coffeebench.typings import (
    Deal,
    Invoice,
    Item,
    JournalEntry,
    Listing,
    Message,
    Offer,
)


class CheckpointVersionError(RuntimeError):
    """Raised when a checkpoint's schema version is incompatible."""


class CheckpointValidationError(RuntimeError):
    """Raised when a restored env fails post-load wiring validation."""


def _rebuild_rng_state(raw) -> tuple:
    """Reconstruct the exact tuple `random.setstate` expects from the
    JSON-decoded form (lists instead of tuples)."""
    version, internal, gauss_next = raw[0], raw[1], raw[2]
    return (int(version), tuple(int(x) for x in internal), gauss_next)


def _synthesize_args(data: dict, model_overrides: Optional[dict]) -> SimpleNamespace:
    """Build an argparse-like namespace so we can reuse `build_run` to
    construct the world skeleton (items, endowments, business apps, agents,
    marketplace, env) exactly as a fresh run would, then overlay saved state.

    The per-agent model map is taken from the checkpoint and layered with any
    CLI `model_overrides`, so a branch can swap a model without disturbing the
    saved economic state.
    """
    run_meta = data.get("run_meta", {}) or {}
    saved_models = {aid: st.get("model") for aid, st in data.get("agents", {}).items()}
    merged = dict(saved_models)
    for aid, m in (model_overrides or {}).items():
        merged[aid] = m
    models_str = ",".join(f"{aid}:{m}" for aid, m in merged.items() if m)

    config_source = run_meta.get("config_source")
    # Only pass the config through if it still exists on disk; otherwise the
    # checkpoint reconstructs from its own stored settings (portable path).
    config_arg = None
    if config_source:
        from pathlib import Path

        if Path(config_source).exists():
            config_arg = config_source

    return SimpleNamespace(
        config=config_arg,
        model=None,
        models=models_str,
        max_days=int(data["max_days"]),
        seed=run_meta.get("seed"),
        main_agent=data.get("main_agent_id"),
        output=None,
    )


def load_checkpoint(
    path: str,
    model_overrides: Optional[dict] = None,
    branch_name: Optional[str] = None,
    events_path: Optional[str] = None,
    valto: Optional[dict] = None,
    verbose: bool = True,
) -> Environment:
    """Reconstruct an `Environment` positioned to resume from `path`.

    Args:
        path: checkpoint JSON written by `Environment.save_checkpoint`.
        model_overrides: {agent_id: model_name} to swap selected agents'
            models for this branch (economic state is preserved).
        branch_name: label recorded in run_meta and the events stream.
        events_path: where this branch writes its live event stream. Defaults
            to a branch-specific file so branches never clobber each other.
        valto: explicit Valto governance spec for THIS branch (wins over the
            checkpoint's original run). `{"enabled": True, "actor": ...,
            "mode": "governed"|"observe", "policy": path, "govern_tools": [...]}`
            governs the fork; `{}` / `{"enabled": False}` / None-with-no-branch
            leaves it ungoverned. A branch-specific `db_path` is derived from
            `branch_name` when not given, so forks never share a ledger.
        verbose: env verbosity.
    """
    import json

    from coffeebench.main import build_run

    with open(path) as f:
        data = json.load(f)

    version = data.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointVersionError(
            f"Checkpoint schema version {version} != supported "
            f"{CHECKPOINT_SCHEMA_VERSION}. Refusing to load."
        )

    # Resolve branch-specific Valto governance. Default a per-branch ledger path
    # next to the checkpoint so parallel forks never write to the same DB.
    valto_override = dict(valto) if valto is not None else {"enabled": False}
    if valto_override.get("enabled") and not valto_override.get("db_path"):
        ckpt_dir = Path(path).resolve().parent
        tag = branch_name or "branch"
        valto_override["db_path"] = str(ckpt_dir / f"valto_ledger_{tag}.db")

    # --- 1. Build a fresh world skeleton (fresh model clients, agents, etc.) ---
    args = _synthesize_args(data, model_overrides)
    env, _output_path, _events_path = build_run(args, valto_override=valto_override)
    env.verbose = verbose

    # Replace the event logger with a branch-specific one so concurrent
    # branches don't truncate each other's stream. build_run already opened one;
    # close it before swapping to avoid a dangling handle.
    if env.event_logger is not None:
        env.event_logger.close()
    if events_path is not None:
        env.event_logger = EventLogger(events_path)
    else:
        env.event_logger = None

    # --- 2. Overlay saved state onto the skeleton ---
    _restore_clock_and_scheduler(env, data)
    _restore_rng(env, data)
    _restore_marketplace(env, data)
    invoice_pool = _restore_invoices(data)
    _restore_agents(env, data, invoice_pool, model_overrides or {})
    _restore_pending(env, data)
    _restore_accounting(env, data)

    # --- 3. Run identity + resume flags ---
    env._main_agent_id = data.get("main_agent_id")
    env._terminated_early = data.get("terminated_early")
    env._run_meta = dict(data.get("run_meta", {}) or {})
    env._run_meta["branch_name"] = branch_name
    env._run_meta["parent_checkpoint"] = path
    env._is_resume = True

    # --- 4. Validate wiring before handing back a runnable env ---
    validate_after_load(env)
    if verbose:
        print(
            f"[checkpoint] Loaded '{path}' → resume at day "
            f"{env.time_manager.get_current_day()} "
            f"(branch={branch_name}, overrides={model_overrides or {}})"
        )
    return env


# ---------------------------------------------------------------------------
# Restore helpers
# ---------------------------------------------------------------------------
def _restore_clock_and_scheduler(env: Environment, data: dict) -> None:
    env.time_manager.virtual_min = int(data["next_virtual_min"])
    env.event_loop.restore_state(data["event_loop"])
    env._available_at = dict(data.get("available_at", {}))
    env._pending_wake_reason = dict(data.get("pending_wake_reason", {}))
    env._last_active_day_per_agent = defaultdict(
        lambda: -1,
        {k: int(v) for k, v in data.get("last_active_day_per_agent", {}).items()},
    )
    env._eod_done_for_day = set(int(d) for d in data.get("eod_done_for_day", []))
    env._today_nudge = data.get("today_nudge")
    env.step_count = int(data.get("step_count", 0))


def _restore_rng(env: Environment, data: dict) -> None:
    env.rng = random.Random()
    env.rng.setstate(_rebuild_rng_state(data["rng_state"]))
    env._demand_path = {k: list(v) for k, v in data.get("demand_path", {}).items()}
    env._retailer_loyalty = dict(data.get("retailer_loyalty", {}) or {})
    env._spoilage_carry = defaultdict(
        lambda: defaultdict(float),
        {
            aid: defaultdict(float, {i: float(c) for i, c in carry.items()})
            for aid, carry in data.get("spoilage_carry", {}).items()
        },
    )


def _restore_marketplace(env: Environment, data: dict) -> None:
    mkt = data["marketplace"]
    env.marketplace.items = {
        iid: Item(**idict) for iid, idict in mkt["items"].items()
    }
    env.marketplace.listings = [Listing(**d) for d in mkt["listings"]]
    env.marketplace.offers = [Offer(**d) for d in mkt["offers"]]
    # Deals form a shared pool: marketplace.deals holds the canonical objects,
    # and env._pending_shipments references a subset by id (restored later).
    env.marketplace.deals = [Deal(**d) for d in mkt["deals"]]
    env.marketplace.messages = [Message(**d) for d in mkt["messages"]]
    env.marketplace._read_message_ids = {
        aid: set(ids) for aid, ids in mkt.get("read_message_ids", {}).items()
    }
    env.id_gen.restore(mkt.get("id_gen", {}))
    # env.id_gen is the same object as env.marketplace.id_gen (shared in
    # Environment.__init__), so the counter is restored for both.


def _restore_invoices(data: dict) -> dict[str, Invoice]:
    """Rebuild the shared invoice pool ONCE, keyed by id, so the same object
    can be placed in both the issuer's AR and the payer's AP (preserving the
    aliasing the live env relies on)."""
    return {iid: Invoice(**idict) for iid, idict in data.get("invoices", {}).items()}


def _restore_agents(
    env: Environment,
    data: dict,
    invoice_pool: dict[str, Invoice],
    model_overrides: dict,
) -> None:
    from coffeebench.models import is_scripted_model

    for aid, st in data["agents"].items():
        ba = env.business_apps[aid]
        ba.cash = float(st["cash"])
        ba.inventory = {k: int(v) for k, v in st["inventory"].items()}
        ba.inventory_total_cost = {
            k: float(v) for k, v in st["inventory_total_cost"].items()
        }
        # cost_basis is a live view over (inventory, inventory_total_cost);
        # rebind it so it reflects the restored dicts.
        from coffeebench.business_app import _CostBasisView

        ba.cost_basis = _CostBasisView(ba.inventory, ba.inventory_total_cost)
        ba.retail_prices = {k: float(v) for k, v in st["retail_prices"].items()}
        ba.accounts_receivable = [
            invoice_pool[i] for i in st["accounts_receivable"] if i in invoice_pool
        ]
        ba.accounts_payable = [
            invoice_pool[i] for i in st["accounts_payable"] if i in invoice_pool
        ]
        ba.initial_equity = float(st["initial_equity"])

        # --- agent transcript + model ---
        agent = env.agents[aid]
        agent.messages = list(st.get("messages", []))
        model = getattr(agent, "model", None)
        cur_model_name = getattr(model, "model", None)
        saved_model_name = st.get("model")
        # Restore scripted-model internal routine ONLY when the (possibly
        # overridden) model is the same scripted policy that was saved. When a
        # branch overrides the model, the scripted routine is irrelevant and the
        # new model derives fresh state from the restored transcript.
        if (
            model is not None
            and cur_model_name == saved_model_name
            and is_scripted_model(cur_model_name)
            and st.get("scripted_model_state") is not None
            and hasattr(model, "restore_checkpoint_state")
        ):
            model.restore_checkpoint_state(st["scripted_model_state"])
        # Restore cumulative usage counters onto the fresh (per-branch) client
        # so full-run totals match an uninterrupted run. Skip when the model was
        # overridden — a swapped-in client starts its own usage accounting.
        if model is not None and aid not in model_overrides:
            usage = st.get("usage", {}) or {}
            model.n_calls = int(usage.get("n_calls", 0) or 0)
            model.cost = float(usage.get("cost", 0.0) or 0.0)
            model.total_input_tokens = int(usage.get("total_input_tokens", 0) or 0)
            model.total_output_tokens = int(usage.get("total_output_tokens", 0) or 0)
            model.last_input_tokens = int(usage.get("last_input_tokens", 0) or 0)


def _restore_pending(env: Environment, data: dict) -> None:
    by_id = {d.id: d for d in env.marketplace.deals}
    env._pending_shipments = [
        by_id[did] for did in data.get("pending_shipments", []) if did in by_id
    ]
    env._pending_production = [dict(p) for p in data.get("pending_production", [])]
    env._pending_roasting = [dict(r) for r in data.get("pending_roasting", [])]
    env._production_used_today = defaultdict(
        int, {k: int(v) for k, v in data.get("production_used_today", {}).items()}
    )
    env._roast_used_today = defaultdict(
        int, {k: int(v) for k, v in data.get("roast_used_today", {}).items()}
    )


def _restore_accounting(env: Environment, data: dict) -> None:
    env.truth_ledger = defaultdict(
        list,
        {
            aid: [JournalEntry(**e) for e in entries]
            for aid, entries in data.get("truth_ledger", {}).items()
        },
    )
    env.daily_stats = list(data.get("daily_stats", []))
    env.consumer_sales_log = list(data.get("consumer_sales_log", []))
    env._messages_today = defaultdict(
        int, {k: int(v) for k, v in data.get("messages_today", {}).items()}
    )
    env._messages_per_day = defaultdict(
        dict,
        {
            aid: {int(day): int(n) for day, n in per.items()}
            for aid, per in data.get("messages_per_day", {}).items()
        },
    )
    env._bankrupt_agents = set(data.get("bankrupt_agents", []))
    env._bankrupt_day = {k: int(v) for k, v in data.get("bankrupt_day", {}).items()}
    env._bankrupt_reason = dict(data.get("bankrupt_reason", {}))
    env._bankrupt_equity = {
        k: float(v) for k, v in data.get("bankrupt_equity", {}).items()
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_after_load(env: Environment) -> None:
    """Assert that all non-serializable callbacks and cross-references were
    reconnected, and that shared objects (invoices in AR/AP) really are shared.
    Raises `CheckpointValidationError` on any failure."""
    problems: list[str] = []

    if env.marketplace._env is not env:
        problems.append("marketplace._env not reconnected")
    if env.marketplace.on_deal_accepted is None:
        problems.append("marketplace.on_deal_accepted not wired")
    if env.marketplace.business_apps is not env.business_apps:
        problems.append("marketplace.business_apps not pointing at env's apps")
    if env.id_gen is not env.marketplace.id_gen:
        problems.append("env.id_gen and marketplace.id_gen diverged")

    for aid, ba in env.business_apps.items():
        if getattr(ba, "_env", None) is not env:
            problems.append(f"{aid}: BusinessApp._env not reconnected")
        if not callable(getattr(ba, "_record_truth", None)):
            problems.append(f"{aid}: BusinessApp._record_truth not wired")
        agent = env.agents.get(aid)
        if agent is None:
            problems.append(f"{aid}: no Agent")
            continue
        if not agent.tools:
            problems.append(f"{aid}: agent has no tools bound")
        # Tools must be bound to THIS agent's business app (not a stale one).
        tool_names = {t.__name__ for t in agent.tools}
        if "wait_for_next_day" not in tool_names:
            problems.append(f"{aid}: expected wait_for_next_day tool missing")

    # Invoice sharing: every invoice in an issuer's AR must be the SAME object
    # in the payer's AP (spot-check the whole pool).
    ap_by_id: dict[str, int] = {}
    for ba in env.business_apps.values():
        for inv in ba.accounts_payable:
            ap_by_id[inv.id] = id(inv)
    for ba in env.business_apps.values():
        for inv in ba.accounts_receivable:
            if inv.id in ap_by_id and ap_by_id[inv.id] != id(inv):
                problems.append(
                    f"invoice {inv.id}: AR and AP hold DIFFERENT objects "
                    "(sharing broken)"
                )

    if problems:
        raise CheckpointValidationError(
            "Checkpoint post-load validation failed:\n  - " + "\n  - ".join(problems)
        )
