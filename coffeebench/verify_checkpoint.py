"""Automated proof that the checkpoint is faithful and branch-safe.

Runs the deterministic scripted economy four ways and compares canonical
ECONOMIC summaries:

    Run A  uninterrupted            days 0..max_days-1
    Run B  checkpoint + resume      days 0..D-1, save, reload, days D..max_days-1
    Run C  branch 1 from checkpoint days D..max_days-1
    Run D  branch 2 from checkpoint days D..max_days-1

Asserts A == B (resume is faithful) and C == D (branches are independent but
equivalent under scripted policies). Because IDs are deterministic and the RNG /
scheduler / books are fully restored, the summaries should match exactly.

Usage:

    uv run python -m coffeebench.verify_checkpoint \
        --config experiments/deterministic_economy.toml --seed 0 --checkpoint-day 45
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

from coffeebench.checkpoint import load_checkpoint
from coffeebench.main import _usage_summary, build_run

# Non-economic metadata deliberately excluded from the canonical comparison
# (documented in docs/deterministic_checkpointing.md):
EXCLUDED_FIELDS = (
    "message timestamps (datetime.now on each agent turn)",
    "checkpoint 'saved_at' wall-clock stamp",
    "event-stream ts_ms (wall-clock offsets)",
)


def _fresh_env(config: str, seed: int, max_days: int | None):
    args = SimpleNamespace(
        config=config,
        model=None,
        models=None,
        max_days=max_days,
        seed=seed,
        main_agent=None,
        output=None,
    )
    env, _out, _ev = build_run(args)
    env.verbose = False
    env.event_logger = None  # keep verification quiet + collision-free
    return env


def _round(v, n=4):
    return round(float(v), n) if isinstance(v, (int, float)) else v


def canonical_summary(env) -> dict:
    """Deterministic, economically-meaningful snapshot of a FINISHED env."""
    result = env._finish() if getattr(env, "_final_result", None) is None else env._final_result

    agents = {}
    for aid, ba in sorted(env.business_apps.items()):
        # Per-entry-type truth-ledger rollup (count + summed amount).
        ledger: dict[str, list] = {}
        for e in env.truth_ledger.get(aid, []):
            row = ledger.setdefault(e.entry_type, [0, 0.0])
            row[0] += 1
            row[1] = round(row[1] + e.amount, 4)
        ares = result["agents"][aid]
        agents[aid] = {
            "cash": _round(ba.cash, 2),
            "inventory": dict(sorted(ba.inventory.items())),
            "inventory_total_cost": {
                k: _round(v, 4) for k, v in sorted(ba.inventory_total_cost.items())
            },
            "retail_prices": {k: _round(v, 2) for k, v in sorted(ba.retail_prices.items())},
            "ar_outstanding": _round(ba._ar_outstanding(), 2),
            "ap_outstanding": _round(ba._ap_outstanding(), 2),
            "true_equity": _round(ba._compute_true_equity(), 2),
            "net_income": ares["net_income"],
            "audit_net_income": ares["audit"]["annual"]["true_net_income"],
            "completed": ares["completed"],
            "bankrupt_day": env._bankrupt_day.get(aid),
            "bankrupt_reason": env._bankrupt_reason.get(aid),
            "truth_ledger_rollup": {k: ledger[k] for k in sorted(ledger)},
            "n_calls": ares["usage"]["n_calls"],
        }

    def _dump(objs):
        from dataclasses import asdict

        return [asdict(o) for o in objs]

    return {
        "final_day": int(env.time_manager.get_current_day()),
        "step_count": int(env.step_count),
        "agents": agents,
        "marketplace": {
            "listings": _dump(env.marketplace.listings),
            "offers": _dump(env.marketplace.offers),
            "deals": _dump(env.marketplace.deals),
            "messages": _dump(env.marketplace.messages),
        },
        "consumer_sales": {
            "count": len(env.consumer_sales_log),
            "total_qty": sum(int(s.get("qty", 0)) for s in env.consumer_sales_log),
            "total_revenue": round(
                sum(float(s.get("total_price", 0.0)) for s in env.consumer_sales_log), 2
            ),
        },
        "pending": {
            "shipments": sorted(d.id for d in env._pending_shipments),
            "production": len(env._pending_production),
            "roasting": len(env._pending_roasting),
        },
        "daily_stats_len": len(env.daily_stats),
        "id_gen": env.id_gen.state(),
        "bankrupt_agents": sorted(env._bankrupt_agents),
    }


def _diff(a: dict, b: dict, path: str = "") -> list[str]:
    """Recursive structural diff; returns human-readable mismatch paths."""
    out: list[str] = []
    if type(a) is not type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
    ):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: missing in A")
            elif k not in b:
                out.append(f"{path}.{k}: missing in B")
            else:
                out += _diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: list len {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _diff(x, y, f"{path}[{i}]")
    else:
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
    return out


def compare(name_a: str, a: dict, name_b: str, b: dict) -> bool:
    diffs = _diff(a, b, "")
    if not diffs:
        print(f"  [PASS] {name_a} == {name_b} (economically identical)")
        return True
    print(f"  [FAIL] {name_a} != {name_b}: {len(diffs)} mismatch(es):")
    for d in diffs[:25]:
        print(f"      {d}")
    if len(diffs) > 25:
        print(f"      … and {len(diffs) - 25} more")
    return False


def run() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiments/deterministic_economy.toml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-day", type=int, default=45)
    p.add_argument(
        "--max_days",
        type=int,
        default=None,
        help="Override horizon (faster for CI/tests).",
    )
    args = p.parse_args()

    tmpdir = Path(tempfile.mkdtemp(prefix="coffee_ckpt_"))
    ckpt = str(tmpdir / f"day{args.checkpoint_day}.checkpoint.json")

    print(
        f"[verify] config={args.config} seed={args.seed} "
        f"checkpoint_day={args.checkpoint_day} max_days={args.max_days}"
    )

    # --- Run A: uninterrupted ---
    print("[verify] Run A: uninterrupted …")
    env_a = _fresh_env(args.config, args.seed, args.max_days)
    asyncio.run(env_a.run())
    sum_a = canonical_summary(env_a)

    # --- Run B: checkpoint + resume ---
    print(f"[verify] Run B: run to day {args.checkpoint_day}, save, reload, finish …")
    env_b0 = _fresh_env(args.config, args.seed, args.max_days)
    env_b0._checkpoint_at_day = args.checkpoint_day
    env_b0._checkpoint_out = ckpt
    asyncio.run(env_b0.run())
    if not Path(ckpt).exists():
        print(f"[verify] FAIL: checkpoint not written to {ckpt}")
        return 1
    env_b = load_checkpoint(ckpt, verbose=False)
    asyncio.run(env_b.run())
    sum_b = canonical_summary(env_b)

    # --- Runs C, D: two independent branches from the same checkpoint ---
    print("[verify] Run C, D: two independent branches from the checkpoint …")
    env_c = load_checkpoint(ckpt, branch_name="branch_c", verbose=False)
    asyncio.run(env_c.run())
    sum_c = canonical_summary(env_c)
    env_d = load_checkpoint(ckpt, branch_name="branch_d", verbose=False)
    asyncio.run(env_d.run())
    sum_d = canonical_summary(env_d)

    # Independence: distinct env/agent/model/marketplace/rng objects.
    print("[verify] Object-independence checks …")
    indep_ok = _independence(env_c, env_d)

    print("[verify] Equality checks …")
    ok = True
    ok &= compare("RunA(uninterrupted)", sum_a, "RunB(resumed)", sum_b)
    ok &= compare("RunC(branch)", sum_c, "RunD(branch)", sum_d)
    ok &= compare("RunA(uninterrupted)", sum_a, "RunC(branch)", sum_c)

    print("[verify] External-cost check …")
    usage = _usage_summary(env_a)
    zero_cost = (
        usage["external_api_calls"] == 0
        and usage["external_model_cost"] == 0.0
        and usage["external_input_tokens"] == 0
        and usage["external_output_tokens"] == 0
    )
    print(
        f"  external_api_calls={usage['external_api_calls']} "
        f"cost=${usage['external_model_cost']} "
        f"local_invocations={usage['local_scripted_invocations']} "
        f"→ {'[PASS]' if zero_cost else '[FAIL]'}"
    )

    print("\n[verify] Activity (Run A):")
    _print_activity(sum_a)

    all_ok = ok and indep_ok and zero_cost
    print(f"\n[verify] {'ALL CHECKS PASSED ✅' if all_ok else 'CHECKS FAILED ❌'}")
    print("[verify] Excluded (non-economic) metadata:")
    for f in EXCLUDED_FIELDS:
        print(f"  - {f}")
    return 0 if all_ok else 1


def _independence(env_c, env_d) -> bool:
    ok = True
    checks = [
        ("env", env_c is not env_d),
        ("marketplace", env_c.marketplace is not env_d.marketplace),
        ("event_loop", env_c.event_loop is not env_d.event_loop),
        ("rng", env_c.rng is not env_d.rng),
        ("id_gen", env_c.id_gen is not env_d.id_gen),
    ]
    for aid in env_c.agents:
        checks.append(
            (f"agent[{aid}]", env_c.agents[aid] is not env_d.agents[aid])
        )
        checks.append(
            (
                f"model[{aid}]",
                env_c.agents[aid].model is not env_d.agents[aid].model,
            )
        )
        checks.append(
            (
                f"messages[{aid}]",
                env_c.agents[aid].messages is not env_d.agents[aid].messages,
            )
        )
    for name, passed in checks:
        if not passed:
            print(f"  [FAIL] shared object between branches: {name}")
            ok = False
    if ok:
        print("  [PASS] branches share no mutable objects")
    return ok


def _print_activity(summary: dict) -> None:
    mkt = summary["marketplace"]
    print(f"  listings={len(mkt['listings'])} offers={len(mkt['offers'])} "
          f"deals={len(mkt['deals'])}")
    cs = summary["consumer_sales"]
    print(f"  consumer_sales: {cs['count']} sales, {cs['total_qty']} kg, "
          f"${cs['total_revenue']}")
    print("  net income by agent:")
    for aid, a in summary["agents"].items():
        bk = f" BANKRUPT day {a['bankrupt_day']}" if a["bankrupt_day"] is not None else ""
        print(f"    {aid:12s} NI=${a['audit_net_income']:>10.2f} "
              f"equity=${a['true_equity']:>10.2f}{bk}")


if __name__ == "__main__":
    raise SystemExit(run())
