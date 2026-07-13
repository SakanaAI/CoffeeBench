"""Phase B orchestrator: observe → checkpoint → label+mine → fork → compare.

The full controlled CoffeeBench × Valto experiment in staged steps so the mined
policy can be inspected before the (paid) governed/baseline forks run:

    STAGE 1  observe   run focal agent under Valto observe to `--checkpoint-day`,
                       save a checkpoint; label + mine the days-[0, D-1] ledger.
                       Stops here if `--stop-after-mine` (inspect learned.yaml).
    STAGE 2  fork      resume the SAME checkpoint twice from day D → max_days:
                         • governed  — enforce the mined policy on the focal agent
                         • baseline  — ungoverned (Valto off)
                       label each fork, then print a paired comparison.

Because both forks start from byte-identical state (same seed, same RNG, same
books) and only the focal agent's governance differs, the second-half NI/survival
difference is a clean estimate of the intervention.

    # stage 1 only (then pause to inspect rules)
    uv run python -m coffeebench.valto.run_experiment \
        --config experiments/roaster_valto_observe_haiku.toml --seed 0 \
        --checkpoint-day 45 --stop-after-mine

    # stage 2 (after inspecting/approving the policy)
    uv run python -m coffeebench.valto.run_experiment \
        --config experiments/roaster_valto_observe_haiku.toml --seed 0 \
        --checkpoint-day 45 --forks-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from coffeebench.checkpoint import load_checkpoint
from coffeebench.config import RunConfig
from coffeebench.main import build_run, print_usage_summary
from coffeebench.valto.analyze_coffee import load_rows, mine, split_train_eval
from coffeebench.valto.labeling import label_run


def _out_dir(config: RunConfig, seed: int) -> Path:
    d = Path("trajectories") / config.name / f"seed_{seed}" / "valto_experiment"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _focal(config: RunConfig) -> str:
    actor = (config.valto or {}).get("actor")
    if not actor:
        raise SystemExit("config [valto].actor must name the focal agent")
    return actor


# ---------------------------------------------------------------------------
# STAGE 1 — observe to checkpoint, then label + mine
# ---------------------------------------------------------------------------
def stage_observe_and_mine(config_path: str, seed: int, checkpoint_day: int) -> dict:
    config = RunConfig.from_toml(config_path)
    config.apply_economy_overrides()
    out = _out_dir(config, seed)
    ckpt = out / f"day{checkpoint_day}.checkpoint.json"
    policy_out = out / "learned.yaml"
    report_out = out / "LEARNED_POLICY_REPORT.md"

    print(f"\n=== STAGE 1: observe → day {checkpoint_day} checkpoint ===")
    args = SimpleNamespace(
        config=config_path, model=None, models=None, max_days=None,
        seed=seed, main_agent=None, output=None,
    )
    env, _o, _e = build_run(args)
    env._checkpoint_at_day = checkpoint_day
    env._checkpoint_out = str(ckpt)
    asyncio.run(env.run())
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not written: {ckpt}")

    # Label the observe ledger (days 0..D-1) and mine a policy from it.
    print(f"\n=== STAGE 1: label + mine (days 0..{checkpoint_day - 1}) ===")
    label_run(env)  # writes utility/metrics onto the observe ledger
    ledger_db = str(config.output_dir_for_seed(seed) / "valto_ledger.db")
    # Export the labeled observe ledger as flat raw data (CSV + JSONL) for
    # offline analysis before we mine / fork.
    from coffeebench.valto.export import export_ledger

    export_ledger(ledger_db, str(config.output_dir_for_seed(seed) / "valto_decisions"))
    rows = load_rows([ledger_db])
    train, ev, held_out = split_train_eval(rows, None)
    rules, report = mine(train, ev)

    from valto.engine import validate_policy
    import yaml

    policy = {
        "version": f"coffee-learned-{config.name}-s{seed}-d{checkpoint_day}",
        "budgets": {},
        "categories": {"buy_green": "procurement", "roast": "production"},
        "rules": rules,
        "default": "approve",
    }
    validate_policy(policy)
    policy_out.write_text(yaml.safe_dump(policy, sort_keys=False, width=100))
    _write_mine_report(report_out, rows, train, ev, held_out, report)

    print_usage_summary(env)
    print(f"\n[experiment] checkpoint: {ckpt}")
    print(f"[experiment] mined policy ({len(rules)} rule(s)): {policy_out}")
    _print_rules(rules, report)
    return {
        "config_path": config_path, "seed": seed, "checkpoint_day": checkpoint_day,
        "checkpoint": str(ckpt), "policy": str(policy_out),
        "n_rules": len(rules), "out_dir": str(out),
    }


# ---------------------------------------------------------------------------
# STAGE 2 — fork governed vs baseline from the checkpoint
# ---------------------------------------------------------------------------
def _run_fork(config, seed, ckpt, branch, valto_spec, out) -> dict:
    print(f"\n--- fork: {branch} ---")
    events = str(out / f"{branch}.events.jsonl")
    env = load_checkpoint(
        str(ckpt), branch_name=branch, events_path=events, valto=valto_spec,
        verbose=True,
    )
    asyncio.run(env.run())
    traj = out / f"{branch}.json"
    env.save_trajectory(str(traj))
    if valto_spec.get("enabled"):
        label_run(env)  # label the governed branch's own ledger
    print_usage_summary(env)
    return _summarize(env, config, branch)


def stage_forks(meta: dict) -> dict:
    config = RunConfig.from_toml(meta["config_path"])
    config.apply_economy_overrides()
    seed = meta["seed"]
    ckpt = Path(meta["checkpoint"])
    policy = meta["policy"]
    actor = _focal(config)
    govern_tools = (config.valto or {}).get("govern_tools", ["make_offer", "roast"])
    out = _out_dir(config, seed)

    print(f"\n=== STAGE 2: fork governed vs baseline from {ckpt.name} ===")
    baseline = _run_fork(
        config, seed, ckpt, "baseline", {"enabled": False}, out
    )
    governed = _run_fork(
        config, seed, ckpt, "governed",
        {"enabled": True, "actor": actor, "mode": "governed",
         "policy": policy, "govern_tools": govern_tools}, out,
    )
    _print_comparison(actor, baseline, governed)
    result = {"actor": actor, "baseline": baseline, "governed": governed}
    (out / "comparison.json").write_text(json.dumps(result, indent=2, default=str))
    return result


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def _summarize(env, config, branch: str) -> dict:
    actor = _focal(config)
    res = env._final_result or env._finish()
    a = res["agents"][actor]
    return {
        "branch": branch,
        "final_day": int(env.time_manager.get_current_day()),
        "net_income": a["audit"]["annual"]["true_net_income"],
        "true_equity": a["true_final_equity"],
        "completed": a["completed"],
        "bankrupt_day": a["bankrupt_day"],
        "bankrupt_reason": a["bankrupt_reason"],
        "model_cost": round(a["usage"]["cost"], 4),
    }


def _print_rules(rules: list[dict], report: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("MINED POLICY — rules to be enforced on the governed fork")
    print("=" * 70)
    if not rules:
        print("  (no rules cleared the minimum-evidence bar — governed fork would\n"
              "   behave identically to baseline. Consider a longer observe window\n"
              "   or more seeds.)")
    for r in rules:
        print(f"\n  • {r['id']}  →  {r['verdict']}")
        print(f"      match: {json.dumps(r['match'])}")
        print(f"      reason: {r['reason']}")
    print("\n  candidate backtest (all proposals):")
    for c in report:
        s = c["stats"]
        print(f"    {'EMIT' if c['emitted'] else 'skip'} {c['id']:<26} "
              f"n={s['n']:<4} prec={s['precision']:.0%} util=${s['total_utility']:.2f}")
    print("=" * 70)


def _print_comparison(actor: str, baseline: dict, governed: dict) -> None:
    lift = governed["net_income"] - baseline["net_income"]
    print("\n" + "=" * 70)
    print(f"PAIRED COMPARISON — {actor} (same seed, forked at checkpoint)")
    print("=" * 70)
    print(f"  {'arm':<10} {'net_income':>12} {'equity':>12} {'completed':>10} {'cost':>8}")
    for s in (baseline, governed):
        print(f"  {s['branch']:<10} {s['net_income']:>12.2f} {s['true_equity']:>12.2f} "
              f"{str(s['completed']):>10} {s['model_cost']:>8}")
    print("-" * 70)
    print(f"  NI lift (governed − baseline): ${lift:,.2f}")
    if baseline["bankrupt_day"] is not None or governed["bankrupt_day"] is not None:
        print(f"  survival: baseline bust day {baseline['bankrupt_day']} | "
              f"governed bust day {governed['bankrupt_day']}")
    print("=" * 70)


def _write_mine_report(path, rows, train, ev, held_out, report) -> None:
    lines = [
        "# Learned CoffeeBench policy report", "",
        f"Labeled decisions: {len(rows)} (train {len(train)} / eval {len(ev)}, "
        f"held-out={held_out}).", "",
        "| candidate | n (eval) | waste precision | total utility | emitted |",
        "|---|---:|---:|---:|---|",
    ]
    for c in report:
        s = c["stats"]
        lines.append(
            f"| {c['id']} | {s['n']} | {s['precision']:.0%} | "
            f"${s['total_utility']:.2f} | {'yes' if c['emitted'] else 'no'} |")
    if not report:
        lines.append("| (no candidates cleared the bar) | | | | |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-day", type=int, default=45)
    p.add_argument("--stop-after-mine", action="store_true",
                   help="Run stage 1 only, then stop so the mined policy can be "
                        "inspected before the paid forks.")
    p.add_argument("--forks-only", action="store_true",
                   help="Skip stage 1; reuse the existing checkpoint + policy and "
                        "run stage 2 (governed vs baseline) only.")
    args = p.parse_args()

    config = RunConfig.from_toml(args.config)
    out = _out_dir(config, args.seed)
    meta_path = out / "experiment_meta.json"

    if args.forks_only:
        if not meta_path.exists():
            raise SystemExit(f"no stage-1 meta at {meta_path}; run without --forks-only first")
        meta = json.loads(meta_path.read_text())
    else:
        meta = stage_observe_and_mine(args.config, args.seed, args.checkpoint_day)
        meta_path.write_text(json.dumps(meta, indent=2))
        if args.stop_after_mine:
            print(f"\n[experiment] stage 1 complete. Inspect {meta['policy']}, then "
                  f"re-run with --forks-only to fork governed vs baseline.")
            return

    stage_forks(meta)


if __name__ == "__main__":
    main()
