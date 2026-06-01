"""Batch runner for CoffeeBench experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from coffeebench.config import RunConfig


def _parse_argv() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--config", type=str, required=True, help="Path to experiment TOML config."
    )
    p.add_argument(
        "--max_days",
        type=int,
        default=None,
        help="Override config.run.max_days for all seeds.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds override (e.g. '0,1,2'). "
        "If omitted, uses config.run.seeds.",
    )
    p.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip seeds whose trajectories/<name>/seed_<N>/run.json already exists.",
    )
    return p.parse_args()


def _seed_completed(config: RunConfig, seed: int) -> bool:
    return config.output_dir_for_seed(seed).joinpath("run.json").exists()


def _summarise_run(traj_path: Path) -> dict:
    """Load a finished run.json and return the per-agent score + audit
    summary used by the aggregate table."""
    if not traj_path.exists():
        return {"loaded": False}
    try:
        data = json.loads(traj_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"loaded": False, "error": str(exc)}
    res = data.get("result") or {}
    agents = res.get("agents") or {}
    bankrupt = data.get("bankrupt_agents") or []
    bankrupt_day = data.get("bankrupt_day") or {}
    per_agent = {}
    for aid, ag in agents.items():
        a = ag.get("audit") or {}
        bs = a.get("balance_sheet") or {}
        ann = a.get("annual") or {}
        per_agent[aid] = {
            "true_net_income": ann.get("true_net_income"),
            "true_revenue": a.get("true_revenue_net"),
            "return_rate": a.get("return_rate"),
            "true_equity": bs.get("true_equity"),
            "roast_metrics": a.get("roast_metrics"),
            "bankrupt_day": bankrupt_day.get(aid),
        }
    return {
        "loaded": True,
        "max_days": data.get("max_days"),
        "final_day": data.get("final_day"),
        "per_agent": per_agent,
        "bankrupt": list(bankrupt),
    }


def _run_one(config_path: str, seed: int, args: argparse.Namespace) -> int:
    """Spawn `coffeebench.main --config <path> --seed N` as a subprocess and
    return its exit code. stdout/stderr stream to the parent terminal so
    the user sees live progress."""
    cmd = [
        sys.executable,
        "-m",
        "coffeebench.main",
        "--config",
        config_path,
        "--seed",
        str(seed),
    ]
    if args.max_days is not None:
        cmd += ["--max_days", str(args.max_days)]
    print(f"\n[experiment] >>> seed {seed}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print(f"[experiment] interrupted during seed {seed}; aborting batch.")
        raise
    dt = time.time() - t0
    rc = proc.returncode
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    print(f"[experiment] <<< seed {seed}: {status} in {dt / 60:.1f} min", flush=True)
    return rc


def _print_summary_table(config: RunConfig, results: dict[int, dict]) -> None:
    print()
    print("=" * 78)
    print(f"Batch summary: {config.name}  ({len(results)} seeds run)")
    print("=" * 78)
    seeds = sorted(results.keys())
    if not seeds:
        return
    # Per-agent audit-NI table (the canonical leaderboard score).
    agent_ids = sorted(
        {
            aid
            for r in results.values()
            if r.get("loaded")
            for aid in (r.get("per_agent") or {})
        }
    )
    header = (
        f"  {'seed':>4}  "
        + "  ".join(f"{aid:>13}" for aid in agent_ids)
        + f"  {'cross':>14}"
    )
    print(header)
    for seed in seeds:
        r = results[seed]
        if not r.get("loaded"):
            print(f"  {seed:>4}  (run not loaded: {r.get('error', 'no run.json')})")
            continue
        ni = " ".join(
            f"{(r['per_agent'].get(aid) or {}).get('true_net_income', '—'):>13}"
            for aid in agent_ids
        )
        c = r.get("cross") or {}
        cross_str = (
            f"c{c.get('circular', 0)}f{c.get('friendly', 0)}o{c.get('overdue', 0)}"
        )
        bk = ",".join(r.get("bankrupt") or []) or "—"
        print(f"  {seed:>4}  {ni}  {cross_str:>14}   bk={bk}")


def _write_aggregate(config: RunConfig, results: dict[int, dict]) -> Path:
    out_dir = Path("trajectories") / config.name / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"experiment": config.name, "seeds": dict(sorted(results.items()))},
            indent=2,
            default=str,
        )
    )
    return summary_path


def main() -> None:
    args = _parse_argv()
    config = RunConfig.from_toml(args.config)
    seeds = (
        [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if args.seeds
        else list(config.seeds)
    )
    print(f"[experiment] {config.name}: {len(seeds)} seed(s) → {seeds}")
    if config.description:
        print(f"[experiment]   {config.description}")

    results: dict[int, dict] = {}
    for seed in seeds:
        if args.skip_completed and _seed_completed(config, seed):
            print(f"[experiment] seed {seed}: trajectory exists, skipping")
        else:
            _run_one(args.config, seed, args)
        traj_path = config.output_dir_for_seed(seed) / "run.json"
        results[seed] = _summarise_run(traj_path)

    _print_summary_table(config, results)
    summary_path = _write_aggregate(config, results)
    print(f"\n[experiment] aggregate summary saved to {summary_path}")


if __name__ == "__main__":
    main()
