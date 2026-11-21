"""Generic concurrency-capped runner for an explicit list of experiment configs.

    nohup uv run python scripts/run_configs.py \
        --configs demand_b40_gpt55,demand_b80_gpt55,... \
        --seeds 0,1 --workers 6 > run_logs/foo.log 2>&1 &

Skips (config, seed) whose run.json already exists (safe to re-run).
"""

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
import tomllib
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma-separated experiment names")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    names = args.configs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = []
    for name in names:
        cfg = Path(f"experiments/{name}.toml")
        if not cfg.exists():
            print(f"[skip] missing config {cfg}", flush=True)
            continue
        exp = tomllib.loads(cfg.read_text())["experiment"]["name"]
        for s in seeds:
            if Path(f"trajectories/{exp}/seed_{s}/run.json").exists():
                print(f"[skip] done {exp} seed {s}", flush=True)
                continue
            jobs.append((cfg, exp, s))

    Path("run_logs").mkdir(exist_ok=True)
    print(f"[run] {len(jobs)} jobs, {args.workers} workers", flush=True)

    def run(job):
        cfg, exp, s = job
        with open(f"run_logs/{exp}_seed{s}.log", "w") as fh:
            rc = subprocess.run(
                [sys.executable, "-m", "coffeebench.main",
                 "--config", str(cfg), "--seed", str(s)],
                stdout=fh, stderr=subprocess.STDOUT,
            ).returncode
        return exp, s, rc

    t0 = time.monotonic()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in cf.as_completed([ex.submit(run, j) for j in jobs]):
            exp, s, rc = fut.result()
            done += 1
            print(f"[run] done {done}/{len(jobs)} :: {exp} seed {s} rc={rc} "
                  f"({(time.monotonic()-t0)/60:.0f} min)", flush=True)
    print("[run] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
