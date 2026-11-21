"""Concurrency-capped orchestrator for the rebuttal ablation batch.

Runs `coffeebench.main` for each (config, seed), at most --workers at a time.
Skips (config, seed) whose run.json already exists, so it is safe to re-run.
Launch detached, e.g.:

    nohup uv run python scripts/run_ablation.py \
        --tags gpt55,opus47,haiku45,kimi,gemini \
        --roles roaster,farmer --seeds 0,1,2 --workers 8 \
        > run_logs/ablation_orchestrator.log 2>&1 &
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
    ap.add_argument("--tags", required=True, help="comma-separated model tags")
    ap.add_argument("--roles", default="roaster,farmer")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tags = args.tags.split(",")
    roles = args.roles.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = []
    for role in roles:
        for tag in tags:
            cfg = Path(f"experiments/abl_qwenbg_{role}_{tag}.toml")
            if not cfg.exists():
                print(f"[skip] missing config {cfg}", flush=True)
                continue
            exp = tomllib.loads(cfg.read_text())["experiment"]["name"]
            for s in seeds:
                done = Path(f"trajectories/{exp}/seed_{s}/run.json")
                if done.exists():
                    print(f"[skip] already done {exp} seed {s}", flush=True)
                    continue
                jobs.append((cfg, exp, s))

    Path("run_logs").mkdir(exist_ok=True)
    print(f"[orch] {len(jobs)} jobs, {args.workers} workers", flush=True)

    def run(job):
        cfg, exp, s = job
        log = Path(f"run_logs/{exp}_seed{s}.log")
        with open(log, "w") as fh:
            rc = subprocess.run(
                [sys.executable, "-m", "coffeebench.main",
                 "--config", str(cfg), "--seed", str(s)],
                stdout=fh, stderr=subprocess.STDOUT,
            ).returncode
        return exp, s, rc

    t0 = time.monotonic()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for fut in cf.as_completed(futs):
            exp, s, rc = fut.result()
            done += 1
            mins = (time.monotonic() - t0) / 60
            print(f"[orch] done {done}/{len(jobs)} :: {exp} seed {s} rc={rc} "
                  f"({mins:.0f} min elapsed)", flush=True)
    print("[orch] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
