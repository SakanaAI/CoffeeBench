"""Head-to-head ("Arena") schedule: balanced pairings → qsub lines + cost estimate.

    uv run python harbor_adapter/h2h_schedule.py                       # pilot: 7 models, k=4, seed 0
    uv run python harbor_adapter/h2h_schedule.py --round-robin --reps 2 --mirror Sonnet55,Qwen38F
    uv run python harbor_adapter/h2h_schedule.py --configs experiments/camera_ready_hb/hb_roaster_d30.toml,...

Pairings (unordered): `--k K` (default 4) = a K-regular circulant graph on a
seeded random permutation of the models (every model plays exactly K distinct
opponents; n·K must be even), or `--round-robin` (all n(n-1)/2 pairs). Every
pairing runs in BOTH seat orders (X_vs_Y: X in roaster_A; Y_vs_X: swapped).
`--mirror A,B` adds self-play runs (A vs A) to estimate the seat advantage.
Replicates: `--reps R` = R seeds per ordered pair (seed-base .. seed-base+R-1;
or `--seeds 0,3,7`). Several `--configs` = separate arenas (e.g. roaster and
farmer H2H; launch_sim maps --focal2 to the other seat of the config's
main_agent role). Runs are grouped `--per-job` (default 3) per cluster job via
CB_PAIRS (one seed and one arena per job). Output order is deterministic
given `--rng-seed`. Prints only — paid runs need explicit user approval.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_MODELS = ["Fable5", "Sonnet55", "MuseW", "Muse13", "Gemini38F", "Qwen38F", "MiMo26P"]
# Rough focal cost per 30-day run (USD, max effort; from the seed-0 / pilot runs).
FOCAL_COST = {"Fable5": 30.0, "MuseW": 25.0, "Sonnet55": 12.0, "Muse13": 10.0,
              "Gemini38F": 4.0, "Qwen38F": 0.5, "MiMo26P": 0.4,
              "Astra": 20.0, "GLM53": 5.0, "K3": 3.0, "DSv41F": 0.5}
BG_COST = {"Sonnet55": 8.0}  # 4 background seats per 30-day run
# 90-day runs ≈ 3× the 30-day cost (focal and background; the H2H bg is Sonnet55, not the slower Qwen roster).
HORIZON_MULT = {30: 1.0, 90: 3.0}
DEFAULT_CONFIG = {30: "experiments/camera_ready_hb/hb_roaster_d30.toml",   # pilot
                  90: "experiments/camera_ready_hb/hb_roaster.toml"}       # paper runs (festival days 40-53)
# 12 h jobs for both horizons: a 90-day run continues across jobs via the
# walltime stop + exact resume + self-requeue (as the seed-0 90-day matrix did).
# --walltime-h N > 12 passes -l walltime and CB_WALLTIME_SEC (kept on requeue).
DEFAULT_WALLTIME_H = {30: 12, 90: 12}


def k_regular_pairs(models: list[str], k: int, rng: random.Random) -> list[tuple[str, str]]:
    """Circulant K-regular graph on a random permutation: offsets 1..K/2 (+ n/2 if K odd)."""
    n = len(models)
    if not 0 < k < n:
        raise SystemExit(f"--k must be in 1..{n - 1}")
    if (n * k) % 2:
        raise SystemExit(f"n·k must be even (n={n}, k={k})")
    perm = models[:]
    rng.shuffle(perm)
    offsets = list(range(1, k // 2 + 1)) + ([n // 2] if k % 2 else [])
    edges = set()
    for i in range(n):
        for o in offsets:
            a, b = perm[i], perm[(i + o) % n]
            edges.add(tuple(sorted((a, b), key=models.index)))
    return sorted(edges, key=lambda e: (models.index(e[0]), models.index(e[1])))


def round_robin_pairs(models: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]


def ordered_runs(pairs: list[tuple[str, str]], mirror: list[str]) -> list[tuple[str, str]]:
    runs = [r for a, b in pairs for r in ((a, b), (b, a))]
    return runs + [(m, m) for m in mirror]


def run_cost(x: str, y: str, bg: str, focal_cost: dict, bg_cost: float) -> float:
    return focal_cost.get(x, 0.0) + focal_cost.get(y, 0.0) + bg_cost


def total_cost(runs, bg, focal_cost, bg_cost, n_seeds) -> float:
    return n_seeds * sum(run_cost(x, y, bg, focal_cost, bg_cost) for x, y in runs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--k", type=int, default=4, help="Distinct opponents per model (K-regular).")
    g.add_argument("--round-robin", action="store_true", help="All pairs.")
    ap.add_argument("--mirror", default="", help="Comma-separated models that also play themselves (X vs X).")
    ap.add_argument("--reps", type=int, default=1, help="Seeds per ordered pair (seed-base, seed-base+1, ...).")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="Explicit seed list (overrides --reps/--seed-base).")
    ap.add_argument("--rng-seed", type=int, default=0, help="Pairing RNG seed (k-regular mode).")
    ap.add_argument("--bg", default="Sonnet55")
    ap.add_argument("--days", type=int, choices=(30, 90), default=30,
                    help="Horizon: 30 = pilot (hb_roaster_d30), 90 = paper (hb_roaster, festival days 40-53).")
    ap.add_argument("--configs", default=None, help="Comma-separated arena configs (default by --days).")
    ap.add_argument("--walltime-h", type=int, default=None, help="Job walltime in hours (default 12; longer runs requeue).")
    ap.add_argument("--per-job", type=int, default=3, help="Simulations per cluster job (CB_PAIRS).")
    ap.add_argument("--effort", default="max")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    mirror = [m for m in args.mirror.split(",") if m]
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(range(args.seed_base, args.seed_base + args.reps)))
    rng = random.Random(args.rng_seed)
    pairs = round_robin_pairs(models) if args.round_robin else k_regular_pairs(models, args.k, rng)
    runs = ordered_runs(pairs, mirror)
    configs = [c for c in (args.configs or DEFAULT_CONFIG[args.days]).split(",") if c]
    wall_h = args.walltime_h or DEFAULT_WALLTIME_H[args.days]
    # cluster/pbs_job.sh's header is 12 h; longer jobs pass -l walltime and the matching CB_WALLTIME_SEC
    # (launch_sim's walltime stop + requeue keep it for the continuation jobs).
    lopt, wvar = (f"-l walltime={wall_h:02d}:00:00 ", f",CB_WALLTIME_SEC={wall_h * 3600}") if wall_h != 12 else ("", "")
    bg_cost = BG_COST.get(args.bg, 8.0)

    mode = "round-robin" if args.round_robin else f"k={args.k}"
    print(f"# H2H schedule ({mode}, rng-seed {args.rng_seed}): {len(models)} models, {len(pairs)} pairings "
          f"× 2 seat orders + {len(mirror)} mirror = {len(runs)} runs per seed × {len(seeds)} seed(s) "
          f"× {len(configs)} arena(s) = {len(runs) * len(seeds) * len(configs)} sims; bg {args.bg}; "
          f"{args.days} days, {wall_h} h jobs")
    for a, b in pairs:
        print(f"#   {a} – {b}")
    opp = {m: sorted({b if a == m else a for a, b in pairs if m in (a, b)}) for m in models}
    for m in models:
        print(f"#   {m}: {len(opp[m])} opponents {opp[m]}")

    print("\n# qsub lines (submit from the deploy dir; one seed and one arena per job)")
    for cfg in configs:
        tag = Path(cfg).stem.removeprefix("hb_")
        for seed in seeds:
            for j in range(0, len(runs), args.per_job):
                chunk = runs[j:j + args.per_job]
                pairs_v = "+".join(f"{x}:{y}" for x, y in chunk)
                name = f"cb_h2h_{args.bg}_{tag}_s{seed}_j{j // args.per_job:02d}"
                print(f"qsub -N {name} {lopt}-v RTYPE=rt_HC,USE_SSH=1,CB_CONFIG={cfg},CB_SEEDS={seed},"
                      f"CB_BG={args.bg},CB_EFFORT={args.effort},CB_PAIRS={pairs_v}{wvar} harbor_adapter/cluster/pbs_job.sh")

    # Cost estimate (rough per-run focal costs for 30 days; 90 days ≈ 3×).
    print(f"\n# Cost estimate (USD; per run = focal(X) + focal(Y) + bg {args.bg} ${bg_cost:.0f}, 30-day costs "
          f"{FOCAL_COST}; 90-day = ×{HORIZON_MULT[90]:.0f})")
    n_mult = len(seeds) * len(configs)
    print(f"#   {'model':10s} seat-runs   30-day    90-day")
    for m in models:
        n = sum((x == m) + (y == m) for x, y in runs) * n_mult
        c = n * FOCAL_COST.get(m, 0.0)
        print(f"#   {m:10s} {n:9d} ${c:8.1f} ${c * HORIZON_MULT[90]:8.1f}")
    bg_total = len(runs) * n_mult * bg_cost
    total = total_cost(runs, args.bg, FOCAL_COST, bg_cost, n_mult)
    print(f"#   {'background':10s} {len(runs) * n_mult:9d} ${bg_total:8.1f} ${bg_total * HORIZON_MULT[90]:8.1f}")
    print(f"#   TOTAL this schedule ({len(runs) * n_mult} sims): 30-day ${total:,.0f} | 90-day ${total * HORIZON_MULT[90]:,.0f}"
          f"   <- this run uses {args.days} days")
    rr = ordered_runs(round_robin_pairs(models), [])
    print("#   per seed per arena, 30-day | 90-day:")
    print(f"#     round-robin  {len(rr):3d} runs  ${total_cost(rr, args.bg, FOCAL_COST, bg_cost, 1):,.0f} | "
          f"${total_cost(rr, args.bg, FOCAL_COST, bg_cost, HORIZON_MULT[90]):,.0f}")
    if len(models) > 4 and (len(models) * 4) % 2 == 0:
        k4 = ordered_runs(k_regular_pairs(models, 4, random.Random(args.rng_seed)), [])
        print(f"#     k=4          {len(k4):3d} runs  ${total_cost(k4, args.bg, FOCAL_COST, bg_cost, 1):,.0f} | "
              f"${total_cost(k4, args.bg, FOCAL_COST, bg_cost, HORIZON_MULT[90]):,.0f}")
    for m in mirror:
        c = run_cost(m, m, args.bg, FOCAL_COST, bg_cost)
        print(f"#     mirror {m:9s} 1 run  ${c:,.0f} | ${c * HORIZON_MULT[90]:,.0f}")
    unknown = sorted({m for x, y in runs for m in (x, y)} - set(FOCAL_COST))
    if unknown:
        print(f"#   WARNING: no cost estimate for {unknown} (counted as $0)")


if __name__ == "__main__":
    main()
