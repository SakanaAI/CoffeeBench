"""$0 tests for the head-to-head ("Arena") mode.

    unset VIRTUAL_ENV; uv run python scripts/test_h2h.py [--only prompt,early_stop,schedule,ratings]
    uv run python scripts/test_h2h.py --only harbor   # after the local scripted Harbor run below

prompt      hb_roaster_d30 / hb_roaster: roaster_A and roaster_B instructions are identical up to
            the seat identity (A <-> B swap); farmer H2H seats likewise.
early_stop  [run] early_stop_agents / server --early-stop-agents: roaster_B's bankruptcy ends an
            H2H run; without them (default configs) only main_agent does (unchanged behaviour).
schedule    h2h_schedule: k-regular (each model exactly k distinct opponents), round-robin,
            mirror, reps; both seat orders; deterministic.
ratings     h2h_ratings on synthetic runs (written in the launcher's on-disk layout) with known
            strengths and seat effect: run-level margin model and Elo recover the order, CIs cover
            the truth, seat effect from the margin model and mirror runs, two arenas → per-arena +
            overall tables and figures; weekly time profile (diagnostic only).
harbor      checks the local scripted run (Docker, $0):
              HARBOR_CMD=/private/tmp/hbvenv/bin/harbor SCRIPTED_DIE_AGENT=roaster_B SCRIPTED_DIE_STEP=6 \\
              uv run python harbor_adapter/launch_sim.py --config experiments/hb_roaster_d30.toml \\
                --seed 0 --scripted --focal2 scripted --bg Sonnet55 --max-days 3 --network relay
            → three Harbor jobs (focal: roaster_A, focal2: roaster_B, bg: 4 seats), output under
            H2H_none/scripted_vs_scripted, h2h.json seats, crash restart after roaster_B's trial died,
            simulation completed.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harbor_adapter"))
sys.path.insert(0, str(ROOT / "scripts"))
CFG = ROOT / "experiments"


def ok(cond: bool, msg: str) -> bool:
    print(f"  {'ok ' if cond else 'FAIL'} {msg}")
    return bool(cond)


def swap_ab(text: str, role: str) -> str:
    cap = role.capitalize()
    for a, b in ((f"{role}_A", f"{role}_B"), (f"{cap} A", f"{cap} B")):
        text = text.replace(a, "\0").replace(b, a).replace("\0", b)
    return text


def test_prompt(tmp: Path) -> bool:
    from generate_tasks import generate

    res = []
    for cfg, role in (("hb_roaster_d30.toml", "roaster"), ("hb_roaster.toml", "roaster")):
        ds = generate(CFG / cfg, tmp / "ds", "none", None, "Sonnet55", ("roaster_B",))
        a = (ds / f"{role}_A" / "instruction.md").read_text()
        b = (ds / f"{role}_B" / "instruction.md").read_text()
        res.append(ok(a != b and swap_ab(a, role) == b, f"{cfg}: {role}_A / {role}_B prompts equal up to identity"))
    # Persona bg: the opponent seat gets no persona block.
    ds = generate(CFG / "hb_roaster_d30.toml", tmp / "ds", "none", None, "Persona:Sonnet55", ("roaster_B",))
    rb = (ds / "roaster_B" / "instruction.md").read_text()
    fa = (ds / "farmer_A" / "instruction.md").read_text()
    res.append(ok("Private business profile" not in rb and "Private business profile" in fa,
                  "Persona bg: no persona for the opponent seat, persona for farmers"))
    from coffeebench.config import RunConfig
    from generate_tasks import opponent_seat

    c = RunConfig.from_toml(CFG / "hb_farmer.toml")
    res.append(ok(c.main_agent == "farmer_A" and opponent_seat(c) == "farmer_B", "farmer arena: opponent = farmer_B"))
    return all(res)


def env_for(cfg: Path, tmp: Path, early_stop: str | None = None):
    from coffeebench.main import build_run

    passive = ",".join(f"{a}:passive" for a in ("farmer_A", "farmer_B", "roaster_A", "roaster_B",
                                                 "retailer_A", "retailer_B"))
    args = Namespace(config=str(cfg), model=None, models=passive, max_days=5, seed=0, main_agent=None,
                     early_stop_agents=early_stop, output=None, output_root=str(tmp / "es"))
    env, _, _ = build_run(args, open_events=False)
    return env


def test_early_stop(tmp: Path) -> bool:
    res = []
    env = env_for(CFG / "hb_roaster_d30.toml", tmp)
    env._mark_bankrupt("roaster_B", 2, "test")
    res.append(ok(env._terminated_early and env._terminated_early["agent_id"] == "roaster_B",
                  "hb_roaster_d30: roaster_B bankruptcy ends the run (symmetric)"))
    env = env_for(CFG / "hb_roaster.toml", tmp)
    env._mark_bankrupt("roaster_B", 2, "test")
    res.append(ok(env._terminated_early is None, "hb_roaster (default): roaster_B bankruptcy does not end the run"))
    env._mark_bankrupt("roaster_A", 3, "test")
    res.append(ok(env._terminated_early and env._terminated_early["agent_id"] == "roaster_A",
                  "hb_roaster (default): main agent bankruptcy ends the run"))
    env = env_for(CFG / "hb_roaster.toml", tmp, "roaster_A,roaster_B")
    env._mark_bankrupt("roaster_B", 2, "test")
    res.append(ok(env._terminated_early is not None, "hb_roaster + --early-stop-agents (H2H launcher): symmetric"))
    return all(res)


def test_schedule(_tmp: Path) -> bool:
    import h2h_schedule as hs

    res = []
    models = hs.DEFAULT_MODELS
    p1 = hs.k_regular_pairs(models, 4, random.Random(0))
    p2 = hs.k_regular_pairs(models, 4, random.Random(0))
    deg = {m: sum(m in e for e in p1) for m in models}
    res.append(ok(p1 == p2 and len(p1) == 14 and set(deg.values()) == {4} and len(set(p1)) == 14,
                  f"k=4: 14 distinct pairings, every model 4 opponents, deterministic ({deg})"))
    runs = hs.ordered_runs(p1, ["Sonnet55"])
    res.append(ok(len(runs) == 29 and all((y, x) in runs for x, y in runs), "both seat orders + 1 mirror = 29 runs"))
    rr = hs.round_robin_pairs(models)
    res.append(ok(len(rr) == 21 and len(hs.ordered_runs(rr, [])) == 42, "round-robin: 21 pairs, 42 runs"))
    out = subprocess.run([sys.executable, str(ROOT / "harbor_adapter" / "h2h_schedule.py"), "--reps", "2",
                          "--mirror", "Qwen38F"], capture_output=True, text=True, check=True).stdout
    q = [ln for ln in out.splitlines() if ln.startswith("qsub")]
    n_sims = sum(ln.split("CB_PAIRS=")[1].split()[0].count(":") for ln in q)
    res.append(ok(n_sims == 58 and any("CB_SEEDS=1" in ln for ln in q) and "90-day" in out,
                  f"--reps 2 --mirror: {n_sims} sims in {len(q)} jobs; 30- and 90-day costs printed"))
    out90 = subprocess.run([sys.executable, str(ROOT / "harbor_adapter" / "h2h_schedule.py"), "--days", "90"],
                           capture_output=True, text=True, check=True).stdout
    res.append(ok("CB_CONFIG=experiments/hb_roaster.toml," in out90 and "walltime" not in out90.split("# Cost")[0].split("qsub lines")[1],
                  "--days 90: hb_roaster.toml, default 12 h jobs (requeue)"))
    out48 = subprocess.run([sys.executable, str(ROOT / "harbor_adapter" / "h2h_schedule.py"), "--days", "90",
                            "--walltime-h", "48"], capture_output=True, text=True, check=True).stdout
    res.append(ok("-l walltime=48:00:00" in out48 and "CB_WALLTIME_SEC=172800" in out48, "--walltime-h 48 → -l walltime + CB_WALLTIME_SEC"))
    return all(res)


# ------------------------------------------------------------------ ratings
def fake_run(root: Path, arena: str, x: str, y: str, seed: int, strength: dict, seat: float, days: int,
             rng, role: str = "roaster") -> None:
    """Writes run.json + h2h.json in the launcher layout. Per-day NI of a seat =
    strength/days + noise (+ seat/days for the first seat)."""
    sa, sb = f"{role}_A", f"{role}_B"
    ledger, ni = {}, {}
    for seat_id, m, bonus in ((sa, x, seat), (sb, y, 0.0)):
        entries, tot = [], 0.0
        for d in range(days):
            v = round((strength[m] + bonus) / days + rng.normal(0, 60), 2)
            tot += v
            entries.append({"day": d, "entry_type": "sale_revenue" if v >= 0 else "operating_expense",
                            "amount": abs(v), "memo": "test"})
            entries.append({"day": d, "entry_type": "cash_out", "amount": 5.0, "memo": "ignored"})
        ledger[seat_id], ni[seat_id] = entries, round(tot, 2)
    run_dir = root / "H2H_Sonnet55" / f"{x}_vs_{y}" / arena / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    agents = {a: {"audit": {"annual": {"true_net_income": ni[a]}}} for a in (sa, sb)}
    (run_dir / "run.json").write_text(json.dumps({
        "max_days": days, "truth_ledger": ledger,
        "result": {"agents": agents, "main_agent": sa, "actual_final_day": days - 1}}))
    (run_dir / "h2h.json").write_text(json.dumps({"X": x, "Y": y, "seat_X": sa, "seat_Y": sb, "config": arena,
                                                  "seed": seed}))


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def test_ratings(tmp: Path) -> bool:
    import h2h_ratings as hr
    import h2h_schedule as hs

    res = []
    rng = np.random.default_rng(1)
    truth = {"M1": 1500.0, "M2": 900.0, "M3": 400.0, "M4": 0.0, "M5": -300.0, "M6": -900.0, "M7": -1600.0}
    seat = 250.0
    models = list(truth)
    root = tmp / "hb_h2h"
    pairs = hs.round_robin_pairs(models)
    for seed in range(3):
        mirror = ["M4", "M2"] if seed < 2 else ["M4"]
        for x, y in hs.ordered_runs(pairs, mirror):
            fake_run(root, "hb_roaster", x, y, seed, truth, seat, 90, rng)
    # Second arena (farmer H2H, 30 days, k=4, fewer runs): same order, smaller scale.
    for x, y in hs.ordered_runs(hs.k_regular_pairs(models, 4, random.Random(0)), []):
        fake_run(root, "hb_farmer_d30", x, y, 0, {m: v / 3 for m, v in truth.items()}, seat / 3, 30, rng, "farmer")
    runs = hr.load_runs([str(root)])
    n90 = [r for r in runs if r.arena == "hb_roaster"]
    res.append(ok(len(n90) == 42 * 3 + 5 and all(len(r.daily_x) == 90 for r in n90),
                  f"loaded {len(runs)} runs ({len(n90)} in the 90-day arena, 90 daily NI values each)"))
    r0 = n90[0]
    res.append(ok(abs(sum(r0.daily_x) - r0.ni_x) < 0.05, "per-day truth-ledger NI sums to the audit NI"))
    res.append(ok(hr.Run("a", "X", "Y", 0, 5, 3).outcome() == 1.0 and hr.Run("a", "X", "Y", 0, 3, 5).outcome() == 0.0
                  and hr.Run("a", "X", "Y", 0, 2, 2).outcome() == 0.5, "run outcome: higher full-period NI wins, tie ½"))
    results = {a: hr.analyze([r for r in runs if r.arena == a], n_boot=200, seed=0)
               for a in ("hb_roaster", "hb_farmer_d30")}
    results["overall"] = hr.analyze(runs, n_boot=100, seed=0, standardize=True)
    rr = results["hb_roaster"]
    tv = np.array([truth[m] for m in rr["models"]])
    tv_c = tv - tv.mean()
    f = rr["fits"]
    for kind in ("margin", "elo"):
        rho = spearman(f[kind]["score"], tv)
        res.append(ok(rho >= 0.95, f"roaster arena {kind}: Spearman vs truth {rho:.2f}; order agreement "
                      f"{f[kind]['agree_np']:.1f}% (np) / {f[kind]['agree_par']:.1f}% (par)"))
    cover = np.mean((f["margin"]["lo"] <= tv_c) & (tv_c <= f["margin"]["hi"]))
    width = np.mean(f["margin"]["hi"] - f["margin"]["lo"])
    res.append(ok(cover >= 6 / 7 and 0 < width < 1500, f"margin CIs cover the truth for {cover:.0%} (mean width ${width:.0f})"))
    h, hlo, hhi = f["margin"]["seat"][0], f["margin"]["seat_lo"][0], f["margin"]["seat_hi"][0]
    res.append(ok(hlo <= seat <= hhi, f"margin seat term ${h:.0f} [{hlo:.0f}, {hhi:.0f}] covers the true ${seat:.0f}"))
    mi = rr["mirror"]["hb_roaster"]
    res.append(ok(mi["n"] == 5 and mi["margin_lo"] <= seat <= mi["margin_hi"],
                  f"mirror runs: n={mi['n']}, mean A-B ${mi['mean_margin']:.0f} [{mi['margin_lo']:.0f}, {mi['margin_hi']:.0f}]"))
    res.append(ok(abs(f["elo"]["score"].mean() - 1200.0) < 1e-6 and f["elo"]["seat"][0] > 0,
                  f"Elo centred at 1200; seat term {f['elo']['seat'][0]:.0f} Elo > 0"))
    rho = spearman(results["overall"]["fits"]["elo"]["score"], tv)
    rho_m = spearman(results["overall"]["fits"]["margin"]["score"], tv)
    res.append(ok(rho >= 0.95 and rho_m >= 0.95 and len(results["overall"]["arenas"]) == 2,
                  f"overall (2 arenas, one seat term each): Elo Spearman {rho:.2f}, margin {rho_m:.2f}"))
    tp = hr.time_profile(runs)
    w = tp["hb_roaster"]["M1"]["weekly"]
    res.append(ok(len(w) == 13 and len(tp["hb_farmer_d30"]["M1"]["weekly"]) == 5
                  and tp["hb_roaster"]["M1"]["cum"][-1] > 0 > tp["hb_roaster"]["M7"]["cum"][-1],
                  "time profile (diagnostic): 13 weekly blocks for 90 days, 5 for 30; best model ends ahead"))
    out = tmp / "h2h_out"
    hr.write_outputs(results, runs, out)
    figs = hr.figure_paths(results, tmp / "fig" / "h2h_ratings.pdf")
    for label, p in figs.items():
        hr.plot_ratings(label, results[label], p)
    hr.plot_time_profile(runs, tmp / "fig" / "h2h_time_profile.pdf")
    md = (out / "ratings.md").read_text()
    names = sorted(p.name for p in (tmp / "fig").iterdir())
    res.append(ok(all((out / n).exists() for n in ("ratings.csv", "pairings.md", "pairings.csv", "seat.csv",
                                                   "time_profile.md", "time_profile.csv"))
                  and "## overall" in md and "## hb_farmer_d30" in md and "round" not in md.lower()
                  and "not used for any ranking" in (out / "time_profile.md").read_text().lower()
                  and names == ["h2h_ratings.pdf", "h2h_ratings_hb_farmer_d30.pdf", "h2h_ratings_hb_roaster.pdf",
                                "h2h_time_profile.pdf"],
                  f"outputs: ratings/pairings/time_profile md+csv, seat.csv, figures {names}"))
    prs = (out / "pairings.csv").read_text().splitlines()
    res.append(ok(len(prs) == 1 + 21 * 3 + 5 + 14, f"pairings.csv: one row per (arena, pair, seed) ({len(prs) - 1})"))
    # CLI end to end (single arena → one ratings figure + time profile).
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "h2h_ratings.py"), str(root / "H2H_Sonnet55"),
                        "--out", str(tmp / "cli"), "--fig", str(tmp / "cli" / "h.pdf"), "--n-boot", "50"],
                       capture_output=True, text=True)
    res.append(ok(r.returncode == 0 and (tmp / "cli" / "ratings.md").exists() and (tmp / "cli" / "h.pdf").exists()
                  and (tmp / "cli" / "h2h_time_profile.pdf").exists(), f"CLI run {r.stderr[-300:]}"))
    return all(res)


def test_harbor(_tmp: Path) -> bool:
    res = []
    run_dir = ROOT / "trajectories/harbor/H2H_none/scripted_vs_scripted/hb_roaster_d30/seed_0"
    if not (run_dir / "run.json").exists():
        return ok(False, f"{run_dir}/run.json missing (run the scripted launch first)")
    meta = json.loads((run_dir / "h2h.json").read_text())
    res.append(ok(meta["seat_X"] == "roaster_A" and meta["seat_Y"] == "roaster_B", f"h2h.json seats {meta}"))
    d = json.loads((run_dir / "run.json").read_text())
    res.append(ok(d["result"].get("early_stop_agents") == ["roaster_A", "roaster_B"], "symmetric early stop passed to the server"))
    res.append(ok(d["result"]["actual_final_day"] == d["max_days"] - 1, f"completed all {d['max_days']} days"))
    restarts = json.loads((run_dir / "restarts.json").read_text()) if (run_dir / "restarts.json").exists() else []
    res.append(ok(any(r["reason"] == "crash" for r in restarts), f"crash restart recorded: {restarts}"))
    jobs = sorted((ROOT / "harbor_adapter/jobs").glob("cb_scripted_vs_scripted_none_roaster_d30_s0_2*"))
    if not jobs:
        return ok(False, "jobs dir missing")
    job = jobs[-1]
    segs = sorted(job.glob("seg*"))
    for seg in segs:
        trials = {k: sorted(p.name.split("__")[0] for p in (seg / k).glob("*__*")) for k in ("focal", "focal2", "bg")}
        res.append(ok(trials == {"focal": ["roaster_A"], "focal2": ["roaster_B"],
                                 "bg": ["farmer_A", "farmer_B", "retailer_A", "retailer_B"]},
                      f"{seg.name}: three Harbor jobs {trials}"))
    res.append(ok(len(segs) >= 2, f"{len(segs)} segments (crash restart)"))
    return all(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="prompt,early_stop,schedule,ratings")
    args = ap.parse_args()
    fns = {"prompt": test_prompt, "early_stop": test_early_stop, "schedule": test_schedule,
           "ratings": test_ratings, "harbor": test_harbor}
    results = {}
    with tempfile.TemporaryDirectory(prefix="cb_h2h_") as td:
        for name in args.only.split(","):
            print(f"[{name}]")
            results[name] = fns[name](Path(td))
            print(f"  -> {'PASS' if results[name] else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
