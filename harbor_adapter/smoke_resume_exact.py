"""$0 Harbor test of exact resume (CoffeeTerminus + scripted fake LLM, docker).

    unset VIRTUAL_ENV; HARBOR_CMD=/private/tmp/hbvenv/bin/harbor \
        uv run python harbor_adapter/smoke_resume_exact.py [--config experiments/camera_ready_hb/hb_roaster.toml]
    # head-to-head (three Harbor jobs: focal roaster_A, focal2 roaster_B, bg):
    ... smoke_resume_exact.py --h2h --config experiments/camera_ready_hb/hb_roaster_d30.toml

Three 4-day scripted runs (all 6 agents = ScriptedCoffeeTerminus; `--network relay`
by default = no-network containers on Docker Desktop, `none` on Linux):
  A  uninterrupted;
  B  planned stop at day 2 (--segment-days 2) → exact resume;
  C  crash: the server is SIGKILLed once latest.pkl is at day >= 2 and some agent
     has written snapshots beyond the one latest.pkl references → the launcher
     restarts from latest.pkl (frozen as day_NNN.pkl).
Checks: every LLM input (full message history + prompt, per agent and step;
the latest segment's copy of each step) of B and C equals A's; B/C resumed
trials report "Exact resume from snapshot call <id>" with the id recorded in
checkpoints/refs.json for the checkpoint resumed from (C: not a later
snapshot that existed at the crash); no resume digest reached any model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "harbor_adapter" / "jobs"
NETWORK = "relay"
AGENTS = ["farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B"]
PREFIX = "cb_scripted_none_"  # --h2h: cb_scripted_vs_scripted_none_
H2H: list[str] = []  # --h2h: ["--focal2", "scripted"] (three Harbor jobs: focal, focal2, bg)


def launch(config: str, *extra: str) -> Path:
    before = {p for p in JOBS.glob(PREFIX + "*") if p.is_dir()}
    cmd = [sys.executable, "harbor_adapter/launch_sim.py", "--config", config, "--seed", "0",
           "--scripted", "--max-days", "4", "--network", NETWORK, *H2H, *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    print("\n".join("    " + ln for ln in r.stdout.splitlines() if "[launch]" in ln and "trials" not in ln))
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"launch failed rc={r.returncode}")
    (new,) = {p for p in JOBS.glob(PREFIX + "*") if p.is_dir()} - before
    return new


def llm_inputs(run: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {a: {} for a in AGENTS}
    for seg in sorted(run.glob("seg*")):
        for f in sorted(seg.glob("*/*__*/agent/scripted_prompts.jsonl")):
            aid = f.parent.parent.name.split("__")[0]
            for ln in f.read_text().splitlines():
                d = json.loads(ln)
                out[aid][d["step"]] = {"messages": d["messages"], "prompt": d["prompt"]}
    return out


def resumed_from(run: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for traj in sorted(run.glob("seg*/*/*__*/agent/trajectory.json")):
        steps = json.loads(traj.read_text()).get("steps") or []
        if steps and str(steps[0].get("message", "")).startswith("Exact resume from snapshot call"):
            aid = traj.parent.parent.name.split("__")[0]
            out.setdefault(aid, []).append(steps[0]["message"].split("call ")[1].split(" ")[0])
    return out


def compare(ref: dict, got: dict, label: str) -> bool:
    ok = True
    for aid in AGENTS:
        a, b = ref[aid], got[aid]
        diffs = [s for s in sorted(set(a) | set(b)) if a.get(s) != b.get(s)]
        if diffs:
            ok = False
            s = diffs[0]
            print(f"  {label} {aid}: {len(diffs)} differing steps (of {len(a)}/{len(b)}), first {s}")
            x, y = a.get(s), b.get(s)
            if x and y:
                for i, (m1, m2) in enumerate(zip(x["messages"] + [x["prompt"]], y["messages"] + [y["prompt"]])):
                    if m1 != m2:
                        print(f"    msg {i}:\n    A: {str(m1)[-600:]!r}\n    B: {str(m2)[-600:]!r}")
                        break
        else:
            print(f"  {label} {aid}: {len(a)} LLM inputs identical")
    # The CLI prints the marker on a line of its own (the instruction only quotes it).
    digest = [aid for aid in AGENTS for st, d in got[aid].items()
              if st > 0 and re.search(r"^--- resume digest ---$", d["prompt"], re.M)]
    if digest:
        print(f"  {label}: resume digest seen by {sorted(set(digest))}")
    return ok and not digest


def crash_watch(run_dir: Path, stop: threading.Event, info: dict) -> None:
    """SIGKILL the seg01 server once latest.pkl is at day >= 2 and a later
    snapshot than the referenced one exists for some agent."""
    from coffeebench.server.agent_state import call_seq, load_refs

    ckpt_dir = run_dir / "checkpoints"
    t_start = time.time()
    while not stop.is_set():
        time.sleep(0.2)
        refs = (load_refs(str(ckpt_dir)).get("latest.pkl") or {}).get("calls") or {}
        latest = ckpt_dir / "latest.pkl"
        if not refs or not latest.exists() or latest.stat().st_mtime < t_start:
            continue
        from launch_sim import ckpt_day
        try:
            day = ckpt_day(str(latest))
        except Exception:  # noqa: BLE001 — being written
            continue
        if day < 2:
            continue
        later = {}
        for aid, call in refs.items():
            seqs = [int(p.stem[5:]) for p in (run_dir / "agent_state" / aid).glob("call_*.json")]
            if call_seq(call) is not None and seqs and max(seqs) > call_seq(call):
                later[aid] = (call, max(seqs))
        if not later:
            continue
        runs = sorted(JOBS.glob(PREFIX + "*/seg01/server_ready.json"),
                      key=lambda p: p.stat().st_mtime)
        pid = json.loads(runs[-1].read_text())["pid"]
        os.kill(pid, signal.SIGKILL)
        info.update(day=day, refs=dict(refs), later=later, pid=pid)
        print(f"  [crash] killed server pid {pid} at latest day {day}; later snapshots: {later}")
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/camera_ready_hb/hb_roaster.toml")
    ap.add_argument("--only", default="A,B,C")
    ap.add_argument("--reuse-a", default=None, help="Jobs dir of an earlier run A (skip A).")
    ap.add_argument("--reuse-b", default=None, help="Jobs dir of an earlier run B (skip launching B; "
                    "refs.json must still be from it).")
    ap.add_argument("--network", default="relay", choices=("none", "relay", "bridge"))
    ap.add_argument("--h2h", action="store_true",
                    help="Head-to-head mode (--focal2 scripted): three Harbor jobs, H2H output path.")
    args = ap.parse_args()
    global NETWORK, PREFIX
    NETWORK = args.network
    if args.h2h:
        PREFIX = "cb_scripted_vs_scripted_none_"
        H2H.extend(["--focal2", "scripted"])
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "harbor_adapter"))
    run_dir = (ROOT / ("trajectories/harbor/H2H_none/scripted_vs_scripted" if args.h2h else "trajectories/harbor/none/scripted")
               / Path(args.config).stem / "seed_0")
    ok = True
    print("[A] uninterrupted")
    a = Path(args.reuse_a) if args.reuse_a else launch(args.config, "--regen")
    ref = llm_inputs(a)
    print(f"  steps per agent: { {k: len(v) for k, v in ref.items()} }")
    if "B" in args.only:
        print("[B] planned stop at day 2")
        b = Path(args.reuse_b) if args.reuse_b else launch(args.config, "--segment-days", "2")
        refs = json.loads((run_dir / "checkpoints" / "refs.json").read_text())["day_002.pkl"]["calls"]
        res = resumed_from(b)
        print(f"  refs day_002: {refs}\n  resumed from: {res}")
        ok &= all(res.get(aid) == [refs[aid]] for aid in AGENTS)
        ok &= compare(ref, llm_inputs(b), "B")
    if "C" in args.only:
        print("[C] crash after day 2 → restart from latest.pkl")
        stop, info = threading.Event(), {}
        t = threading.Thread(target=crash_watch, args=(run_dir, stop, info), daemon=True)
        t.start()
        c = launch(args.config)
        stop.set()
        res = resumed_from(c)
        restarts = json.loads((run_dir / "restarts.json").read_text())
        print(f"  restarts: {restarts}\n  resumed from: {res}")
        ok &= bool(info) and any(r["reason"] == "crash" for r in restarts)
        ok &= all(res.get(aid) == [info["refs"][aid]] for aid in AGENTS if info.get("refs", {}).get(aid))
        ok &= compare(ref, llm_inputs(c), "C")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
