"""Shell usage of the evaluated agent in the mini-swe main runs (90 d, max and min effort).

    uv run python scripts/mf_shell_usage.py

Reads results/hb_mf_traj/CoffeeBench-mf{1,2}/harbor_adapter/jobs/<sim>/seg*/focal/*/agent/trajectory.json
(synced from the cluster), classifies every executed command with scripts/hb_bash_usage.classify (coffee, coffee in a
script/loop, notes, analysis, file ops, harness inspection, other), and prints per model: turns, share of
non-`coffee` commands, notes / analysis / scripted-coffee counts, number of distinct files written in
/workspace, and examples. Writes results/mf_shell_usage.csv.
"""

import csv
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hb_bash_usage import classify, split_commands  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "results" / "hb_mf_traj"
SIM = re.compile(r"cb_(?P<model>[A-Za-z0-9]+(?:-min)?)_PersonaSonnet55-mswe_roaster_none_E_llmrival_v21_s(?P<seed>\d)")
WRITE = re.compile(r"(?:>>?|\btee\s+(?:-a\s+)?)\s*(/workspace/[\w./-]+|[\w-]+\.(?:md|txt|json|csv|py|sh|log))")


def main() -> None:
    runs = defaultdict(lambda: {"cats": Counter(), "turns": 0, "files": set(), "ex": {}})
    for tj in sorted(TRAJ.rglob("focal/*/agent/trajectory.json")):
        m = SIM.search(str(tj))
        if not m:
            continue
        r = runs[(m["model"], m["seed"])]
        for step in json.loads(tj.read_text()).get("steps", []):
            if step.get("source") != "agent":
                continue
            ks = "".join((c.get("arguments") or {}).get("keystrokes", "") for c in step.get("tool_calls") or [])
            cmds = split_commands(ks)
            if not cmds:
                continue
            r["turns"] += 1
            for c in cmds:
                k = classify(c)
                r["cats"][k] += 1
                r["files"].update(WRITE.findall(c))
                if k != "coffee":
                    r["ex"].setdefault(k, c[:140].replace("\n", " ⏎ "))
    by_model = defaultdict(list)
    for (model, seed), r in sorted(runs.items()):
        by_model[model].append(r)
    rows = []
    print(f"{'model':14s} {'runs':>4s} {'turns':>6s} {'non-coffee %':>12s} {'notes':>6s} {'analysis':>8s} {'scripted':>8s} {'harness':>7s} {'files':>5s}")
    for model, rs in by_model.items():
        tot = [sum(r["cats"].values()) for r in rs]
        nonc = [1 - r["cats"]["coffee"] / t if t else 0 for r, t in zip(rs, tot)]
        f = lambda k: st.mean(r["cats"][k] for r in rs)  # noqa: E731
        row = dict(model=model, runs=len(rs), turns=st.mean(r["turns"] for r in rs), noncoffee=100 * st.mean(nonc),
                   notes=f("notes"), analysis=f("analysis"), scripted=f("coffee_scripted"), harness=f("harness"),
                   files=st.mean(len(r["files"]) for r in rs))
        rows.append(row)
        print(f"{model:14s} {row['runs']:4d} {row['turns']:6.0f} {row['noncoffee']:11.1f}% {row['notes']:6.1f} {row['analysis']:8.1f} "
              f"{row['scripted']:8.1f} {row['harness']:7.1f} {row['files']:5.1f}")
    print("\nexamples:")
    for model, rs in by_model.items():
        ex = {}
        for r in rs:
            for k, v in r["ex"].items():
                ex.setdefault(k, v)
        print(f"  {model}: " + " | ".join(f"[{k}] {v}" for k, v in ex.items() if k in ("notes", "analysis", "coffee_scripted", "harness")))
    with open(ROOT / "results" / "mf_shell_usage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
