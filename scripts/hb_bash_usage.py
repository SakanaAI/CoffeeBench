"""Shell-command usage of focal agents in synced Harbor runs (complements the
`coffee` tool-call distribution, which comes from env agent_step events).

Reads each focal agent's terminus trajectory (results/hb_seed*/<roster>/<focal>/
hb_<role>/harbor/**/focal/*/agent/trajectory.json), splits every LLM response's
keystrokes into shell commands and classifies them:

  coffee            a `coffee <tool>` call typed directly
  coffee_scripted   `coffee` invoked inside a loop/pipeline/script line
                    (e.g. `for … do coffee …`, `python3 … subprocess … coffee`)
  notes             writing/reading notes in /workspace (tee/cat/echo >> files)
  analysis          python3 / jq / awk / bc / sed / grep on data or numbers
  file_ops          ls / mkdir / mv / cp / rm / head / tail / wc
  harness           inspecting the harness (coffee client source, env, ps, ss, find /)
  other             anything else (clear, echo narration, …)

Also reports commands per LLM response and `coffee` calls per response.

    uv run python scripts/hb_bash_usage.py [--root results/hb_seed0] [--roster Sonnet55] [--role roaster]

Writes results/hb_seed0/bash_usage_<roster>_<role>.csv and
paper/images_cr/bash_usage_<roster>[_farmer].pdf.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATS = ["coffee", "coffee_scripted", "notes", "analysis", "file_ops", "harness", "other"]
BASELINES = {"passive", "heuristic", "heuristicv2"}

_HARNESS = re.compile(r"/usr/local/bin/coffee|\bps\b|\bss\b|netstat|\benv\b|printenv|find\s+/(\s|$)|grep\s+-R\S*\s+\S+\s+/(usr|opt|etc)")
_ANALYSIS = re.compile(r"^(python3?|jq|awk|bc|sed|grep|sort|uniq|cut)\b")
_FILE = re.compile(r"^(ls|mkdir|mv|cp|rm|head|tail|wc|touch|stat|du)\b")
_NOTES = re.compile(r"/workspace/(?!out/)|>>?\s*\S+\.(md|txt|json|jsonl|csv|log)\b|\btee\b")


def split_commands(keystrokes: str) -> list[str]:
    """One entry per shell line; heredoc bodies are folded into their line."""
    out, buf, end = [], [], None
    for line in keystrokes.replace("\r", "").split("\n"):
        if end is not None:
            buf.append(line)
            if line.strip() == end:
                out.append("\n".join(buf))
                buf, end = [], None
            continue
        if not line.strip():
            continue
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if m:
            buf, end = [line], m.group(1)
        else:
            out.append(line)
    if buf:
        out.append("\n".join(buf))
    return [c.strip() for c in out if c.strip() and c.strip() not in ("{", "}", "C-c", "C-l")]


def classify(cmd: str) -> str:
    first = cmd.split("\n", 1)[0].strip().lstrip("{ ").strip()
    if re.match(r"^coffee\b", first):
        return "coffee"
    if "coffee " in cmd and re.search(r"\b(for|while|xargs|subprocess|os\.system)\b|\|", cmd):
        return "coffee_scripted"
    if _HARNESS.search(cmd):
        return "harness"
    if _NOTES.search(cmd):
        return "notes"
    if _ANALYSIS.match(first):
        return "analysis"
    if _FILE.match(first):
        return "file_ops"
    return "other"


def analyse(run_dir: Path) -> dict:
    cats: Counter = Counter()
    per_resp, coffee_per_resp = [], []
    examples: dict[str, str] = {}
    for tj in sorted(run_dir.glob("harbor/*/seg*/focal/*/agent/trajectory.json")):
        for step in json.loads(tj.read_text()).get("steps", []):
            calls = step.get("tool_calls") or []
            ks = "".join((c.get("arguments") or {}).get("keystrokes", "") for c in calls)
            cmds = split_commands(ks)
            if not cmds:
                continue
            per_resp.append(len(cmds))
            n_coffee = 0
            for c in cmds:
                k = classify(c)
                cats[k] += 1
                n_coffee += k in ("coffee", "coffee_scripted")
                examples.setdefault(k, c[:120].replace("\n", " ⏎ "))
            coffee_per_resp.append(n_coffee)
    n = sum(cats.values())
    return {
        "cats": cats, "n": n, "examples": examples,
        "cmds_per_resp": sum(per_resp) / len(per_resp) if per_resp else 0.0,
        "coffee_per_resp": sum(coffee_per_resp) / len(coffee_per_resp) if coffee_per_resp else 0.0,
        "multi_coffee_share": sum(1 for x in coffee_per_resp if x > 1) / len(coffee_per_resp) if coffee_per_resp else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "results" / "hb_seed0")
    ap.add_argument("--roster", default="Sonnet55")
    ap.add_argument("--role", default="roaster")
    args = ap.parse_args()
    rows = []
    for run_dir in sorted((args.root / args.roster).glob(f"*/hb_{args.role}")):
        model = run_dir.parent.name
        if model in BASELINES:
            continue
        r = analyse(run_dir)
        if r["n"] == 0:
            continue
        rows.append((model, r))
    if not rows:
        print("no focal terminal trajectories found")
        return
    out_csv = args.root / f"bash_usage_{args.roster}_{args.role}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_commands", *CATS, "cmds_per_response", "coffee_per_response", "share_responses_multi_coffee"])
        for model, r in rows:
            w.writerow([model, r["n"], *[r["cats"][c] for c in CATS], round(r["cmds_per_resp"], 2),
                        round(r["coffee_per_resp"], 2), round(r["multi_coffee_share"], 3)])
    print(f"{'model':11s} {'n':>5s} " + " ".join(f"{c[:9]:>9s}" for c in CATS) + "  cmd/resp coffee/resp multi")
    for model, r in sorted(rows, key=lambda x: -x[1]["n"]):
        pct = [100 * r["cats"][c] / r["n"] for c in CATS]
        print(f"{model:11s} {r['n']:5d} " + " ".join(f"{p:8.1f}%" for p in pct)
              + f"  {r['cmds_per_resp']:7.2f} {r['coffee_per_resp']:10.2f} {100 * r['multi_coffee_share']:5.1f}%")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows.sort(key=lambda x: x[1]["cats"]["coffee"] / x[1]["n"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.2))
    left = [0.0] * len(rows)
    for c in CATS:
        vals = [100 * r["cats"][c] / r["n"] for _, r in rows]
        ax.barh([m for m, _ in rows], vals, left=left, label=c)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("share of shell commands (%)")
    ax.set_title(f"Focal shell-command usage ({args.roster} roster, {args.role})")
    ax.legend(ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    out = ROOT / "paper" / "images_cr" / f"bash_usage_{args.roster}{'' if args.role == 'roaster' else '_' + args.role}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out, "and", out_csv)


if __name__ == "__main__":
    main()
