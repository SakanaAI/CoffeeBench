"""LLM-based classification of the evaluated agent's shell commands (mini-swe main runs, 90 d).

    uv run python scripts/mf_shell_classify.py [--model openai/gpt-6-luna] [--dry-run]

Plain single-line `coffee <tool> ...` commands are labelled `coffee_cli` by rule. Every other command (shell,
python, heredocs, pipelines) is classified by an LLM (default GPT-6 Luna via OpenRouter, key from .env) into:

  memo_write        writes notes / plans / ledgers to a file
  memo_read         reads back its own notes or saved files
  analysis          computes on data (python/jq/awk/bc), e.g. margins, demand, prices
  automation        a script, function or loop that calls `coffee` (or builds a reusable helper library)
  file_management   ls / mkdir / mv / rm / wc etc.
  environment_probe inspects the harness, system, processes, network or coffee CLI internals
  other             anything else

Results are cached in results/mf_shell_llm_cache.jsonl (keyed by command hash); the summary is printed and
written to results/mf_shell_llm.csv.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics as st
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hb_bash_usage import split_commands  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAJ = ROOT / "results" / "hb_mf_traj"
CACHE = ROOT / "results" / "mf_shell_llm_cache.jsonl"
SIM = re.compile(r"cb_(?P<model>[A-Za-z0-9]+(?:-min)?)_PersonaSonnet55-mswe_roaster_none_E_llmrival_v21_s(?P<seed>\d)")
CATS = ["memo_write", "memo_read", "analysis", "automation", "file_management", "environment_probe", "other"]
PLAIN_COFFEE = re.compile(r"^coffee\s+\w+(\s|$)")
PROMPT = """You classify shell commands typed by an AI agent that runs a coffee-roasting business in a simulation.
The agent acts on the simulation only through the `coffee` command-line tool; other commands are its own work.
Categories:
- memo_write: writes notes, plans, logs or ledgers to a file (cat > file, echo >> file, tee, python writing a text file)
- memo_read: reads back its own notes or previously saved files (cat/head/grep of its files)
- analysis: computes on data (python/jq/awk/bc), e.g. margins, costs, demand, price comparisons
- automation: a script, shell function or loop that calls `coffee`, or creates/uses a reusable helper library or script
- file_management: ls, mkdir, mv, rm, wc and similar
- environment_probe: inspects the harness, operating system, processes, network, environment variables or the coffee CLI source
- other: anything else
Return ONLY a JSON array with one category string per command, in order, e.g. ["memo_write","analysis"].
Commands (numbered):
"""


def key() -> str:
    for line in open(ROOT / ".env"):
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no OPENROUTER_API_KEY in .env")


def ask(model: str, cmds: list[str], api_key: str) -> list[str]:
    text = PROMPT + "\n".join(f"{i + 1}. {c[:1500]}" for i, c in enumerate(cmds))
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 2000}
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                         {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            out = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
            labels = json.loads(out[out.index("["): out.rindex("]") + 1])
            if len(labels) == len(cmds):
                return [lab if lab in CATS else "other" for lab in labels]
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, type(e).__name__, file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return ["other"] * len(cmds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-6-luna")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    runs = defaultdict(list)  # (model, seed) -> commands
    for tj in sorted(TRAJ.rglob("focal/*/agent/trajectory.json")):
        m = SIM.search(str(tj))
        if not m:
            continue
        for step in json.loads(tj.read_text()).get("steps", []):
            if step.get("source") != "agent":
                continue
            ks = "".join((c.get("arguments") or {}).get("keystrokes", "") for c in step.get("tool_calls") or [])
            runs[(m["model"], m["seed"])].extend(split_commands(ks))
    cache = {}
    if CACHE.exists():
        for line in open(CACHE):
            r = json.loads(line)
            cache[r["h"]] = r["label"]
    h = lambda c: hashlib.sha1(c.encode()).hexdigest()  # noqa: E731
    todo = sorted({c for cmds in runs.values() for c in cmds if not PLAIN_COFFEE.match(c) and h(c) not in cache})
    print(f"{sum(len(v) for v in runs.values())} commands, {len(todo)} to classify with {args.model}", file=sys.stderr)
    if args.dry_run:
        return
    api_key = key()
    with open(CACHE, "a") as fh:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i: i + args.batch]
            for c, lab in zip(chunk, ask(args.model, chunk, api_key)):
                cache[h(c)] = lab
                fh.write(json.dumps({"h": h(c), "label": lab, "cmd": c[:300]}) + "\n")
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}", file=sys.stderr)
    by_model = defaultdict(list)
    for (model, seed), cmds in sorted(runs.items()):
        cnt = Counter("coffee_cli" if PLAIN_COFFEE.match(c) else cache.get(h(c), "other") for c in cmds)
        by_model[model].append(cnt)
    cols = ["coffee_cli"] + CATS
    rows = []
    print(f"{'model':14s} {'cmds':>6s} {'non-coffee%':>11s} " + " ".join(f"{c[:10]:>10s}" for c in CATS))
    for model, cs in by_model.items():
        tot = [sum(c.values()) for c in cs]
        row = {"model": model, "cmds": st.mean(tot), "noncoffee_pct": 100 * st.mean(1 - c["coffee_cli"] / t for c, t in zip(cs, tot))}
        row.update({c: st.mean(x[c] for x in cs) for c in cols})
        rows.append(row)
        print(f"{model:14s} {row['cmds']:6.0f} {row['noncoffee_pct']:10.1f}% " + " ".join(f"{row[c]:10.1f}" for c in CATS))
    with open(ROOT / "results" / "mf_shell_llm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
