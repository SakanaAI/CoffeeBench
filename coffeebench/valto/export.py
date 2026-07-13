"""Export the Valto decision ledger as clean, flat raw data for offline analysis.

After a run + the labeling pass, the Valto SQLite ledger holds, per governed
decision: the full decision-time `context`, the agent's actual stated `intent`,
the verdict, and the realized-outcome `metrics`. This module flattens all of
that into one row per decision and writes:

  - `<stem>.csv`    — one row/decision, union of all context+metric keys as
                      columns (prefixed `ctx_` / `out_`), for spreadsheet / pandas.
  - `<stem>.jsonl`  — one JSON object/decision, nested context+metrics preserved,
                      for programmatic analysis.

Nothing is interpreted or scored here — it is the raw joined decision→outcome
table, exactly what you asked to analyze offline.

    uv run python -m coffeebench.valto.export \
        --ledger trajectories/<run>/seed_0/valto_ledger.db \
        --out trajectories/<run>/seed_0/valto_decisions
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# Top-level decision columns (before the flattened context/metrics).
_BASE_COLS = [
    "id", "ts", "agent_id", "action", "amount", "currency", "counterparty",
    "ctx_day", "verdict", "reason", "rule_hits", "policy_version",
    "executed", "useful", "usefulness", "usefulness_note", "intent",
]


def load_decisions(ledger_path: str) -> list[dict]:
    from valto.ledger import Ledger

    return Ledger(ledger_path).list_decisions(agent_id=None, limit=1_000_000)


def _flatten(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Return (flat_rows, ordered_columns). Context keys become `ctx_<k>`,
    metric keys `out_<k>`; the union across all rows is the column set."""
    ctx_keys: list[str] = []
    out_keys: list[str] = []
    flat: list[dict] = []
    for r in rows:
        ctx = r.get("context") or {}
        met = r.get("metrics") or {}
        rec: dict = {}
        for c in _BASE_COLS:
            v = r.get(c)
            if isinstance(v, (list, dict)):
                v = json.dumps(v, default=str)
            rec[c] = v
        for k, v in ctx.items():
            key = f"ctx_{k}"
            if key not in ctx_keys:
                ctx_keys.append(key)
            rec[key] = v
        for k, v in met.items():
            key = f"out_{k}"
            if key not in out_keys:
                out_keys.append(key)
            rec[key] = v
        flat.append(rec)
    columns = _BASE_COLS + sorted(ctx_keys) + sorted(out_keys)
    return flat, columns


def export_ledger(ledger_path: str, out_stem: str, verbose: bool = True) -> dict:
    rows = load_decisions(ledger_path)
    flat, columns = _flatten(rows)
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    csv_path = stem.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for rec in flat:
            w.writerow(rec)

    jsonl_path = stem.with_suffix(".jsonl")
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    # Per-action counts + how many rows carry a realized-outcome utility (i.e.
    # were labeled) — a quick completeness signal for the offline dataset.
    from collections import Counter

    by_action = Counter(r["action"] for r in rows)
    labeled = sum(1 for r in rows if (r.get("metrics") or {}).get("utility") is not None
                  or (r.get("metrics") or {}).get("realized_total_margin") is not None
                  or (r.get("metrics") or {}).get("fill_rate") is not None)
    report = {
        "decisions": len(rows),
        "by_action": dict(by_action),
        "labeled_with_outcomes": labeled,
        "columns": len(columns),
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
    }
    if verbose:
        print(f"[valto-export] {len(rows)} decisions → {csv_path} ({len(columns)} cols)")
        print(f"[valto-export]   by action: {dict(by_action)}")
        print(f"[valto-export]   with realized outcomes: {labeled}/{len(rows)}")
        print(f"[valto-export]   also wrote {jsonl_path}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", required=True, help="Valto ledger .db path")
    p.add_argument("--out", required=True,
                   help="Output path stem (writes <stem>.csv and <stem>.jsonl)")
    args = p.parse_args()
    export_ledger(args.ledger, args.out)


if __name__ == "__main__":
    main()
