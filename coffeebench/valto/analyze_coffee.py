"""Mine an enforceable Valto policy from the labeled decision ledger.

    uv run python -m coffeebench.valto.analyze_coffee \
        --ledger trajectories/<obs>/seed_0/valto_ledger.db \
        --policy-out coffeebench/valto/policies/learned.yaml

Architecture — a deliberate GENERATOR / VALIDATOR split (see the design
discussion in the chat + VALTO_EXPERIMENT_PLAN.md):

    propose(train_rows) ─▶ candidate rules ─▶ backtest(rule, eval_rows) ─▶ keep?

* GENERATORS propose candidate rules (concrete match + verdict). Today these are
  interpretable hand templates with grid-searched thresholds. A decision-tree
  rule-learner or an LLM rule-proposer is a drop-in additional generator later —
  the ONLY contract is "return candidate rules"; nothing else changes.
* The VALIDATOR is fixed and strict: a rule ships only if its matches on the
  EVAL rows are net-negative utility (n ≥ MIN_MATCHES, waste-precision ≥
  MIN_PRECISION, total_utility < 0). This is what keeps every generator honest —
  the LLM proposes, the held-out data disposes.

Each decision's utility is `metrics.utility` written by the labeling pass
(coffeebench/valto/labeling.py). Rules are emitted in the Valto policy grammar
(`context_gt`/`context_lt`/`counterparties`), and every reason quotes the
measured stats — the distilled ledger speaking to the agent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import yaml

MIN_MATCHES = 8
MIN_PRECISION = 0.60


# ---------------------------------------------------------------------------
# Load labeled rows from one or more Valto ledgers
# ---------------------------------------------------------------------------
def load_rows(ledger_paths: list[str]) -> list[dict]:
    # Read the ledger directly with agent_id=None so we get EVERY decision
    # (the client's decisions() filters by its own actor — wrong for a miner
    # that doesn't know which actor was governed).
    from valto.ledger import Ledger

    rows: list[dict] = []
    for i, p in enumerate(ledger_paths):
        led = Ledger(p)
        for r in led.list_decisions(agent_id=None, limit=1_000_000):
            met = r.get("metrics") or {}
            if "utility" not in met:
                continue  # unlabeled — skip
            rows.append(
                {
                    "ledger": i,
                    "action": r["action"],
                    "counterparty": r.get("counterparty"),
                    "amount": r.get("amount"),
                    "context": r.get("context") or {},
                    "utility": float(met["utility"]),
                    "waste": float(met.get("waste", 0.0) or 0.0),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Match semantics — mirror valto.engine._rule_matches for the keys we emit, so
# a proposed rule can be back-tested on held-out rows without the live engine.
# ---------------------------------------------------------------------------
def _row_matches(match: dict, row: dict) -> bool:
    ctx = row["context"]
    if "actions" in match and row["action"] not in match["actions"]:
        return False
    if "counterparties" in match and row["counterparty"] not in match["counterparties"]:
        return False
    amt = row["amount"]
    if "amount_gt" in match and not (amt is not None and amt > match["amount_gt"]):
        return False
    if "amount_lt" in match and not (amt is not None and amt < match["amount_lt"]):
        return False
    for key, thr in match.get("context_gt", {}).items():
        v = ctx.get(key)
        if not isinstance(v, (int, float)) or not v > thr:
            return False
    for key, thr in match.get("context_lt", {}).items():
        v = ctx.get(key)
        if not isinstance(v, (int, float)) or not v < thr:
            return False
    return True


def stats(matched: list[dict]) -> dict:
    n = len(matched)
    waste = [r for r in matched if r["utility"] < 0]
    tot = sum(r["utility"] for r in matched)
    return {
        "n": n,
        "precision": round(len(waste) / n, 3) if n else 0.0,
        "total_utility": round(tot, 2),
        "mean_utility": round(tot / n, 3) if n else 0.0,
    }


def backtest(match: dict, eval_rows: list[dict]) -> dict:
    return stats([r for r in eval_rows if _row_matches(match, r)])


# ---------------------------------------------------------------------------
# GENERATORS — each returns candidate rules proposed from `train` rows.
# A candidate is (rule_id, verdict, reason_template, match, suggest?).
# ---------------------------------------------------------------------------
def gen_overpay(train: list[dict]) -> list[dict]:
    """Bidding above the market median erodes margin. Grid over the cut that is
    worst on train; deny (or modify down) above it."""
    buys = [r for r in train if r["action"] == "buy_green"]
    out = []
    best = None
    for cut in (1.05, 1.1, 1.2, 1.35, 1.5):
        m = {"actions": ["buy_green"], "context_gt": {"price_to_median": cut}}
        s = stats([r for r in buys if _row_matches(m, r)])
        if s["n"] >= MIN_MATCHES and (best is None or s["total_utility"] < best[1]["total_utility"]):
            best = (cut, s, m)
    if best:
        cut, _s, m = best
        out.append(
            {
                "id": "green_above_market",
                "verdict": "deny",
                "reason": (
                    "Green bid is >{cut:.0f}% of the market median. {n} such "
                    "historical bids netted ${total_utility:.2f} (waste "
                    "precision {precision:.0%}). Re-bid at or below the median."
                ),
                "match": m,
                "_fmt": {"cut": cut * 100},
            }
        )
    return out


def gen_overcommit(train: list[dict]) -> list[dict]:
    """Buying / roasting when the inventory cap headroom is already thin tends
    to end in spoilage. Grid the headroom threshold."""
    out = []
    for action in ("buy_green", "roast"):
        rows = [r for r in train if r["action"] == action]
        best = None
        for thr in (5, 10, 20, 30, 40):
            m = {"actions": [action], "context_lt": {"cap_headroom_kg": thr}}
            s = stats([r for r in rows if _row_matches(m, r)])
            if s["n"] >= MIN_MATCHES and (best is None or s["total_utility"] < best[1]["total_utility"]):
                best = (thr, s, m)
        if best:
            thr, _s, m = best
            out.append(
                {
                    # deny, not require_approval: this experiment is fully
                    # automated (no human approver), so the only enforceable
                    # verdicts are deny (block) and modify (block + suggestion).
                    "id": f"overcommit_{action}",
                    "verdict": "deny",
                    "reason": (
                        "Committed holdings leave <{thr} kg of cap headroom; {n} "
                        "such {action}s netted ${total_utility:.2f} (mostly "
                        "spoilage). Clear inventory before committing more."
                    ),
                    "match": m,
                    "_fmt": {"thr": thr, "action": action},
                }
            )
    return out


def gen_solvency(train: list[dict]) -> list[dict]:
    """Spends that drop the cash runway below a buffer risk the DNF; the run
    ends if the focal agent goes bankrupt."""
    out = []
    best = None
    for thr in (10, 20, 30, 45):
        m = {"context_lt": {"cash_runway_days": thr}}
        s = stats([r for r in train if _row_matches(m, r)])
        if s["n"] >= MIN_MATCHES and (best is None or s["total_utility"] < best[1]["total_utility"]):
            best = (thr, s, m)
    if best:
        thr, _s, m = best
        out.append(
            {
                "id": "solvency_buffer",
                "verdict": "deny",
                "reason": (
                    "Cash runway would fall below {thr} days; {n} such spends "
                    "netted ${total_utility:.2f}. Preserve solvency — a focal-agent "
                    "bankruptcy ends the run."
                ),
                "match": m,
                "_fmt": {"thr": thr},
            }
        )
    return out


def gen_counterparty(train: list[dict]) -> list[dict]:
    """Any green seller whose observed economics are net-negative overall."""
    out = []
    sellers = sorted({r["counterparty"] for r in train
                      if r["action"] == "buy_green" and r["counterparty"]})
    for s_id in sellers:
        m = {"actions": ["buy_green"], "counterparties": [s_id]}
        s = stats([r for r in train if _row_matches(m, r)])
        if s["n"] >= MIN_MATCHES and s["total_utility"] < 0 and s["precision"] >= MIN_PRECISION:
            out.append(
                {
                    "id": f"seller_{s_id}_unprofitable",
                    "verdict": "modify",
                    "reason": (
                        "Ledger: buys from {seller} netted ${total_utility:.2f} over "
                        "{n} deals (waste {precision:.0%}). Source elsewhere."
                    ),
                    "match": m,
                    "suggest": {"note": f"avoid {s_id}"},
                    "_fmt": {"seller": s_id},
                }
            )
    return out


GENERATORS: list[Callable[[list[dict]], list[dict]]] = [
    gen_overpay,
    gen_overcommit,
    gen_solvency,
    gen_counterparty,
]


# ---------------------------------------------------------------------------
# Mine: propose on train, VALIDATE on eval (held-out when available)
# ---------------------------------------------------------------------------
def mine(train: list[dict], eval_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    emitted: list[dict] = []
    report: list[dict] = []
    seen_ids: set[str] = set()
    for gen in GENERATORS:
        for cand in gen(train):
            if cand["id"] in seen_ids:
                continue
            seen_ids.add(cand["id"])
            s = backtest(cand["match"], eval_rows)  # validate on held-out rows
            keep = (
                s["n"] >= MIN_MATCHES
                and s["precision"] >= MIN_PRECISION
                and s["total_utility"] < 0
            )
            report.append({"id": cand["id"], "stats": s, "emitted": keep})
            if keep:
                fmt = {**cand.get("_fmt", {}), **s}
                emitted.append(
                    {
                        "id": cand["id"],
                        "verdict": cand["verdict"],
                        "reason": cand["reason"].format(**fmt),
                        "match": cand["match"],
                        **({"suggest": cand["suggest"]} if cand.get("suggest") else {}),
                    }
                )
    # Order: hard denials before require_approval before modify.
    order = {"deny": 0, "require_approval": 1, "modify": 2}
    emitted.sort(key=lambda r: order.get(r["verdict"], 3))
    return emitted, report


def split_train_eval(rows: list[dict], train_ledgers: set[int] | None):
    """Held-out split by ledger index when multiple ledgers (seeds) are given;
    otherwise train == eval (flagged; single-ledger runs can't hold out)."""
    if train_ledgers:
        train = [r for r in rows if r["ledger"] in train_ledgers]
        ev = [r for r in rows if r["ledger"] not in train_ledgers]
        if train and ev:
            return train, ev, True
    return rows, rows, False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", nargs="+", required=True,
                   help="One or more Valto ledger .db paths (multiple = seeds).")
    p.add_argument("--train-ledgers", default=None,
                   help="Comma-separated ledger indices (0-based) to TRAIN on; "
                        "the rest are held out for validation. Omit ⇒ no holdout.")
    p.add_argument("--policy-out", type=Path,
                   default=Path("coffeebench/valto/policies/learned.yaml"))
    p.add_argument("--report-out", type=Path,
                   default=Path("coffeebench/valto/policies/LEARNED_POLICY_REPORT.md"))
    args = p.parse_args()

    rows = load_rows(args.ledger)
    print(f"[analyze] loaded {len(rows)} labeled decisions from {len(args.ledger)} ledger(s)")
    if not rows:
        print("[analyze] no labeled decisions — run the labeling pass first.")
        return

    train_ledgers = (
        {int(x) for x in args.train_ledgers.split(",")} if args.train_ledgers else None
    )
    train, ev, held_out = split_train_eval(rows, train_ledgers)
    print(f"[analyze] train={len(train)} eval={len(ev)} held_out={held_out}")

    rules, report = mine(train, ev)

    from valto.engine import validate_policy

    policy = {
        "version": "coffee-learned-v1",
        "budgets": {},
        "categories": {"buy_green": "procurement", "roast": "production"},
        "rules": rules,
        "default": "approve",
    }
    validate_policy(policy)
    args.policy_out.parent.mkdir(parents=True, exist_ok=True)
    args.policy_out.write_text(yaml.safe_dump(policy, sort_keys=False, width=100))
    print(f"[analyze] wrote {args.policy_out} with {len(rules)} rule(s)")

    lines = [
        "# Learned CoffeeBench policy report", "",
        f"Labeled decisions: {len(rows)} (train {len(train)} / eval {len(ev)}, "
        f"held-out={held_out}).", "",
        "| candidate | n (eval) | waste precision | total utility | emitted |",
        "|---|---:|---:|---:|---|",
    ]
    for c in report:
        s = c["stats"]
        lines.append(
            f"| {c['id']} | {s['n']} | {s['precision']:.0%} | "
            f"${s['total_utility']:.2f} | {'yes' if c['emitted'] else 'no'} |"
        )
    if not report:
        lines.append("| (no candidates cleared the minimum-evidence bar) | | | | |")
    args.report_out.write_text("\n".join(lines) + "\n")
    print(f"[analyze] wrote {args.report_out}")
    for c in report:
        s = c["stats"]
        print(f"  {'EMIT' if c['emitted'] else 'skip'} {c['id']:<28} "
              f"n={s['n']:<4} prec={s['precision']:.0%} util=${s['total_utility']:.2f}")


if __name__ == "__main__":
    main()
