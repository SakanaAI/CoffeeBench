"""Results table for synced Harbor runs (focal NI per model x roster, all six participants).

    uv run python scripts/hb_results_table.py [--root results/hb_seed0]

Writes <root>/results_table.csv and <root>/results_table.md. One row per
(roster, role, focal) with: focal true NI (audit.annual.true_net_income), focal
revenue, NI of all six agents, total market deals and deals involving the focal
agent, focal DMs sent, focal / total API cost (usage.cost), restarts, bankruptcies.
Also prints Kendall tau between roaster-focal and farmer-focal rankings per roster
(LLM focals present in both roles).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from itertools import combinations
from pathlib import Path

from coffeebench.audit_metrics import compute_audit_metrics

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ["farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B"]
BASELINES = {"passive", "heuristic", "heuristicv2"}


def kendall_tau(xs, ys):
    conc = disc = 0
    for i, j in combinations(range(len(xs)), 2):
        s = (xs[i] - xs[j]) * (ys[i] - ys[j])
        conc += s > 0
        disc += s < 0
    n = conc + disc
    return (conc - disc) / n if n else float("nan")


def lcnrv_writedown(run: dict, agent: str, window_days: int = 14) -> float:
    """End-of-run lower-of-cost-or-net-realisable-value write-down (IAS 2 /
    ASC 330 style), a SECONDARY robustness metric: agents were scored on and
    told about plain net income, which carries unsold inventory at cost.
    NRV per item = median unit price of non-cancelled B2B deals in the last
    `window_days` of the run (whole run if none; no write-down if the item
    never traded). Returns the write-down amount (>= 0)."""
    final = int(run.get("final_day") or run["result"].get("actual_final_day") or 89)
    by_item: dict[str, list[float]] = {}
    recent: dict[str, list[float]] = {}
    for dl in run["marketplace"]["deals"]:
        if dl.get("cancelled_day") is not None:
            continue
        p = float(dl["unit_price"])
        by_item.setdefault(dl["item_id"], []).append(p)
        if dl["deal_at"] // 1440 > final - window_days:
            recent.setdefault(dl["item_id"], []).append(p)
    nrv = {it: statistics.median(recent.get(it) or ps) for it, ps in by_item.items()}
    wd = 0.0
    for lot in run["agent_books"][agent].get("lots", []):
        ref = nrv.get(lot["item_id"])
        if ref is not None and lot["unit_cost"] > ref:
            wd += (lot["unit_cost"] - ref) * lot["qty"]
    return wd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "results" / "hb_seed0")
    args = ap.parse_args()
    rows = []
    # Layouts: <root>/<roster>/<focal>/hb_<role>/run.json (sync_hb_results.sh) or
    # <root>/<roster>/<focal>/<config>/seed_<n>/run.json (raw trajectories/harbor copy).
    for rj in sorted(list(args.root.glob("*/*/hb_*/run.json")) + list(args.root.glob("*/*/*/seed_*/run.json"))):
        parts = rj.parent.relative_to(args.root).parts
        roster, focal_model, cfg = parts[0].removeprefix("Persona"), parts[1], parts[2]
        if len(parts) == 4:  # raw layout: role from the config name, seed in the label
            cfg = "hb_" + ("farmer" if "farmer" in cfg else "roaster") + ("" if parts[3] == "seed_0" else f"_{parts[3]}")
        run = json.loads(rj.read_text())
        res = run["result"]
        focal = res["main_agent"]
        ag = res["agents"]
        n_steps_msgs = 0
        focal_deals = 0
        for line in open(rj.parent / "run.events.jsonl"):
            if '"agent_step"' in line and '"send_message"' in line:
                ev = json.loads(line)
                if ev.get("agent_id") == focal and ev.get("action") == "send_message":
                    n_steps_msgs += 1
        for dl in run["marketplace"]["deals"]:
            if focal in (dl["seller_id"], dl["buyer_id"]):
                focal_deals += 1
        restarts = 0
        if (rj.parent / "restarts.json").exists():
            restarts = len(json.loads((rj.parent / "restarts.json").read_text()))
        row = {
            "roster": roster, "role": cfg.removeprefix("hb_"), "focal_model": focal_model,
            "focal_agent": focal,
            "focal_ni": round(ag[focal]["audit"]["annual"]["true_net_income"], 2),
            "focal_revenue": round(ag[focal]["audit"]["annual"]["true_revenue"], 2),
        }
        wd = lcnrv_writedown(run, focal)
        row["focal_ni_lcnrv"] = round(row["focal_ni"] - wd, 2)  # secondary metric
        row["focal_lcnrv_writedown"] = round(wd, 2)
        # Integrity audit (coffeebench.audit_metrics): NI after the end-of-run
        # markdown to fixed reference ceilings + round-trip / stuffing flags.
        am = compute_audit_metrics(rj.parent)
        row["focal_ni_audited"] = am["audited_ni"].get(focal, {}).get("audited_net_income")
        row["round_trips"] = am["summary"]["n_round_trips"]
        row["stuffing"] = am["summary"]["n_stuffing"]
        for a in AGENTS:
            row[f"ni_{a}"] = round(ag[a]["audit"]["annual"]["true_net_income"], 2)
        row["ni_all6_sum"] = round(sum(row[f"ni_{a}"] for a in AGENTS), 2)
        row.update({
            "deals_total": res["marketplace_summary"]["total_deals"],
            "deals_focal": focal_deals,
            "dms_focal": n_steps_msgs,
            "cost_focal": round((ag[focal].get("usage") or {}).get("cost", 0.0), 2),
            "cost_total": round(sum((ag[a].get("usage") or {}).get("cost", 0.0) for a in AGENTS), 2),
            "restarts": restarts,
            "bankrupt": ",".join(run.get("bankrupt_agents") or []),
            "final_day": res.get("actual_final_day"),
        })
        rows.append(row)
    rows.sort(key=lambda r: (r["roster"], r["role"] != "roaster", r["focal_model"] in BASELINES, -r["focal_ni"]))
    out_csv = args.root / "results_table.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["roster", "role", "focal_model", "focal_ni", "focal_ni_audited", "focal_ni_lcnrv", "focal_revenue"] + [f"ni_{a}" for a in AGENTS] + [
        "ni_all6_sum", "deals_total", "deals_focal", "dms_focal", "round_trips", "stuffing", "cost_focal", "cost_total",
        "restarts", "bankrupt"]
    hdr = ["roster", "role", "focal", "focal NI", "focal NI (audited)", "focal NI (LCNRV)", "focal rev"] + [a.replace("_", " ") for a in AGENTS] + [
        "sum NI (6)", "deals", "focal deals", "focal DMs", "round trips", "stuffing", "focal $", "total $", "restarts",
        "bankrupt"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = f"{v:,.0f}" if not c.startswith("cost") else f"{v:,.2f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    taus = []
    for roster in sorted({r["roster"] for r in rows}):
        ro = {r["focal_model"]: r["focal_ni"] for r in rows if r["roster"] == roster and r["role"] == "roaster"
              and r["focal_model"] not in BASELINES}
        fa = {r["focal_model"]: r["focal_ni"] for r in rows if r["roster"] == roster and r["role"] == "farmer"
              and r["focal_model"] not in BASELINES}
        common = sorted(set(ro) & set(fa))
        if len(common) >= 3:
            t = kendall_tau([ro[m] for m in common], [fa[m] for m in common])
            taus.append(f"- {roster}: Kendall tau (roaster-focal vs farmer-focal NI ranking, n={len(common)} LLMs) = {t:.3f}")
    md = ["# Harbor seed-0 results (auto-generated by scripts/hb_results_table.py)", "",
          "Seed 0 only; NI = audit.annual.true_net_income; cost = provider-reported usage.cost (USD).", "",
          *lines, "", *taus, ""]
    (args.root / "results_table.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
