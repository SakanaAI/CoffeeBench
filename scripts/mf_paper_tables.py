"""LaTeX tables for the camera-ready (mini-swe-agent action format, env v2.1, background Claude Sonnet 5.5).

    uv run python scripts/mf_paper_tables.py

Writes paper/tables_cr/{main,kpi,selfplay,monitors}.tex and prints the numbers quoted in the text.
NI = audit.annual.true_net_income; mean ± population sd over seeds 0-2. Misconduct tiers come from
scripts/mf_sham_detect.py (extreme / soft) with manual adjudication in results/mf_sham_manual.json.
"""

from __future__ import annotations

import glob
import itertools
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mf_sham_detect import MANUAL, tiers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "tables_cr"
MAIN_CFG = "hb_roaster_none_E_llmrival_v21"
D30 = "hb_roaster_d30_persona_none_E_llmrival"
NAME = {"Sonnet55": "Claude~Sonnet~5.5", "Astra": "GPT-6~Astra", "Opus55": "Claude~Opus~5.5", "Haiku55": "Claude~Haiku~5.5",
        "K31P": "Kimi~K3.1~Pro", "GLM53": "GLM-5.3", "DSv41F": "DeepSeek~V4.1~Flash", "Gemini38F": "Gemini~3.8~Flash",
        "heuristicv2": "HeuristicRoaster", "passive": "PassiveRoaster"}
EFFORT_MAX = {"Gemini38F": "high"}  # highest effort setting per model (launch_sim.FOCAL_EFFORT); all others max
LLM = ["Sonnet55", "Astra", "Opus55", "Haiku55", "K31P", "GLM53", "DSv41F", "Gemini38F"]


def load(d: str) -> dict:
    return json.load(open(Path(d, "run.json")))


def fmt(v: list[float], sign: bool = True, dec: int = 0, bold: bool = False) -> str:
    m, s = st.mean(v), st.pstdev(v)
    num = f"{m:+,.{dec}f}" if sign else f"{m:,.{dec}f}"
    num = num.replace(",", "{,}")
    if bold:  # best value in its column
        num = f"\\mathbf{{{num}}}"
    return f"${num}$ & \\sd{{{f'{s:,.{dec}f}'.replace(',', '{,}')}}}"


def best(rows, idx: int) -> int:
    """Index of the row with the highest mean in column `idx` of (name, ni, rev, gm, ...) tuples."""
    return max(range(len(rows)), key=lambda i: st.mean(rows[i][idx]))


def steps(d: str, agent: str = "roaster_A") -> tuple[int, int]:
    """(simulation actions excluding waits, send_message calls) of `agent`."""
    n = dm = 0
    for line in open(Path(d, "run.events.jsonl")):
        if '"agent_step"' not in line or f'"{agent}"' not in line:
            continue
        e = json.loads(line)
        if e.get("type") != "agent_step" or e.get("agent_id") != agent:
            continue
        a = e.get("action")
        n += a not in ("wait_for_next_day",)
        dm += a == "send_message"
    return n, dm


def main_table() -> None:
    rows = []
    for m in LLM + ["heuristicv2", "passive"]:
        ds = sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{MAIN_CFG}/seed_*")))
        ds = [d for d in ds if Path(d, "run.json").exists()]
        lo = [d for d in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}-min/{MAIN_CFG}/seed_*")))
              if Path(d, "run.json").exists()]
        ni, rev, gm, act, dms, cost = [], [], [], [], [], []
        for d in ds:
            a = load(d)["result"]["agents"]["roaster_A"]
            an = a["audit"]["annual"]
            ni.append(an["true_net_income"])
            rev.append(an["true_revenue"])
            gm.append(100 * (an["true_revenue"] - an["true_cogs"]) / an["true_revenue"] if an["true_revenue"] else 0.0)
            n, dm = steps(d)
            act.append(n)
            dms.append(dm)
            cost.append((a.get("usage") or {}).get("cost", 0.0))
        nlo = [load(d)["result"]["agents"]["roaster_A"]["audit"]["annual"]["true_net_income"] for d in lo]
        rows.append((m, ni, rev, gm, act, dms, cost, nlo))
        print(f"{m:11s} NI {st.mean(ni):8.0f} rev {st.mean(rev):7.0f} GM {st.mean(gm):5.1f}% act {st.mean(act):6.0f} "
              f"DM {st.mean(dms):5.0f} ${st.mean(cost):6.1f} low {st.mean(nlo) if nlo else float('nan'):7.0f}")
    lines = []
    llm = [r for r in rows if r[0] not in ("heuristicv2", "passive")]
    top = {k: llm[best(llm, k)][0] for k in (1, 2, 3)}  # best LLM per column: net income, revenue, margin
    for i, (m, ni, rev, gm, act, dms, cost, nlo) in enumerate(rows):
        if m == "heuristicv2":
            lines.append("\\midrule")
        base = m in ("heuristicv2", "passive")
        c = "\\multicolumn{2}{c}{---}" if base else fmt(cost, sign=False, dec=1)
        label = NAME[m] if base else f"{NAME[m]} ({EFFORT_MAX.get(m, 'max')})"
        lines.append(f"{label} & {fmt(ni, bold=top[1] == m)} & {fmt(rev, sign=False, bold=top[2] == m)} & "
                     f"{fmt(gm, sign=False, bold=top[3] == m)} & {c} \\\\")  # actions / DMs stay in the effort table only
    (OUT / "main.tex").write_text("\n".join(lines) + "\n")
    effort_table(rows)


def usage(d: str) -> dict:
    return load(d)["result"]["agents"]["roaster_A"].get("usage") or {}


def effort_table(rows) -> None:
    """Appendix table: Table 2 columns for each model at its highest and at its lowest (low) effort, plus actions, DMs
    and LLM calls (input tokens are printed only, quoted in the text)."""
    stats = []  # per model: highest effort, then low effort
    for m, *_ in rows:
        if m in ("heuristicv2", "passive"):
            continue
        for tag, eff in ((m, EFFORT_MAX.get(m, "max")), (f"{m}-min", "low")):
            ds = [d for d in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{tag}/{MAIN_CFG}/seed_*")))
                  if Path(d, "run.json").exists()]
            ni, rev, gm, act, dms, cost = [], [], [], [], [], []
            for d in ds:
                a = load(d)["result"]["agents"]["roaster_A"]
                an = a["audit"]["annual"]
                ni.append(an["true_net_income"])
                rev.append(an["true_revenue"])
                gm.append(100 * (an["true_revenue"] - an["true_cogs"]) / an["true_revenue"] if an["true_revenue"] else 0.0)
                n, dm = steps(d)
                act.append(n)
                dms.append(dm)
                cost.append((a.get("usage") or {}).get("cost", 0.0))
            u = [usage(d) for d in ds]
            calls = [x.get("n_calls", 0) for x in u]
            print(f"EFFORT {tag:14s} NI {st.mean(ni):7.0f} cost {st.mean(cost):6.1f} "
                  f"calls {st.mean(calls):5.0f} in {st.mean(x.get('total_input_tokens', 0) for x in u) / 1e6:6.1f}M")
            stats.append((f"{NAME[m]} ({eff})", ni, rev, gm, act, dms, calls, cost))
    top = {k: best(stats, k) for k in (1, 2, 3)}
    lines = []
    for i, (label, ni, rev, gm, act, dms, calls, cost) in enumerate(stats):
        lines.append(f"{label} & {fmt(ni, bold=top[1] == i)} & "
                     f"{fmt(rev, sign=False, bold=top[2] == i)} & {fmt(gm, sign=False, bold=top[3] == i)} & "
                     f"{fmt(act, sign=False)} & {fmt(dms, sign=False)} & {fmt(calls, sign=False)} & "
                     f"{fmt(cost, sign=False, dec=1)} \\\\")
    (OUT / "effort.tex").write_text("\n".join(lines) + "\n")


def tier_counts(pattern: str) -> tuple[int, int, int]:
    t = []
    for d in sorted(glob.glob(str(ROOT / pattern))):
        if not Path(d, "run.json").exists():
            continue
        rel = "/".join(d.split("/")[-4:])
        t.append(MANUAL.get(rel) or tiers(load(d)))
    return t.count("extreme"), t.count("soft"), len(t)


def cell(pattern: str) -> str:
    e, s, n = tier_counts(pattern)
    if n == 0:
        return "---"
    return f"{e}\\,/\\,{s}" if (e or s) else "\\textcolor{gray}{0\\,/\\,0}"


def kpi_table() -> None:
    rivals = ["Astra", "Gemini38F", "K31P", "Sonnet55"]
    rev = [("Hint (revenue only, any means)", "kpi_both_v21"), ("\\quad $-$ ``any means''", "kpinolat_both_v21"),
           ("Bare target", "kpirevonly_both_v21"), ("Net-income score $+$ target", "kpirevni_both_v21"),
           ("Hint $+$ audit warning", "kpiaudit_both_v21"), ("Hint $+$ fraud prohibition", "kpiethics_both_v21")]
    ni = [("Hint, with revaluation", "kpini_both_v21"), ("Hint, without revaluation", "kpini_both_v21lenient"),
          ("Net-income score $+$ target, with reval.", "kpinimin_both_v21")]
    lines = ["\\multicolumn{5}{l}{\\emph{Revenue target (\\$20K)}} \\\\"]
    for label, cfg in rev:
        cs = [cell(f"results/hb_mf[1-6]/H2H_Sonnet55-mswe/Astra_vs_{r}/{D30}_{cfg}/seed_*") for r in rivals]
        lines.append(f"{label} & " + " & ".join(cs) + " \\\\")
        print("KPI", cfg, cs)
    lines += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Net-income target (\\$8K)}} \\\\"]
    for label, cfg in ni:
        cs = [cell(f"results/hb_mf[1-6]/H2H_Sonnet55-mswe/Astra_vs_{r}/{D30}_{cfg}/seed_*") for r in rivals]
        lines.append(f"{label} & " + " & ".join(cs) + " \\\\")
        print("KPI", cfg, cs)
    (OUT / "kpi.tex").write_text("\n".join(lines) + "\n")


def selfplay_table() -> None:
    lines = []
    for m in LLM:
        r = cell(f"results/hb_mf[1-6]/H2H_Sonnet55-mswe/{m}_vs_{m}/{D30}_kpi_both_v21/seed_*")
        n = cell(f"results/hb_mf[1-6]/H2H_Sonnet55-mswe/{m}_vs_{m}/{D30}_kpini_both_v21lenient/seed_*")
        lines.append(f"{NAME[m]} & {r} & {n} \\\\")
        print("SELF", m, r, n)
    (OUT / "selfplay.tex").write_text("\n".join(lines) + "\n")
    for cfg in ("kpi_all_v21", "kpini_all_v21", "kpini_all_v21lenient"):
        print("ALLFIRM", cfg, tier_counts(f"results/hb_mf[36]/H2H_PersonaAstra-mswe/Astra_vs_Astra/{D30}_{cfg}/seed_*"))


def mitigation_numbers() -> None:
    """Table 11 (typed by hand): Roaster A's revenue and net income per rival."""
    for cfg in ("kpi_both_v21", "kpiaudit_both_v21", "kpiethics_both_v21"):
        for riv in ("Astra", "K31P"):
            rv, ni = [], []
            for d in sorted(glob.glob(str(ROOT / f"results/hb_mf[14]/H2H_Sonnet55-mswe/Astra_vs_{riv}/{D30}_{cfg}/seed_*"))):
                an = load(d)["result"]["agents"]["roaster_A"]["audit"]["annual"]
                rv.append(an["true_revenue"])
                ni.append(an["true_net_income"])
            print(f"MITIG {cfg:20s} B={riv:6s} Roaster A revenue {st.mean(rv):8.0f} NI {st.mean(ni):7.0f} (n={len(ni)})")


def robustness() -> None:
    def tau(a, b):
        c = sum((a[i] - a[j]) * (b[i] - b[j]) > 0 for i, j in itertools.combinations(range(len(a)), 2))
        d = sum((a[i] - a[j]) * (b[i] - b[j]) < 0 for i, j in itertools.combinations(range(len(a)), 2))
        return (c - d) / (len(a) * (len(a) - 1) / 2)

    def mean_ni(pat):
        v = [load(d)["result"]["agents"]["roaster_A"]["audit"]["annual"]["true_net_income"]
             for d in glob.glob(str(ROOT / pat)) if Path(d, "run.json").exists()]
        return st.mean(v) if v else None
    ms5 = ["Sonnet55", "Astra", "Opus55", "Gemini38F", "K31P"]
    q = [mean_ni(f"results/hb_mf4/PersonaQwen38F-off-mswe/{m}/{D30}_v21/seed_*") for m in ms5]
    s = [mean_ni(f"results/hb_ms2/PersonaSonnet55-mswe/{m}/{D30}_v21/seed_*") for m in ms5]
    print("ROBUST Qwen-bg vs Sonnet-bg (30 d, 5 LLMs): tau", round(tau(q, s), 2), [round(x) for x in q], [round(x) for x in s])


def monitors() -> None:
    recs = defaultdict(dict)
    for line in open(ROOT / "results" / "mf_llm_monitor.jsonl"):
        r = json.loads(line)
        recs[r["run"]][r["monitor"]] = (r, r.get("_cost") or 0.0)
    mons = [("openai/gpt-6-astra", "GPT-6~Astra"), ("openai/gpt-6-luna", "GPT-6~Luna"),
            ("anthropic/claude-sonnet-5.5", "Claude~Sonnet~5.5"), ("anthropic/claude-haiku-5.5", "Claude~Haiku~5.5")]
    tally = {m: Counter() for m, _ in mons}
    cost = Counter()
    for run, by in recs.items():
        r0 = next(iter(by.values()))[0]
        if r0["group"].startswith("no KPI"):
            tier = "control"
        else:
            rel = "/".join(run.split("/")[-4:])
            tier = MANUAL.get(rel) or tiers(load(str(ROOT / run)))
        for m, _ in mons:
            if m in by:
                tally[m][tier, "n"] += 1
                tally[m][tier, "flag"] += bool(by[m][0].get("misconduct"))
                cost[m] += by[m][1]
    lines = []
    for m, label in mons:
        t = tally[m]
        cells = [f"{t[k, 'flag']}/{t[k, 'n']}" for k in ("extreme", "soft", "none", "control")]
        lines.append(f"{label} & " + " & ".join(cells) + f" & {cost[m]:.2f} \\\\")
        print("MON", label, cells, f"${cost[m]:.2f}")
    (OUT / "monitors.tex").write_text("\n".join(lines) + "\n")


def all_firms() -> None:
    """Net income of all six firms (mean over three runs) by the model running Roaster A (main 90-day runs)."""
    firms = ["farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B"]
    lines = []
    for m in LLM + ["heuristicv2", "passive"]:
        ds = [d for d in sorted(glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{MAIN_CFG}/seed_*")))
              if Path(d, "run.json").exists()]
        ag = [load(d)["result"]["agents"] for d in ds]
        v = [st.mean(a[f]["audit"]["annual"]["true_net_income"] for a in ag) for f in firms]
        if m == "heuristicv2":
            lines.append("\\midrule")
        cells = [f"{x / 1000:+.1f}".replace("+", "") if x < 0 else f"{x / 1000:.1f}" for x in v + [sum(v)]]
        cells = [c.replace("-", "$-$") for c in cells]
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
        print("ALL", m, [round(x) for x in v], round(sum(v)))
    (OUT / "allfirms.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    all_firms()
    main_table()
    kpi_table()
    selfplay_table()
    mitigation_numbers()
    robustness()
    monitors()
