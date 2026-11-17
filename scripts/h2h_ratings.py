"""Head-to-head ("Arena") ratings from synced H2H runs.

    uv run python scripts/h2h_ratings.py [DIR ...] [--out results/h2h]
        [--fig paper/images_cr/h2h_ratings.pdf] [--fig-time paper/images_cr/h2h_time_profile.pdf]
        [--n-boot 1000] [--prior 0.01] [--rng 0]

DIR (default: results/hb_h2h*) is searched for run dirs with a run.json under an
`H2H_<bg>/<X>_vs_<Y>/<config>/seed_N` path (layout of launch_sim --focal2 and
`scripts/sync_hb_results.sh --h2h`). Seats: <run_dir>/h2h.json (written by
launch_sim), else X = the config's main_agent seat and Y = the other seat of
that role. Arena = config name (hb_roaster_d30, hb_roaster, a farmer H2H
config ...); every arena is fitted separately, and "overall" pools them.

The unit of analysis is the RUN (agents are told their score is the
full-period net income):
  margin   m = NI(X) - NI(Y), NI = audit.annual.true_net_income (truth ledger).
  outcome  X wins (1) if NI(X) > NI(Y), loses (0) if lower, draw (½) if equal.

PRIMARY metric — margin model: least squares m = s_X - s_Y + h_arena with
sum-to-zero strengths s ($ per run; overall: margins divided by the arena's
margin SD, so s is in SD units) and a seat term h (advantage of the first seat,
e.g. roaster_A).
Elo — Bradley-Terry fitted by maximum likelihood to the run outcomes only,
P(X beats Y) = 1 / (1 + 10^-((E_X - E_Y + H) / 400)), base 1200 (mean of the
models), slope 400, as in CodeClash (arXiv 2511.00839); draws count ½; a weak
ridge (--prior, on logit strengths; 0 = pure MLE) keeps the fit finite under
perfect separation (small pilots).
Uncertainty (both): non-parametric bootstrap (resample runs within arena) →
95% percentile CIs; parametric bootstrap (outcomes resimulated from the fitted
model: Bernoulli for Elo, Gaussian residuals for margins) → pairwise order
agreement % (share of bootstrap fits × model pairs keeping the point-estimate
order), reported for both bootstraps.
Seat advantage: the fitted seat terms, and mirror runs (X vs X): mean margin
and seat-A win rate with bootstrap CIs.

Time profile (DIAGNOSTIC ONLY — never used for ranking): per-seat daily NI from
the truth ledger (same formula as the audit; sums to the run NI), averaged per
model as cumulative margin over the run and weekly margins (7-day blocks),
plus early (first half) vs late (second half) margin.

Outputs (--out): ratings.{md,csv} (per arena + overall), pairings.{md,csv}
(per pairing and seed, both seat orders), seat.csv, time_profile.{md,csv}.
Figures: --fig (one arena: that arena; several: overall) plus
<fig stem>_<arena>.pdf per arena when there are several; --fig-time = the
time-profile figure (one panel per arena).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE, SLOPE = 1200.0, 400.0
K = SLOPE / math.log(10)  # logit → Elo points
WEEK = 7
TIME_NOTE = "Diagnostic only: time profile of the run-level margin; NOT used for any ranking."


@dataclass
class Run:
    arena: str
    x: str
    y: str
    seed: int
    ni_x: float
    ni_y: float
    daily_x: list[float] = field(default_factory=list)  # per-day NI of X's seat (diagnostic)
    daily_y: list[float] = field(default_factory=list)
    path: str = ""

    @property
    def margin(self) -> float:
        return self.ni_x - self.ni_y

    @property
    def mirror(self) -> bool:
        return self.x == self.y

    def outcome(self) -> float:
        """X's run-level score: 1 higher full-period NI, 0 lower, ½ equal (to the cent)."""
        m = self.margin
        return 0.5 if abs(m) < 0.005 else float(m > 0)


# ---------------------------------------------------------------- loading
def daily_ni(truth: list[dict], last_day: int) -> list[float]:
    """Per-day NI with the audit formula (environment._compute_audit_for):
    revenue - returns - (inventory_out - cogs_reversal) - opex - net interest."""
    from coffeebench.environment import classify_opex_entry

    class _E:  # classify_opex_entry reads attributes
        def __init__(self, d):
            self.entry_type, self.memo = d.get("entry_type"), d.get("memo")

    sign = {"sale_revenue": 1, "sale_reversal": -1, "inventory_out": -1, "cogs_reversal": 1,
            "interest_expense": -1, "interest_revenue": 1}
    out = [0.0] * (last_day + 1)
    for e in truth:
        d = int(e.get("day", 0))
        if d < 0 or d > last_day:
            continue
        et, amt = e.get("entry_type"), float(e.get("amount") or 0.0)
        if et in sign:
            out[d] += sign[et] * amt
        elif classify_opex_entry(_E(e)) is not None:
            out[d] -= amt
    return out


def other_seat(seat: str) -> str:
    role, letter = seat.split("_")
    return f"{role}_{'B' if letter == 'A' else 'A'}"


def load_run(run_json: Path) -> Run | None:
    run_dir = run_json.parent
    pair = run_dir.parent.parent.name
    if "_vs_" not in pair:
        return None
    d = json.loads(run_json.read_text())
    meta_p = run_dir / "h2h.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    x, y = meta.get("X"), meta.get("Y")
    if not (x and y):
        x, y = pair.split("_vs_", 1)
    res = d["result"]
    seat_x = meta.get("seat_X") or res.get("main_agent") or "roaster_A"
    seat_y = meta.get("seat_Y") or other_seat(seat_x)
    ag = res["agents"]
    last = int(res.get("actual_final_day", d.get("max_days", 30) - 1))
    truth = d.get("truth_ledger") or {}
    dx = daily_ni(truth.get(seat_x, []), last)
    dy = daily_ni(truth.get(seat_y, []), last)
    seed = int(meta.get("seed", run_dir.name.removeprefix("seed_")))
    arena = meta.get("config") or run_dir.parent.name
    return Run(arena, x, y, seed, float(ag[seat_x]["audit"]["annual"]["true_net_income"]),
               float(ag[seat_y]["audit"]["annual"]["true_net_income"]), dx, dy, str(run_dir))


def load_runs(dirs: list[str]) -> list[Run]:
    runs, seen = [], set()
    for d in dirs:
        pats = [f"{d}/**/H2H_*/*_vs_*/*/seed_*/run.json"]
        if Path(d).name.startswith("H2H_"):  # an H2H_<bg> dir itself
            pats.append(f"{d}/*_vs_*/*/seed_*/run.json")
        for p in sorted({q for pat in pats for q in glob.glob(pat, recursive=True)}):
            rp = Path(p).resolve()
            if rp in seen:
                continue
            seen.add(rp)
            r = load_run(rp)
            if r is not None:
                runs.append(r)
    return runs


# ---------------------------------------------------------------- fitting
def design(rows: list[tuple[int, int, float, int]], n: int, n_seat: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(rows), n + n_seat))
    yv = np.zeros(len(rows))
    for r, (i, j, v, g) in enumerate(rows):
        if i != j:  # mirror runs inform only the seat term
            X[r, i] += 1.0
            X[r, j] -= 1.0
        X[r, n + g] = 1.0
        yv[r] = v
    return X, yv


def fit_bt(games: list[tuple[int, int, float, int]], n: int, n_seat: int, prior: float = 0.01,
           iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Bradley-Terry MLE (Newton) of P(i beats j) = σ(s_i - s_j + h_g) on
    games (i, j, score_i ∈ {0, ½, 1}, seat group g); ridge `prior`·|s|².
    Returns (s logit, sum-to-zero; h logit per group)."""
    if not games:
        return np.zeros(n), np.zeros(n_seat)
    X, yv = design(games, n, n_seat)
    p = n + n_seat
    reg = np.full(p, prior)
    reg[n:] = min(prior, 1e-3)  # seat terms: (nearly) flat
    th = np.zeros(p)
    for _ in range(iters):
        pr = 1.0 / (1.0 + np.exp(-(X @ th)))
        grad = X.T @ (yv - pr) - 2 * reg * th
        H = (X * (pr * (1 - pr))[:, None]).T @ X + np.diag(2 * reg) + 1e-9 * np.eye(p)
        step = np.linalg.lstsq(H, grad, rcond=None)[0]
        step = np.clip(step, -2.0, 2.0)  # damped (separation with prior 0 drifts slowly instead of exploding)
        th += step
        if np.max(np.abs(step)) < 1e-8:
            break
    s = th[:n] - th[:n].mean()
    return s, th[n:]


def fit_margin(rows: list[tuple[int, int, float, int]], n: int, n_seat: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares m = s_i - s_j + h_g (sum-to-zero s). Returns (s, h, residual SD)."""
    if not rows:
        return np.zeros(n), np.zeros(n_seat), 0.0
    X, yv = design(rows, n, n_seat)
    # Sum-to-zero constraint as an extra (heavily weighted) row: identifies s even
    # when a bootstrap sample drops a model's runs.
    C = np.zeros((1, n + n_seat))
    C[0, :n] = 1.0
    th, *_ = np.linalg.lstsq(np.vstack([X, 1e3 * C]), np.concatenate([yv, [0.0]]), rcond=None)
    s = th[:n] - th[:n].mean()
    resid = yv - X @ np.concatenate([s, th[n:]])
    dof = max(len(rows) - np.linalg.matrix_rank(X), 1)
    return s, th[n:], float(np.sqrt(resid @ resid / dof))


KINDS = ("margin", "elo")


def observations(runs: list[Run], models: list[str], arenas: list[str], scale: dict[str, float] | None):
    idx = {m: k for k, m in enumerate(models)}
    gidx = {a: k for k, a in enumerate(arenas)}
    obs: dict[str, list] = {k: [] for k in KINDS}
    for r in runs:
        i, j, g = idx[r.x], idx[r.y], gidx[r.arena]
        obs["margin"].append((i, j, r.margin / (scale[r.arena] if scale else 1.0), g))
        obs["elo"].append((i, j, r.outcome(), g))
    return obs


def fit_all(runs, models, arenas, prior, scale=None) -> dict:
    obs = observations(runs, models, arenas, scale)
    n, g = len(models), len(arenas)
    s, h, sd = fit_margin(obs["margin"], n, g)
    return {"margin": (s, h), "elo": fit_bt(obs["elo"], n, g, prior), "_margin_sd": sd, "_obs": obs}


def order_agreement(point: np.ndarray, boots: np.ndarray) -> float:
    """Mean over model pairs of the share of bootstrap fits keeping the point order (%)."""
    n = len(point)
    agree, tot = 0.0, 0
    for a in range(n):
        for b in range(a + 1, n):
            if point[a] == point[b]:
                continue
            sgn = np.sign(point[a] - point[b])
            agree += float(np.mean(np.sign(boots[:, a] - boots[:, b]) == sgn))
            tot += 1
    return 100.0 * agree / tot if tot else float("nan")


def simulate(obs_k: list, s: np.ndarray, h: np.ndarray, rng, kind: str, sd: float) -> list:
    out = []
    for i, j, _, g in obs_k:
        z = (s[i] - s[j] if i != j else 0.0) + h[g]
        if kind == "margin":
            out.append((i, j, z + rng.normal(0, sd), g))
        else:
            out.append((i, j, float(rng.random() < 1 / (1 + math.exp(-z))), g))
    return out


def boot_ci(values: np.ndarray, rng, n_boot: int, stat=np.mean) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    b = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyze(runs: list[Run], n_boot: int = 1000, seed: int = 0, prior: float = 0.01,
            standardize: bool = False) -> dict:
    """Point fits + non-parametric / parametric bootstrap for one arena (or
    the pooled overall set when `standardize`: margins / arena SD)."""
    rng = np.random.default_rng(seed)
    models = sorted({r.x for r in runs} | {r.y for r in runs})
    arenas = sorted({r.arena for r in runs})
    scale = None
    if standardize:
        scale = {a: (float(np.std([r.margin for r in runs if r.arena == a])) or 1.0) for a in arenas}
    point = fit_all(runs, models, arenas, prior, scale)
    n, g = len(models), len(arenas)
    by_arena = {a: [r for r in runs if r.arena == a] for a in arenas}
    npb = {k: (np.zeros((n_boot, n)), np.zeros((n_boot, g))) for k in KINDS}
    for b in range(n_boot):
        sample = [rs[k] for rs in by_arena.values() for k in rng.integers(0, len(rs), len(rs))]
        f = fit_all(sample, models, arenas, prior, scale)
        for k in KINDS:
            npb[k][0][b], npb[k][1][b] = f[k]
    par = {k: np.zeros((n_boot, n)) for k in KINDS}
    for b in range(n_boot):
        for k in KINDS:
            s, h = point[k]
            sim = simulate(point["_obs"][k], s, h, rng, k, point["_margin_sd"])
            par[k][b] = fit_margin(sim, n, g)[0] if k == "margin" else fit_bt(sim, n, g, prior)[0]
    res = {"models": models, "arenas": arenas, "scale": scale, "n_runs": len(runs), "fits": {},
           "margin_resid_sd": point["_margin_sd"]}
    for k in KINDS:
        s, h = point[k]
        conv = (lambda v: BASE + K * v) if k == "elo" else (lambda v: v)
        hconv = (lambda v: K * v) if k == "elo" else (lambda v: v)
        res["fits"][k] = {
            "score": conv(s), "lo": conv(np.percentile(npb[k][0], 2.5, axis=0)),
            "hi": conv(np.percentile(npb[k][0], 97.5, axis=0)),
            "seat": hconv(h), "seat_lo": hconv(np.percentile(npb[k][1], 2.5, axis=0)),
            "seat_hi": hconv(np.percentile(npb[k][1], 97.5, axis=0)),
            "agree_np": order_agreement(s, npb[k][0]), "agree_par": order_agreement(s, par[k]),
        }
    # Descriptives per model (non-mirror runs, from the model's own perspective).
    desc = {}
    for m in models:
        own = [(r.outcome(), r.margin, r.ni_x) for r in runs if r.x == m and not r.mirror]
        own += [(1 - r.outcome(), -r.margin, r.ni_y) for r in runs if r.y == m and not r.mirror]
        a = np.array(own) if own else np.zeros((0, 3))
        desc[m] = {"n": len(own), "win_rate": float(a[:, 0].mean()) if own else float("nan"),
                   "mean_margin": float(a[:, 1].mean()) if own else float("nan"),
                   "mean_ni": float(a[:, 2].mean()) if own else float("nan")}
    res["desc"] = desc
    # Mirror runs (X vs X): seat-A advantage, per arena.
    res["mirror"] = {}
    for a in arenas:
        mir = [r for r in by_arena[a] if r.mirror]
        if not mir:
            continue
        mm = np.array([r.margin for r in mir])
        wa = np.array([r.outcome() for r in mir])
        mlo, mhi = boot_ci(mm, rng, n_boot)
        wlo, whi = boot_ci(wa, rng, n_boot)
        res["mirror"][a] = {"n": len(mir), "models": sorted({r.x for r in mir}), "mean_margin": float(mm.mean()),
                            "margin_lo": mlo, "margin_hi": mhi, "seatA_win_rate": float(wa.mean()),
                            "win_lo": wlo, "win_hi": whi}
    return res


# ---------------------------------------------------------------- time profile (diagnostic)
def time_profile(runs: list[Run]) -> dict[str, dict[str, dict]]:
    """{arena: {model: {"cum": mean cumulative margin per day, "weekly": [...],
    "weekly_ni": [...], "early": .., "late": .., "n": ..}}} from each model's own
    seat (both seats of a mirror run count). Diagnostic only."""
    out: dict[str, dict[str, dict]] = {}
    for a in sorted({r.arena for r in runs}):
        rs = [r for r in runs if r.arena == a]
        days = min(len(r.daily_x) for r in rs)  # runs that stopped early: common prefix
        per: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for r in rs:
            dx, dy = np.array(r.daily_x[:days]), np.array(r.daily_y[:days])
            per.setdefault(r.x, []).append((dx - dy, dx))
            per.setdefault(r.y, []).append((dy - dx, dy))
        out[a] = {}
        for m, lst in sorted(per.items()):
            marg = np.mean([d for d, _ in lst], axis=0)
            ni = np.mean([v for _, v in lst], axis=0)
            weeks = range(0, days, WEEK)
            half = days // 2
            out[a][m] = {"n": len(lst), "days": days, "cum": np.cumsum(marg),
                         "weekly": [float(marg[i:i + WEEK].sum()) for i in weeks],
                         "weekly_ni": [float(ni[i:i + WEEK].sum()) for i in weeks],
                         "early": float(marg[:half].sum()), "late": float(marg[half:].sum())}
    return out


# ---------------------------------------------------------------- output
def ranking_rows(label: str, res: dict) -> list[dict]:
    f = res["fits"]
    rows = []
    for k, m in enumerate(res["models"]):
        d = res["desc"][m]
        row = {"arena": label, "model": m, "n_runs": d["n"], "win_rate": d["win_rate"],
               "mean_margin": d["mean_margin"], "mean_ni": d["mean_ni"]}
        for kind in KINDS:
            row[kind] = f[kind]["score"][k]
            row[f"{kind}_lo"] = f[kind]["lo"][k]
            row[f"{kind}_hi"] = f[kind]["hi"][k]
        rows.append(row)
    rows.sort(key=lambda r: -r["margin"])
    return rows


def pairing_rows(runs: list[Run]) -> list[dict]:
    """Per (arena, unordered pair, seed): model_1's margin in each seat order."""
    out = {}
    for r in runs:
        a, b = sorted((r.x, r.y))
        key = (r.arena, a, b, r.seed)
        row = out.setdefault(key, {"arena": r.arena, "model_1": a, "model_2": b, "seed": r.seed,
                                   "m_1_seatA": None, "m_1_seatB": None, "w_1_seatA": None, "w_1_seatB": None})
        if r.x == a:  # mirror runs (a vs a): seat-A margin only
            row["m_1_seatA"], row["w_1_seatA"] = r.margin, r.outcome()
        else:
            row["m_1_seatB"], row["w_1_seatB"] = -r.margin, 1 - r.outcome()
    rows = list(out.values())
    for row in rows:
        ms = [v for v in (row["m_1_seatA"], row["m_1_seatB"]) if v is not None]
        row["m_1_mean"] = float(np.mean(ms)) if ms else None
    return sorted(rows, key=lambda r: (r["arena"], r["model_1"], r["model_2"], r["seed"]))


def fmt(v, nd=0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:,.{nd}f}"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_outputs(results: dict[str, dict], runs: list[Run], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for label, res in results.items() for r in ranking_rows(label, res)]
    write_csv(out / "ratings.csv", rows)
    md = ["# Head-to-head ratings (run level)", "",
          "Margin score s (PRIMARY): least squares m = NI(X) - NI(Y) = s_X - s_Y + seat term, sum-to-zero; "
          "$ per run (overall: arena-SD units). Elo: Bradley-Terry MLE on run outcomes (higher full-period NI "
          "wins, ties ½), base 1200, slope 400, with a seat term. [95% non-parametric bootstrap CI over runs]. "
          "Win % and mean margin are raw (non-mirror runs, own perspective).", ""]
    seat_rows = []
    for label, res in results.items():
        f = res["fits"]
        nd_m = 2 if label == "overall" else 0
        md += [f"## {label} ({res['n_runs']} runs)", "",
               "| model | runs | win % | mean margin $ | margin score | Elo |", "|---|---|---|---|---|---|"]
        for r in ranking_rows(label, res):
            cell = lambda k, nd=0: f"{fmt(r[k], nd)} [{fmt(r[k + '_lo'], nd)}, {fmt(r[k + '_hi'], nd)}]"  # noqa: E731
            md.append(f"| {r['model']} | {r['n_runs']} | {fmt(100 * r['win_rate'])} | {fmt(r['mean_margin'])} "
                      f"| {cell('margin', nd_m)} | {cell('elo')} |")
        md += ["", "Pairwise order agreement (non-parametric / parametric bootstrap): "
               + "; ".join(f"{k} {fmt(f[k]['agree_np'], 1)}% / {fmt(f[k]['agree_par'], 1)}%" for k in KINDS)
               + f". Margin residual SD {fmt(res['margin_resid_sd'], nd_m)}.", ""]
        for gi, a in enumerate(res["arenas"]):
            for k in KINDS:
                unit = "Elo" if k == "elo" else ("SD" if label == "overall" else "$")
                seat_rows.append({"arena": label, "seat_group": a, "estimate": f"{k} seat term ({unit})",
                                  "value": f[k]["seat"][gi], "lo": f[k]["seat_lo"][gi], "hi": f[k]["seat_hi"][gi]})
        for a, mi in res["mirror"].items():
            seat_rows.append({"arena": label, "seat_group": f"mirror {a}",
                              "estimate": f"mirror mean margin A-B ($, n={mi['n']}, {'/'.join(mi['models'])})",
                              "value": mi["mean_margin"], "lo": mi["margin_lo"], "hi": mi["margin_hi"]})
            seat_rows.append({"arena": label, "seat_group": f"mirror {a}", "estimate": "mirror seat-A win rate",
                              "value": mi["seatA_win_rate"], "lo": mi["win_lo"], "hi": mi["win_hi"]})
    md += ["## Seat advantage (first seat, e.g. roaster_A, minus second seat)", "",
           "| fit | group | estimate | value [95% CI] |", "|---|---|---|---|"]
    for s in seat_rows:
        nd = 2 if ("rate" in s["estimate"] or "SD" in s["estimate"]) else 0
        md.append(f"| {s['arena']} | {s['seat_group']} | {s['estimate']} | {fmt(s['value'], nd)} "
                  f"[{fmt(s['lo'], nd)}, {fmt(s['hi'], nd)}] |")
    (out / "ratings.md").write_text("\n".join(md) + "\n")
    write_csv(out / "seat.csv", seat_rows, ["arena", "seat_group", "estimate", "value", "lo", "hi"])
    prs = pairing_rows(runs)
    write_csv(out / "pairings.csv", prs)
    pm = ["# Per-pairing run margins (model_1's NI minus model_2's, $)", "",
          "| arena | model_1 | model_2 | seed | m (model_1 in seat A) | m (model_1 in seat B) | mean | wins (A / B) |",
          "|---|---|---|---|---|---|---|---|"]
    for p in prs:
        pm.append(f"| {p['arena']} | {p['model_1']} | {p['model_2']} | {p['seed']} | {fmt(p['m_1_seatA'])} "
                  f"| {fmt(p['m_1_seatB'])} | {fmt(p['m_1_mean'])} | {fmt(p['w_1_seatA'], 1)} / {fmt(p['w_1_seatB'], 1)} |")
    (out / "pairings.md").write_text("\n".join(pm) + "\n")
    # Time profile (diagnostic).
    tp = time_profile(runs)
    trows, tmd = [], ["# Time profile (DIAGNOSTIC)", "", TIME_NOTE, "",
                      "Mean over each model's runs (own seat; mirror runs count both seats). "
                      "Early = first half of the run, late = second half ($ margin vs the opponent).", ""]
    for a, per in tp.items():
        tmd += [f"## {a}", "", "| model | seat-runs | early margin $ | late margin $ | weekly margins $ |",
                "|---|---|---|---|---|"]
        for m, v in per.items():
            tmd.append(f"| {m} | {v['n']} | {fmt(v['early'])} | {fmt(v['late'])} | "
                       + " ".join(fmt(w) for w in v["weekly"]) + " |")
            for wk, (wm, wn) in enumerate(zip(v["weekly"], v["weekly_ni"])):
                trows.append({"arena": a, "model": m, "week": wk + 1, "days": f"{wk * WEEK}-{min((wk + 1) * WEEK, v['days']) - 1}",
                              "seat_runs": v["n"], "mean_weekly_ni": wn, "mean_weekly_margin": wm,
                              "mean_cum_margin": float(v["cum"][min((wk + 1) * WEEK, v["days"]) - 1])})
        tmd.append("")
    write_csv(out / "time_profile.csv", trows)
    (out / "time_profile.md").write_text("\n".join(tmd) + "\n")


INK, CI_INK, GRID = "#0b0b0b", "#86b6ef", "#d9d8d4"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _style(ax) -> None:
    ax.tick_params(labelsize=8, colors="#52514e")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def plot_ratings(label: str, res: dict, path: Path) -> None:
    """Margin score (primary) and Elo, point + 95% CI, models sorted by margin score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = res["fits"]
    order = np.argsort(f["margin"]["score"])
    ypos = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.3 * len(order) + 1.0), sharey=True)
    titles = (("margin", "Margin score (SD units)" if label == "overall" else "Margin score ($ per run)"),
              ("elo", "Elo (run outcomes)"))
    for ax, (kind, title) in zip(axes, titles):
        sc, lo, hi = f[kind]["score"][order], f[kind]["lo"][order], f[kind]["hi"][order]
        ax.axvline(BASE if kind == "elo" else 0.0, color=GRID, lw=1, zorder=0)
        ax.hlines(ypos, lo, hi, color=CI_INK, lw=2)
        ax.plot(sc, ypos, "o", color=SERIES[0], ms=6, mec="white", mew=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([res["models"][i] for i in order])
        ax.set_title(title, fontsize=9, color=INK)
        _style(ax)
    fig.suptitle(f"{label}: {res['n_runs']} runs (95% bootstrap CI)", fontsize=9, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_time_profile(runs: list[Run], path: Path) -> None:
    """Mean cumulative margin vs day per model, one panel per arena (diagnostic)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tp = time_profile(runs)
    models = sorted({r.x for r in runs} | {r.y for r in runs})
    color = {m: SERIES[k % len(SERIES)] for k, m in enumerate(models)}
    dash = {m: "-" if k < len(SERIES) else "--" for k, m in enumerate(models)}
    fig, axes = plt.subplots(1, len(tp), figsize=(3.6 * len(tp) + 1.6, 3.0), squeeze=False)
    for ax, (a, per) in zip(axes[0], tp.items()):
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        for m, v in per.items():
            ax.plot(np.arange(v["days"]), v["cum"], dash[m], color=color[m], lw=2, label=m)
        ax.set_title(a, fontsize=9, color=INK)
        ax.set_xlabel("day", fontsize=8, color="#52514e")
        ax.set_ylabel("mean cumulative margin vs opponent ($)", fontsize=8, color="#52514e")
        _style(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, frameon=False)
    fig.suptitle("Time profile (diagnostic only, not used for ranking)", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1 - 1.4 / (3.6 * len(tp) + 1.6), 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def figure_paths(results: dict[str, dict], fig: Path) -> dict[str, Path]:
    """--fig = the single arena, or 'overall' plus <stem>_<arena>.pdf per arena."""
    if len(results) == 1:
        return {next(iter(results)): fig}
    out = {"overall": fig}
    out.update({a: fig.with_name(f"{fig.stem}_{a}{fig.suffix}") for a in results if a != "overall"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "h2h"))
    ap.add_argument("--fig", default=str(ROOT / "paper" / "images_cr" / "h2h_ratings.pdf"))
    ap.add_argument("--fig-time", default=None, help="Default: <fig dir>/h2h_time_profile.pdf.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--prior", type=float, default=0.01, help="Ridge on logit strengths (0 = pure MLE).")
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()
    dirs = args.dirs or sorted(glob.glob(str(ROOT / "results" / "hb_h2h*")))
    runs = load_runs(dirs)
    if not runs:
        raise SystemExit(f"no H2H runs under {dirs}")
    arenas = sorted({r.arena for r in runs})
    print(f"[h2h] {len(runs)} runs, arenas {arenas}")
    results = {a: analyze([r for r in runs if r.arena == a], args.n_boot, args.rng, args.prior) for a in arenas}
    if len(arenas) > 1:
        results["overall"] = analyze(runs, args.n_boot, args.rng, args.prior, standardize=True)
    write_outputs(results, runs, Path(args.out))
    figs = figure_paths(results, Path(args.fig))
    for label, p in figs.items():
        plot_ratings(label, results[label], p)
    fig_time = Path(args.fig_time) if args.fig_time else Path(args.fig).with_name("h2h_time_profile.pdf")
    plot_time_profile(runs, fig_time)
    print((Path(args.out) / "ratings.md").read_text())
    print(f"[h2h] wrote {args.out}/{{ratings,pairings,time_profile}}.{{md,csv}}, seat.csv; "
          f"figures {', '.join(map(str, figs.values()))}, {fig_time}")


if __name__ == "__main__":
    main()
