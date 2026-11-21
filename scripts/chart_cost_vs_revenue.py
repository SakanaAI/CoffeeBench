"""Scatter of focal-agent API cost vs revenue, one point per model:effort.

Reads completed runs under trajectories/roaster_focal_*/seed_* and renders a
light-mode PNG following the dataviz palette (categorical slots 1-4).
"""

import glob
import json
import os
import statistics as st
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- dataviz palette (light) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# family -> (categorical hue, marker) ; slots 1..4 = blue/green/magenta/yellow
FAMILY = {
    "claude": ("#2a78d6", "o", "Claude Opus 4.8"),
    "z-ai": ("#008300", "s", "GLM-5.2"),
    "openai": ("#e87ba4", "^", "GPT-5.5"),
    "gemini": ("#eda100", "D", "Gemini 3 Pro"),
}


def family_of(model: str) -> str:
    if model.startswith("claude"):
        return "claude"
    if model.startswith("gemini"):
        return "gemini"
    return model.split("/")[0]  # z-ai / openai / moonshotai ...


def focal(exp: str) -> str:
    cfg = f"experiments/{exp}.toml"
    if os.path.exists(cfg):
        with open(cfg, "rb") as f:
            return tomllib.load(f).get("models", {}).get("roaster_A", "?")
    return "?"


def per_seed(rj: str, ev: str):
    d = json.load(open(rj))
    ra = d["result"]["agents"]["roaster_A"]
    cost = 0.0
    calls = 0
    for ln in open(ev):
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "agent_step" and e.get("agent_id") == "roaster_A":
            cost = max(cost, e.get("cost_so_far") or 0.0)
            calls = max(calls, e.get("n_calls_so_far") or 0)
    return dict(ni=ra["net_income"], cost=cost, calls=calls)


def collect():
    out = []
    for ed in sorted(glob.glob("trajectories/roaster_focal_*")):
        exp = os.path.basename(ed)
        ss = []
        for sd in sorted(glob.glob(f"{ed}/seed_*")):
            rj, ev = f"{sd}/run.json", f"{sd}/run.events.jsonl"
            if os.path.exists(rj) and os.path.exists(ev):
                try:
                    ss.append(per_seed(rj, ev))
                except Exception:
                    pass
        if not ss:
            continue
        if st.mean(x["calls"] for x in ss) == 0:  # exclude failed (0-call) runs
            continue
        model = focal(exp)
        eff = model.rsplit(":", 1)[1] if ":" in model else "off"

        def ms(key):
            xs = [x[key] for x in ss]
            return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)

        out.append(dict(model=model, eff=eff, fam=family_of(model),
                        cost=ms("cost"), ni=ms("ni")))
    return out


def main():
    data = collect()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    seen = {}
    for d in data:
        hue, marker, label = FAMILY[d["fam"]]
        cx, cxe = d["cost"]
        ry, rye = d["ni"]
        ax.errorbar(cx, ry, xerr=cxe, yerr=rye, fmt="none",
                    ecolor=hue, elinewidth=1.2, capsize=3, alpha=0.55, zorder=2)
        ax.scatter(cx, ry, s=130, c=hue, marker=marker,
                   edgecolors=SURFACE, linewidths=1.8, zorder=3,
                   label=label if d["fam"] not in seen else None)
        seen[d["fam"]] = True
        # small offset to the right; dodge the near-coincident GLM points apart
        dodge = {"z-ai/glm-5.2:max": (9, 7), "z-ai/glm-5.2:high": (9, -9)}
        dx, dy = dodge.get(d["model"], (9, 0))
        ax.annotate(d["eff"], (cx, ry), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, color=INK2,
                    va="center", ha="left", zorder=4)

    ax.set_xlabel("API cost — focal agent ($)", color=INK2, fontsize=11)
    ax.set_ylabel("Net income — focal agent ($)", color=INK2, fontsize=11)
    ax.set_title("CoffeeBench: API cost vs net income by model & reasoning effort",
                 color=INK, fontsize=13, pad=12, loc="left")
    ax.text(0, 1.015, "focal = roaster_A · 90 days · 3-seed mean (±std)",
            transform=ax.transAxes, color=MUTED, fontsize=9)

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:.0f}")

    leg = ax.legend(loc="lower right", frameon=False, fontsize=9,
                    labelcolor=INK2, title="Model family")
    leg.get_title().set_color(INK2)

    fig.tight_layout()
    os.makedirs("assets", exist_ok=True)
    out = "assets/cost_vs_net_income.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
