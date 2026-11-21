"""What the evaluated agent does in the shell (main 90-day run, seed 0, mini-SWE-agent, background Claude Sonnet 5.5,
highest effort): the composition of Roaster A's shell commands, classified by rule.

    uv run python scripts/mf_action_source.py

Unit: one `coffee` call (typed, or issued from inside the agent's script) or one other shell command. A heredoc, a
quoted multi-line argument and indented continuation lines stay attached to their opening line, so a script is one
command, and the `coffee` calls it issues are added from run.events.jsonl (each simulation action carries the LLM call
count, which identifies the turn that issued it). Categories, by purpose:
  view state (typed)      a plain `coffee` command that only reads (view_*, read_message, wait_for_next_day),
                          possibly piped into a formatter
  view state (via script) such a call issued from inside a script, loop or pipeline
  act (typed)             a plain `coffee` command that changes the simulation (post_listing, make_offer,
                          accept_offer, roast, pay_invoice, send_message, ...)
  act (via script)        such a call issued from inside a script
  write memo              writes to a notes / log / plan file (cat, echo, tee, printf with a redirect)
  write script            writes a .py or .sh file, from the shell (cat/tee redirect) or from Python (write_text)
  other                   anything else (ls, head, mkdir, python that issued nothing). The command that runs a script
                          is not counted itself: it is represented by the `coffee` calls it issued.
No LLM is involved. Writes paper/images_cr/mf_action_source.{pdf,png} and paper/tables_cr/action_source.tex.
"""

from __future__ import annotations

import glob
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = "hb_roaster_none_E_llmrival_v21"
SEED = 0
TRAJ = ROOT / "results" / "hb_mf_traj"
MODELS = [("Sonnet55", "Claude Sonnet 5.5"), ("Astra", "GPT-6 Astra"), ("Opus55", "Claude Opus 5.5"),
          ("Haiku55", "Claude Haiku 5.5"), ("K31P", "Kimi K3.1 Pro"), ("GLM53", "GLM-5.3"),
          ("DSv41F", "DeepSeek V4.1 Flash"), ("Gemini38F", "Gemini 3.8 Flash")]
SHORT = {"Claude Sonnet 5.5": "Sonnet 5.5", "GPT-6 Astra": "GPT-6 Astra", "Claude Opus 5.5": "Opus 5.5",
         "Claude Haiku 5.5": "Haiku 5.5", "Kimi K3.1 Pro": "Kimi K3.1 Pro", "GLM-5.3": "GLM-5.3",
         "DeepSeek V4.1 Flash": "DeepSeek 4.1F", "Gemini 3.8 Flash": "Gemini 3.8F"}  # y-axis labels of the half-width figure
LABELS = [("view", "view state", "#9ecae1", ""), ("view_s", "view state, via script", "#9ecae1", "///"),
          ("act", "act", "#1f77b4", ""), ("act_s", "act, via script", "#1f77b4", "///"),
          ("script", "write script", "#9467bd", ""), ("memo", "write memo", "#d6a000", ""),
          ("other", "other", "#c9c9c9", "")]
VIEW_NAME = re.compile(r"^(view_\w+|read_message|wait_for_next_day)$")
FOLD: dict[str, str] = {}
VIEW = re.compile(r"\bcoffee\s+(view_\w+|read_message|wait_for_next_day)\b")
COFFEE = re.compile(r"\bcoffee\s+\w+")
PY = re.compile(r"\bpython3?\b|\bjq\b|\bawk\b|\bbc\b")
MEMO = re.compile(r"\b(cat|echo|tee|printf)\b.*(>|>>).*(\.md|\.txt|notes|memo|log|ledger|plan)", re.S)
SCRIPT = re.compile(r"\b(cat|tee)\b[^\n]*(>|>>)[^\n]*\.(py|sh)\b", re.S)
PY_WRITES_FILE = re.compile(r"write_text\(|open\([^)]*['\"][wa]['\"]")
SCRIPT_PATH = re.compile(r"\.(py|sh)\b")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def split_commands(keystrokes: str) -> list[str]:
    """Top-level shell commands; heredocs, unbalanced-quote blocks and indented continuations stay attached."""
    out, buf, end, quote = [], [], None, None
    for line in keystrokes.replace("\r", "").split("\n"):
        if end is not None:
            buf.append(line)
            if line.strip() == end:
                out.append("\n".join(buf))
                buf, end = [], None
            continue
        if quote is not None:
            buf.append(line)
            if line.count(quote) % 2 == 1:
                out.append("\n".join(buf))
                buf, quote = [], None
            continue
        if not line.strip():
            continue
        m = re.search(r"<<-?\s*['\"]?(\w+)['\"]?", line)
        if m:
            if buf:
                out.append("\n".join(buf))
            buf, end = [line], m.group(1)
            continue
        for ch in ('"', "'"):
            if line.count(ch) % 2 == 1:
                if buf:
                    out.append("\n".join(buf))
                buf, quote = [line], ch
                break
        if quote:
            continue
        if buf and line.startswith((" ", "\t")):
            buf.append(line)
            continue
        if buf:
            out.append("\n".join(buf))
        buf = [line]
    if buf:
        out.append("\n".join(buf))
    return [c for c in out if c.strip() and c.strip() not in ("{", "}")]


def actions_per_turn(run_dir: Path) -> dict[int, list[str]]:
    """n_calls_so_far -> names of the simulation actions roaster_A issued in that LLM turn."""
    c: dict[int, list[str]] = {}
    for line in open(run_dir / "run.events.jsonl"):
        if '"agent_step"' not in line or '"roaster_A"' not in line:
            continue
        e = json.loads(line)
        if e.get("type") == "agent_step" and e["agent_id"] == "roaster_A":
            c.setdefault(e["n_calls_so_far"], []).append(e["action"])
    return c


def classify(cmd: str) -> str:
    t = cmd.strip()
    if MEMO.search(t):
        return "memo"
    if SCRIPT.search(t) or (PY_WRITES_FILE.search(t) and SCRIPT_PATH.search(t)):
        return "script"
    if COFFEE.search(t):
        return "view" if VIEW.search(t) else "act"
    return "other"


def count(traj: Path, per_turn: dict[int, list[str]]) -> Counter:
    """Trajectory agent step k is LLM call k (both count from 1)."""
    out: Counter = Counter()
    steps = [s for s in json.load(open(traj))["steps"] if s.get("source") == "agent"]
    for k, step in enumerate(steps, start=1):
        ks = "".join((c.get("arguments") or {}).get("keystrokes", "") for c in step.get("tool_calls") or [])
        typed = 0
        issued = per_turn.get(k, [])
        for cmd in split_commands(ks):
            lab = classify(cmd)
            if lab == "other" and len(issued) > typed and PY.search(cmd):
                continue  # the command that runs a script: represented by the coffee calls it issued
            out[lab] += 1
            typed += lab in ("view", "act")
        # coffee calls this turn beyond the typed ones were issued by a script: classify each by its subcommand
        for name in per_turn.get(k, [])[typed:]:
            out["view_s" if VIEW_NAME.match(name) else "act_s"] += 1
    return out


def main() -> None:
    rows = []
    for m, label in MODELS:
        run_dir = next(Path(d) for d in glob.glob(str(ROOT / f"results/hb_mf[12]/PersonaSonnet55-mswe/{m}/{CFG}/seed_{SEED}")))
        traj = glob.glob(str(TRAJ / f"CoffeeBench-mf*/harbor_adapter/jobs/cb_{m}_PersonaSonnet55-mswe_roaster_none_E_llmrival_v21_s{SEED}_*/seg*/focal/*/agent/trajectory.json"))
        assert len(traj) == 1, (m, traj)
        cnt = count(Path(traj[0]), actions_per_turn(run_dir))
        tot = sum(cnt.values())
        rows.append((m, label, cnt, tot))
        print(f"{label:20s} cmds {tot:5d}  " + "  ".join(f"{k} {100 * cnt[k] / tot:4.0f}%" for k, *_ in LABELS))

    plt.rcParams.update({"font.size": 6.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(2.75, 2.7))
    y = list(range(len(rows)))[::-1]
    left = [0.0] * len(rows)
    plt.rcParams["hatch.linewidth"] = 0.5
    plt.rcParams["hatch.color"] = "#333333"
    for key, name, col, hatch in LABELS:
        v = [100 * r[2][key] / r[3] for r in rows]
        ax.barh(y, v, left=left, color=col, height=0.62, edgecolor="white", linewidth=0.8, hatch=hatch, label=name)
        left = [p + q for p, q in zip(left, v)]
    ax.set_yticks(y, [SHORT[r[1]] for r in rows], fontsize=6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of commands (%)", fontsize=6.5, color=INK)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.legend(loc="lower center", ncol=3, fontsize=5.0, frameon=False, handlelength=1.0, handletextpad=0.4,
               columnspacing=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = ROOT / "paper" / "images_cr" / "mf_action_source"
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=300)
    lines = [f"{label.replace(' ', '~')} & {tot:,} & " + " & ".join(f"{cnt[k]:,}" for k, *_ in LABELS) + " \\\\"
             for m, label, cnt, tot in rows]
    (ROOT / "paper" / "tables_cr" / "action_source.tex").write_text("\n".join(lines).replace(",", "{,}") + "\n")
    print("wrote", f"{out}.pdf")


if __name__ == "__main__":
    main()
