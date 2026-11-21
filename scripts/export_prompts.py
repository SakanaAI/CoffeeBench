"""Export the exact prompt an evaluated agent received (first user message of a Harbor trajectory) for the paper appendix.

    uv run python scripts/export_prompts.py [trajectory.json]

Default: Claude Sonnet 5.5 as roaster_A, main 90-day config, seed 0 (results/hb_mf_traj/CoffeeBench-mf1). Writes
paper/prompts_cr/harness_template.txt (mini-swe-agent action format), rules_sections.txt (rules §1-10) and
tools_section.txt (§11). Non-ASCII symbols are transliterated because LaTeX listings cannot typeset them.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "prompts_cr"
ASCII = {"−": "-", "–": "-", "—": "--", "≤": "<=", "≥": ">=", "×": "x", "→": "->", "÷": "/", "…": "...", "’": "'",
         "‘": "'", "“": '"', "”": '"', "§": "Sec. ", "·": "*", "≈": "~", "€": "EUR", " ": " "}


def main() -> None:
    if len(sys.argv) > 1:
        traj = Path(sys.argv[1])
    else:
        traj = Path(sorted(glob.glob(str(ROOT / "results/hb_mf_traj/CoffeeBench-mf1/harbor_adapter/jobs/"
                                         "cb_Sonnet55_PersonaSonnet55-mswe_roaster_none_E_llmrival_v21_s0_*/"
                                         "seg01/focal/*/agent/trajectory.json")))[0])
    msg = json.loads(traj.read_text())["steps"][0]["message"]
    msg = msg.split("\nCurrent terminal state:")[0].rstrip()
    for k, v in ASCII.items():
        msg = msg.replace(k, v)
    bad = sorted({c for c in msg if ord(c) > 126})
    if bad:
        raise SystemExit(f"unmapped non-ASCII characters: {bad}")
    head, rules = msg.split("Simulation rules and interface:\n", 1)
    rules, tools = rules.split("## 11. Tools", 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "harness_template.txt").write_text(head.rstrip() + "\n\nSimulation rules and interface:\n<rules below>\n")
    (OUT / "rules_sections.txt").write_text(rules.rstrip() + "\n")
    (OUT / "tools_section.txt").write_text("## 11. Tools" + tools.rstrip() + "\n")
    print("from", traj.relative_to(ROOT))
    for f in sorted(OUT.glob("*.txt")):
        print(f.name, len(f.read_text().splitlines()), "lines")


if __name__ == "__main__":
    main()
