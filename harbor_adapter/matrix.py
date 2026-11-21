"""Print the camera-ready Harbor matrix as launch_sim.py command lines.

    uv run python harbor_adapter/matrix.py [--seeds 0,1,2] [--roles roaster,farmer]
        [--rosters Haiku55,Qwen38F] [--focal Astra,...] [--no-baselines]

One line per simulation (focal × roster × role × seed). Baselines run the
focal agent in-process (passive for both roles; heuristic_roaster — the original
20-line policy, kept for continuity — and the strengthened heuristic_roaster_v2
for roaster) with the LLM background roster (sim tags passive / heuristic /
heuristicv2). Paid runs need explicit user approval.
"""

from __future__ import annotations

import argparse

# Focal panel (short names from launch_sim.MODELS).
FOCAL = ["Astra", "Fable5", "Opus55", "Sonnet55", "K3", "Gemini38F", "Muse13", "MuseW", "DSv41F", "Qwen38F", "MiMo26P", "GLM53"]
ROSTERS = ["Sonnet55", "Qwen38F"]
BASELINES = {"roaster": ["passive", "heuristic_roaster", "heuristic_roaster_v2"], "farmer": ["passive"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--roles", default="roaster,farmer")
    ap.add_argument("--rosters", default=",".join(ROSTERS))
    ap.add_argument("--focal", default=",".join(FOCAL))
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    for role in args.roles.split(","):
        cfg = f"experiments/hb_{role}.toml"
        for roster in args.rosters.split(","):
            for seed in args.seeds.split(","):
                base = f"--config {cfg} --seed {seed} --bg {roster}"
                for focal in args.focal.split(","):
                    print(f"python harbor_adapter/launch_sim.py {base} --focal {focal}")
                if not args.no_baselines:
                    for b in BASELINES[role]:
                        print(f"python harbor_adapter/launch_sim.py {base} --local {role}_A:{b}")


if __name__ == "__main__":
    main()
