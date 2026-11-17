#!/bin/sh
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -N cb_sim
# Project, queue and site-specific -v variables are passed on the qsub command line (see below).
#PBS -k oed
# CoffeeBench on Harbor: one node, one or more simulations (one per seed).
# Submit from a fresh checkout ("deploy dir") that has .env (OPENROUTER_API_KEY) and the harness installed:
#   qsub -P <project> -q <queue> -N cb_<focal>_<roster>_<role>_s<seed> \
#     -v CB_CONFIG=experiments/hb_<role>_..._v21.toml,CB_SEEDS=0,CB_FOCAL=<short>,CB_BG=Sonnet55|Qwen38F|Rule \
#     harbor_adapter/cluster/pbs_job.sh
# CB_FOCAL=oracle → $0 plumbing run (Harbor oracle agent, tasks' solve.sh).
# CB_FOCAL=baseline + CB_LOCAL=<role>_A:passive|heuristic_roaster → baseline focal, LLM background.
# CB_BG=Rule → rule-based background (all 5 non-focal agents scripted in-process on the server,
#   only the focal Harbor job runs); with CB_FOCAL=baseline the whole sim is in-process ($0).
# CB_BG=Persona:<model> → every non-focal seat is an LLM (<model>) with its private persona block
#   (profile from the config's [models] seat entry); [background] rule_seats stay rule-based.
# CB_SEEDS="0 1 2" runs the seeds concurrently on the node (DESIGN §9: start at ≤ 6/node).
# Restarts (DESIGN §6) only when needed: launch_sim.py asks the server to stop at the next
# simulated morning when the remaining walltime runs low, then self-requeues this script with
# CB_RESUME=<checkpoint> (one job per seed; see harbor_adapter/jobs/<sim>.resume.json); crashed
# segments restart from the newest daily checkpoint. CB_SEGMENT_DAYS=N forces planned segments.
# Head-to-head (Arena, hb_roaster_d30.toml): CB_FOCAL=X CB_FOCAL2=Y → X in roaster_A, Y in the opponent
#   seat (CB_OPPONENT_SEAT, default roaster_B), CB_BG = the 4-seat background; three Harbor jobs per sim.
#   CB_PAIRS=X:Y+X2:Y2+... (or space-separated) runs several pairings concurrently on this node (× CB_SEEDS; CB_FOCAL unused).
#   Pairing lists: harbor_adapter/h2h_schedule.py.
# CB_RESUME=latest: every (focal, seed) of this job resumes from its own latest checkpoint (finished runs are skipped).
# Manual resume: add CB_RESUME=trajectories/harbor/<roster>/<focal>/<cfg>/seed_N/checkpoints/day_NNN.pkl
# (single seed). Optional: CB_LOCAL, CB_MAX_DAYS, CB_SEGMENT_DAYS, CB_ACTION_FORMAT (json|mini_swe).
CB_JOB_START=$(date +%s)
export CB_JOB_START
export CB_WALLTIME_SEC="${CB_WALLTIME_SEC:-43200}"  # = the walltime header (12:00:00)
set -eu
cd "${PBS_O_WORKDIR:?submit with qsub from the deploy dir}"
export PATH=$HOME/bin:$HOME/.local/bin:$PATH
# Harness pin check BEFORE any key is in the environment, by file inspection only
# (starting a compromised interpreter would already run its .pth payload):
# harbor/litellm must match harbor_adapter/requirements-harness.txt and no
# litellm-1.107.0 payload files (cook.pth / cook.py / sitecustomize.py) may exist.
HB_SITE=$(ls -d "$(dirname "$(readlink -f "$(command -v "${HARBOR_CMD:-harbor}")")")"/../lib/python*/site-packages | head -1)
for pin in $(grep -E '^(harbor|litellm)==' harbor_adapter/requirements-harness.txt); do
  [ -d "$HB_SITE/${pin%%==*}-${pin#*==}.dist-info" ] || { echo "harness pin mismatch: $pin not installed in $HB_SITE" >&2; exit 1; }
done
for f in cook.pth cook.py sitecustomize.py; do
  [ ! -e "$HB_SITE/$f" ] || { echo "suspicious $HB_SITE/$f (litellm 1.107.0 payload?)" >&2; exit 1; }
done
# Keys must not reach the job log: NEVER enable `set -x` before .env is loaded.
set -a; . ./.env; set +a
set -x

: "${CB_CONFIG:?set CB_CONFIG}"
[ -n "${CB_PAIRS:-}" ] || : "${CB_FOCAL:?set CB_FOCAL (model short name or oracle) or CB_PAIRS}"
# Pairings "X:Y" (head-to-head) or plain focal models.
if [ -n "${CB_PAIRS:-}" ]; then
  RUNS=$(echo "$CB_PAIRS" | tr '+' ' ')  # "X:Y+X2:Y2" (no spaces in qsub -v) or "X:Y X2:Y2"
elif [ -n "${CB_FOCAL2:-}" ]; then
  RUNS=""; for f in $CB_FOCAL; do RUNS="$RUNS $f:$CB_FOCAL2"; done
else
  RUNS="$CB_FOCAL"
fi
CB_SEEDS="${CB_SEEDS:-0}"
# CB_QSUB_EXTRA (e.g. "-P@<project>@-q@<queue>"; @ = space inside -v) is forwarded to requeues.
export CB_QSUB_EXTRA="${CB_QSUB_EXTRA//@/ }"
CB_SEGMENT_DAYS="${CB_SEGMENT_DAYS:-0}"  # 0 = restart only for walltime / crashes
PY="${CB_PY:-uv run python}"
export HARBOR_CMD="${HARBOR_CMD:-harbor}"

. harbor_adapter/cluster/rootless_podman_env.sh
podman pull docker.io/library/python:3.11-slim

# Dataset (6 tasks, network_mode none + Unix-socket bind mount) + prewarm the shared image once so the
# per-trial builds hit the classic-builder cache (vfs build storms otherwise).
GEN_EXTRA="--bg ${CB_BG:-Haiku55}"  # Persona:<model> renders per-seat persona blocks
[ -n "${CB_MAX_DAYS:-}" ] && GEN_EXTRA="$GEN_EXTRA --max-days $CB_MAX_DAYS"
case "$RUNS" in *:*) GEN_EXTRA="$GEN_EXTRA --h2h" ;; esac  # opponent seat: no persona block
[ -n "${CB_OPPONENT_SEAT:-}" ] && GEN_EXTRA="$GEN_EXTRA --opponent-seat $CB_OPPONENT_SEAT"
DS="harbor_adapter/datasets/$($PY harbor_adapter/generate_tasks.py --config "$CB_CONFIG" --network none $GEN_EXTRA --print-name)"
# Jobs launched together share the dataset: generate it once, under a lock,
# and never regenerate it while other simulations may be using it (deploy a
# fresh directory to change instructions).
mkdir -p harbor_adapter/datasets
(
  flock 9
  [ -f "$DS/.complete" ] || {
    $PY harbor_adapter/generate_tasks.py --config "$CB_CONFIG" --network none $GEN_EXTRA
    touch "$DS/.complete"
  }
) 9> harbor_adapter/datasets/.generate.lock
docker build -t cb_prewarm "$(ls -d "$DS"/*/environment | head -1)"

mkdir -p harbor_adapter/jobs
focal_arg() {
  case "$1" in
    oracle) echo "--oracle" ;;
    scripted) echo "--scripted" ;;  # $0 test (fake LLM)
    baseline) echo "--bg ${CB_BG:-Haiku55}" ;;  # needs CB_LOCAL=<role>_A:<baseline>
    *) echo "--focal $1 --bg ${CB_BG:-Haiku55} --effort ${CB_EFFORT:-max}" ;;
  esac
}
EXTRA=""
[ -n "${CB_RESUME:-}" ] && EXTRA="$EXTRA --resume $CB_RESUME"
[ -n "${CB_LOCAL:-}" ] && EXTRA="$EXTRA --local $CB_LOCAL"
[ -n "${CB_MAX_DAYS:-}" ] && EXTRA="$EXTRA --max-days $CB_MAX_DAYS"
[ -n "${CB_ACTION_FORMAT:-}" ] && EXTRA="$EXTRA --action-format $CB_ACTION_FORMAT"  # json | mini_swe
[ -n "${CB_OPPONENT_SEAT:-}" ] && EXTRA="$EXTRA --opponent-seat $CB_OPPONENT_SEAT"
[ -n "${PBS_JOBID:-}" ] && EXTRA="$EXTRA --requeue"
# CB_FOCAL may list several models ("Astra K3 GLM53"): one simulation per
# (focal, seed), all concurrently on this node (each has its own server,
# socket and containers). A requeue continues a single simulation.
# Wait only for the launchers: a bare `wait` would also wait for the background
# podman service (cluster/rootless_podman_env.sh) and hold the node until walltime.
LAUNCH_PIDS=""
for run in $RUNS; do
  focal="${run%%:*}"; H2H=""; label="$focal"
  case "$run" in *:*) H2H="--focal2 ${run#*:}"; label="${focal}_vs_${run#*:}" ;; esac
  for seed in $CB_SEEDS; do
    $PY harbor_adapter/launch_sim.py --config "$CB_CONFIG" --seed "$seed" $(focal_arg "$focal") $H2H \
      --segment-days "$CB_SEGMENT_DAYS" $EXTRA > "harbor_adapter/jobs/launch_$(basename "$CB_CONFIG" .toml)_${label}_${CB_BG:-Haiku55}_s${seed}_${PBS_JOBID:-manual}.log" 2>&1 &
    LAUNCH_PIDS="$LAUNCH_PIDS $!"
    sleep 20  # stagger starts (image/compose setup)
  done
done
for pid in $LAUNCH_PIDS; do wait "$pid" || true; done

kill $PODMAN_SERVICE_PID 2>/dev/null || true
echo DONE
