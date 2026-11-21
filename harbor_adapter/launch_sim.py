"""Run ONE CoffeeBench simulation on Harbor: env server + 2 Harbor jobs.

    # $0 plumbing test (Harbor oracle agent = tasks' solution/solve.sh):
    uv run python harbor_adapter/launch_sim.py --config X.toml --seed 0 --oracle
    # LLM run (PAID — get approval first):
    uv run python harbor_adapter/launch_sim.py --config X.toml --seed 0 --focal Fable5

Steps: generate the dataset if missing → start `python -m coffeebench.server`
(all non-local agents remote, per-run admin token) → render job YAMLs from
harbor_adapter/templates → launch `echo Y | harbor run -c <yaml> -y` for the
focal job (focal model × focal task) and the background job (bg model × the
other tasks) with the per-agent tokens in the Harbor host environment:
COFFEE_TOKEN_<AID> (= HMAC(run token, aid); each task.toml maps only its own
one to the container's COFFEE_TOKEN), COFFEE_HARNESS_TOKEN_<AID> (CoffeeTerminus,
host only) and COFFEE_VERIFIER_TOKEN_<AID> (tests/test.sh only) → wait for the
server and both jobs → print per-agent NI + Harbor costs.

Networking (`--network`, see generate_tasks.py): `none` (default: Linux cluster with rootless podman) — task
containers have no network; the server also listens on a Unix socket in a
fresh per-segment dir $CB_SOCK_ROOT/cb_XXXX/ (default /tmp: AF_UNIX paths are
limited to ~104 bytes, run dirs are too long), bind-mounted at /run/coffee via
COFFEE_SOCK_DIR. `relay` — Docker Desktop test of `none` (relay container:
socket in a docker volume → host TCP). `bridge` — TCP via host.docker.internal
(local smoke fallback; internet reachable). The launcher and CoffeeTerminus
always use host TCP 127.0.0.1 (COFFEE_HOST_URL).

`--local farmer_B:passive,...` keeps those agents in-process on the server
(hybrid; DESIGN step 5: 1 focal LLM + scripted rest) and drops their tasks.
`--bg Rule` does that for all five non-focal agents with the rule-based market
(the config's [models] seat entries when they name a scripted policy, e.g.
`heuristic_farmer@adversarial`, else heuristic_farmer / heuristic_roaster_v2 /
heuristic_retailer); `--local` overrides single seats. Mixed rosters (some seats
rule, some LLM): `--bg Sonnet55 --local farmer_B:heuristic_farmer@adversarial,...`
(the non-local seats run as the LLM bg job). With --bg Rule only the focal
Harbor job runs; outputs under trajectories/harbor/Rule/<focal>/. With a
baseline focal (`--local roaster_A:heuristic_roaster_v2`) every agent is
scripted and the run is in-process (`coffeebench.main`, same output path).

Head-to-head ("Arena", hb_roaster_d30.toml): `--focal X --focal2 Y --bg Sonnet55`
runs X in the focal seat (roaster_A) and Y in the opponent seat
(`--opponent-seat`, default: the focal role's other seat, roaster_B), both at
their FOCAL_EFFORT (--effort), with THREE Harbor jobs: focal (X × roaster_A),
focal2 (Y × roaster_B) and bg (the bg model × the other 4 seats, API default).
Score = NI(X) - NI(Y). Outputs under trajectories/harbor/H2H_<bg>/<X>_vs_<Y>/,
sim name cb_<X>_vs_<Y>_<bg>_<cfg>_s<seed> (the order encodes the seats:
Y_vs_X is the seat-swapped run); <run_dir>/h2h.json records the seats. The
opponent seat is never `--local`, a Rule seat or a persona seat; with --bg Rule
/ Persona the remaining seats are rule / persona as usual. A baseline focal
(`--local roaster_A:heuristic_roaster_v2 --focal2 Y`) is allowed (tag
heuristicv2_vs_Y). $0 tests: `--scripted --focal2 scripted` (or --oracle).

Restarts (DESIGN §6) happen only when needed (user decision 2026-10-17; no
planned segments by default, `--segment-days N` still forces boundaries):
- walltime: while a segment runs, this script watches the remaining walltime
  (CB_JOB_START + CB_WALLTIME_SEC from cluster/pbs_job.sh, minus --safety-sec). Once
  it drops below max(seconds per simulated day × --stop-margin-days,
  --stop-margin-min-sec), it POSTs /admin/stop_at_next_morning (admin token);
  the server checkpoints at the next MORNING_OPEN, sends segment_end, takes
  the workspace uploads and exits. If less than --min-segment-sec remains,
  this script writes <jobs_dir>/<sim>.resume.json + requeue.sh and (with
  --requeue) submits the same cluster/pbs_job.sh with CB_RESUME=<ckpt>; otherwise
  it continues in this job.
- crash: if the server dies or a Harbor job exits while its agents are still
  in the running sim, the segment is restarted (≤ --max-retries times per
  checkpoint) from the newest daily checkpoint (latest.pkl, frozen as
  day_NNN.pkl).
Every restart is recorded as {day, reason: walltime|crash|planned, segment}
in <run_dir>/restarts.json (and resume.json). Each segment = server
`--resume <ckpt>` + fresh Harbor jobs under <jobs_dir>/<run>/seg<k>/.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pickle
import secrets
import shutil
import socket
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coffeebench.config import RunConfig  # noqa: E402
from coffeebench.server import agent_state  # noqa: E402
from coffeebench.server.app import HARNESS, VERIFIER, agent_token, scoped_token  # noqa: E402

AGENTS = ["farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B"]

# Background model. Slug confirmed in OpenRouter /api/v1/models (2026-10-17).
# Routed via OpenRouter because litellm has no native claude-haiku-5-5 price entry.
HAIKU_55_SLUG = "anthropic/claude-haiku-5.5"

# short name → (OpenRouter slug, pinned provider).
MODELS: dict[str, tuple[str, str]] = {
    "Haiku55": (HAIKU_55_SLUG, "anthropic"),
    "Sonnet55": ("anthropic/claude-sonnet-5.5", "anthropic"),
    "Opus55": ("anthropic/claude-opus-5.5", "anthropic"),
    "Astra": ("openai/gpt-6-astra", "openai"),
    "Fable5": ("anthropic/claude-fable-5", "anthropic"),
    "Fable51": ("anthropic/claude-fable-5.1", "anthropic"),
    "K3": ("moonshotai/kimi-k3", "moonshotai"),
    # Kimi K3.1 Pro (added 2026-10-24; $2/$10 per M, official Moonshot AI endpoint only).
    "K31P": ("moonshotai/kimi-k3.1-pro", "moonshotai"),
    "Gemini38F": ("google/gemini-3.8-flash", "google-ai-studio"),
    "Muse13": ("meta/muse-spark-1.3", "meta"),
    "MuseW": ("meta/muse-wonder", "meta"),
    "DSv41F": ("deepseek/deepseek-v4.1-flash", "deepinfra"),
    # Both first-party Alibaba endpoints: 100+ concurrent Qwen background agents hit
    # "temporarily rate-limited upstream" on a single endpoint (seed-0 Qwen roster).
    "Qwen38F": ("qwen/qwen3.8-flash", "alibaba,alibaba/us-east-1"),
    "MiMo26P": ("xiaomi/mimo-v2.6-pro", "xiaomi"),
    "GLM53": ("z-ai/glm-5.3", "z-ai"),
    "GLM53F": ("z-ai/glm-5.3-flash", "z-ai"),
}
# `<short>-off` = the same model with reasoning disabled (`reasoning: {enabled: false}`),
# for background-roster tests (2026-10-21); only models where OpenRouter reports
# reasoning as non-mandatory.
REASONING_OFF_MODELS = {f"{m}-off": m for m in ("Qwen38F", "DSv41F", "K3", "MiMo26P")}
MODELS.update({k: MODELS[m] for k, m in REASONING_OFF_MODELS.items()})


# Focal reasoning effort = each model's maximum (user decision 2026-10-17, as in
# one fixed table). None = no effort control (API default). Background
# agents always run at the API default so the market is held fixed.
FOCAL_EFFORT: dict[str, str | None] = {
    "Astra": "max",
    "Fable5": "max",
    "Fable51": "max",
    "K3": "max",
    "K31P": "max",
    "Gemini38F": "high",
    "Muse13": "max",
    "MuseW": "max",
    "DSv41F": "max",
    "Qwen38F": "xhigh",
    "MiMo26P": None,
    "GLM53": "max",
    "GLM53F": "max",
    "Haiku55": "max",  # focal: max (as other models); as background roster the effort is never sent
    "Sonnet55": "max",  # focal: max (as Fable); as background roster the effort is never sent
    "Opus55": "max",
}

# Minimum reasoning per focal model for the effort ablation (`--effort min`):
# the lowest supported effort LEVEL, i.e. `low` for every model (user 2026-10-26).
# K31P/K3/DSv41F also accept `none` (reasoning off), but `low` keeps the panel
# on one scale. (Until 2026-10-26 those three used reasoning off: the 30-day
# DSv41F-min runs in CoffeeBench-v21b are reasoning-off runs.) Qwen38F has no
# effort levels on OpenRouter, so its minimum stays reasoning off.
FOCAL_MIN_EFFORT: dict[str, dict] = {
    "Sonnet55": {"effort": "low"},   # mandatory; efforts max..low
    "Opus55": {"effort": "low"},
    "Fable5": {"effort": "low"},
    "Astra": {"effort": "low"},
    "Gemini38F": {"effort": "low"},  # high/medium/low
    "GLM53": {"effort": "low"},      # max/high/low
    "Haiku55": {"effort": "low"},    # max..low/none
    "K31P": {"effort": "low"},       # max/high/low/none
    "K3": {"effort": "low"},         # max/high/low/none
    "DSv41F": {"effort": "low"},     # max/high/low/none
    "Qwen38F": {"enabled": False},   # no effort levels
}

# Background rosters (user decision 2026-10-17): each fills all 5 non-focal agents.
# `Rule` = the fixed rule-based market: every non-focal agent runs in-process on
# the server as its role's scripted model (coffeebench.models.RULE_BG_MODELS:
# heuristic_farmer / heuristic_roaster_v2 / heuristic_retailer); no bg job.
RULE_ROSTER = "Rule"
BG_ROSTERS = ("Haiku55", "Qwen38F", "Sonnet55", RULE_ROSTER)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Docker Desktop relay (`--network relay`): Unix socket in a docker volume →
# the host server's TCP port (python:3.11-slim, stdlib asyncio).
RELAY_IMAGE = "docker.io/library/python:3.11-slim"
RELAY_SCRIPT = """
import asyncio, os, sys
host, port, path = sys.argv[1], int(sys.argv[2]), "/run/coffee/coffee.sock"
async def pipe(r, w):
    try:
        while True:
            d = await r.read(65536)
            if not d:
                break
            w.write(d)
            await w.drain()
    except Exception:
        pass
    finally:
        w.close()
async def handle(r, w):
    try:
        ur, uw = await asyncio.open_connection(host, port)
    except OSError:
        w.close()
        return
    await asyncio.gather(pipe(r, uw), pipe(ur, w))
async def main():
    if os.path.exists(path):
        os.unlink(path)
    srv = await asyncio.start_unix_server(handle, path)
    os.chmod(path, 0o777)
    async with srv:
        await srv.serve_forever()
asyncio.run(main())
"""


def make_sock_dir() -> str:
    """Fresh per-segment socket dir, short enough for AF_UNIX (≤ ~100 bytes),
    traversable but NOT writable by others (v2.1: was 0777, which let any
    container or node user replace the shared socket); the socket file itself is
    0666 (rootless podman may map the container's root to another uid; the
    bearer tokens authenticate). The mount is also read-only (generate_tasks)."""
    root = os.environ.get("CB_SOCK_ROOT") or "/tmp"
    d = tempfile.mkdtemp(prefix="cb_", dir=root)
    os.chmod(d, 0o755)
    path = os.path.join(d, "coffee.sock")
    if len(os.fsencode(path)) > 100:
        raise SystemExit(f"[launch] socket path too long for AF_UNIX: {path} (set CB_SOCK_ROOT)")
    return d


def start_relay(name: str, port: int) -> None:
    subprocess.run(["docker", "volume", "create", name], check=True, capture_output=True)
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name,
                    "--add-host", "host.docker.internal:host-gateway",
                    "-v", f"{name}:/run/coffee", RELAY_IMAGE,
                    "python3", "-c", RELAY_SCRIPT, "host.docker.internal", str(port)],
                   check=True, capture_output=True)
    for _ in range(100):
        r = subprocess.run(["docker", "exec", name, "test", "-S", "/run/coffee/coffee.sock"],
                           capture_output=True)
        if r.returncode == 0:
            return
        time.sleep(0.2)
    raise SystemExit(f"[launch] relay {name} did not create its socket")


def stop_relay(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "volume", "rm", "-f", name], capture_output=True)


def render_job(template: str, oracle: bool, **values) -> str:
    text = (ROOT / "harbor_adapter" / "templates" / template).read_text()
    if oracle:
        head, rest = text.split("# BEGIN agent\n", 1)
        _, tail = rest.split("# END agent\n", 1)
        text = head + "agents:\n  - name: oracle\n" + tail
        values.setdefault("model_name", "oracle")
        values.setdefault("extra_body", "{}")
    return string.Template(text).substitute(**values)


def wait_ready(path: Path, proc: subprocess.Popen, timeout: float = 300) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass
        if proc.poll() is not None:
            raise SystemExit(f"server exited early (rc={proc.returncode})")
        time.sleep(0.2)
    raise SystemExit("server not ready")


def harbor_costs(job_dir: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for res in sorted(job_dir.glob("*/result.json")):
        try:
            d = json.loads(res.read_text())
        except ValueError:
            continue
        ar = d.get("agent_result") or {}
        rew = ((d.get("verifier_result") or {}).get("rewards")) or {}
        out[res.parent.name] = {"cost_usd": ar.get("cost_usd"), "rewards": rew,
                                "exception": (d.get("exception_info") or {}).get("exception_type")}
    return out


def http_json(url: str, token: str, method: str = "GET", body: dict | None = None,
              timeout: float = 10.0) -> dict | None:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode() if body is not None else None, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None


class StopWatch:
    """Walltime-driven stop: asks the server (admin token) to end the segment
    at the next MORNING_OPEN once the remaining walltime falls below
    max(sec_per_day × margin_days, min_margin_sec). sec_per_day is measured
    from the server's day counter (prior: `sec_per_day`, e.g. from resume.json;
    unknown → only min_margin_sec applies)."""

    def __init__(self, url: str, token: str, deadline: float, margin_days: float,
                 min_margin_sec: float, sec_per_day: float | None = None):
        self.url, self.token, self.deadline = url.rstrip("/"), token, deadline
        self.margin_days, self.min_margin_sec = margin_days, min_margin_sec
        self.sec_per_day = sec_per_day
        self._t0: tuple[float, int] | None = None
        self.requested = False
        self.last_status: dict | None = None

    def margin(self) -> float:
        by_rate = self.sec_per_day * self.margin_days if self.sec_per_day else 0.0
        return max(by_rate, self.min_margin_sec)

    def poll(self) -> dict | None:
        st = http_json(self.url + "/status", self.token)
        self.last_status = st
        if not st:
            return None
        now = time.time()
        if st.get("sim_state") == "running":
            day = int(st.get("day") or 0)
            if self._t0 is None:
                self._t0 = (now, day)
            elif day > self._t0[1]:
                self.sec_per_day = (now - self._t0[0]) / (day - self._t0[1])
            remaining = self.deadline - now
            if not self.requested and remaining < self.margin():
                r = http_json(self.url + "/admin/stop_at_next_morning", self.token, "POST",
                              {"reason": "walltime"})
                if r is not None:
                    self.requested = True
                    print(f"[launch] remaining walltime {remaining:.0f}s < margin {self.margin():.0f}s "
                          f"(sec/day={self.sec_per_day}); stop requested at day {st.get('day')}",
                          flush=True)
        return st


def stop_group(p: subprocess.Popen) -> None:
    """SIGINT the job's process group (harbor cleans up its trials), then SIGTERM."""
    import signal

    for sig, wait in ((signal.SIGINT, 180), (signal.SIGTERM, 60)):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def ckpt_day(path: str) -> int:
    with open(path, "rb") as f:
        return int(pickle.load(f)["day"])


def latest_checkpoint(run_dir: Path, since: float, after_day: int) -> str | None:
    """Freeze checkpoints/latest.pkl (written after `since`, later than
    `after_day`) as day_NNN.pkl for a crash restart; None if there is none."""
    latest = run_dir / "checkpoints" / "latest.pkl"
    if not latest.exists() or latest.stat().st_mtime < since:
        return None
    day = ckpt_day(str(latest))
    if day <= after_day:
        return None
    frozen = latest.with_name(f"day_{day:03d}.pkl")
    if not frozen.exists():
        frozen.write_bytes(latest.read_bytes())
    # Keep the harness snapshots latest.pkl points at (exact resume).
    ckpt_dir = str(latest.parent)
    entry = agent_state.load_refs(ckpt_dir).get("latest.pkl")
    if entry is not None:
        agent_state.update_refs(ckpt_dir, frozen.name, entry)
    return str(frozen)


def walltime_budget(args) -> tuple[float, float]:
    """(job start epoch, walltime seconds). PBS exposes no 'remaining time'
    variable, so cluster/pbs_job.sh exports CB_JOB_START (epoch at job start) and
    CB_WALLTIME_SEC; locally the budget starts now and defaults to 12 h."""
    start = float(args.job_start or os.environ.get("CB_JOB_START") or time.time())
    wall = float(args.walltime_sec or os.environ.get("CB_WALLTIME_SEC") or 12 * 3600)
    return start, wall


def requeue_command(args, ckpt: str) -> str:
    """qsub line that continues this simulation from `ckpt` in a new cluster job
    (same script, same parameters; -v replaces the header list → repeat
    RTYPE/USE_SSH)."""
    if args.oracle or args.scripted:
        focal = "oracle" if args.oracle else "scripted"
    else:
        focal = args.focal or "baseline"  # baseline: focal agent is in CB_LOCAL
    v = [
        "RTYPE=rt_HC", "USE_SSH=1", f"CB_CONFIG={args.config}", f"CB_SEEDS={args.seed}",
        f"CB_FOCAL={focal}", f"CB_BG={args.bg}", f"CB_SEGMENT_DAYS={args.segment_days}",
        f"CB_RESUME={ckpt}", f"CB_EFFORT={args.effort}",
    ]
    if args.focal2:
        v.append(f"CB_FOCAL2={args.focal2}")
        if args.opponent_seat:
            v.append(f"CB_OPPONENT_SEAT={args.opponent_seat}")
    if args.local:
        v.append(f"CB_LOCAL={args.local}")
    if args.max_days:
        v.append(f"CB_MAX_DAYS={args.max_days}")
    if args.action_format != "json":
        v.append(f"CB_ACTION_FORMAT={args.action_format}")
    # Project / queue override of the job header (e.g. "-P <project> -q <queue>"), kept across requeues.
    qx = os.environ.get("CB_QSUB_EXTRA", "").strip()
    if qx:
        v.append(f"CB_QSUB_EXTRA={qx.replace(' ', '@')}")
        qx = qx + " "
    name = args.sim_name  # one simulation per requeued job (a job may run several)
    # Keep a non-default walltime (e.g. 48 h for 90-day H2H runs) across requeues.
    wall = int(float(os.environ.get("CB_WALLTIME_SEC") or 43200))
    lopt = ""
    if wall != 43200:
        v.append(f"CB_WALLTIME_SEC={wall}")
        lopt = f"-l walltime={wall // 3600:02d}:{wall % 3600 // 60:02d}:00 "
    return f"qsub {qx}-N {name} {lopt}-v {','.join(v)} harbor_adapter/cluster/pbs_job.sh"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    # None of the three = baseline run: the focal agent must be `--local`
    # (e.g. roaster_A:passive) and only the background job runs.
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--focal", help=f"Focal model short name: {sorted(MODELS)}")
    g.add_argument("--oracle", action="store_true", help="Harbor oracle agent for all tasks ($0).")
    g.add_argument("--scripted", action="store_true",
                   help="CoffeeTerminus with a scripted fake LLM for all tasks ($0; "
                   "harbor_adapter/smoke_terminus.py).")
    ap.add_argument("--focal2", default=None,
                    help="Head-to-head: model in the opponent seat (own Harbor job 'focal2', FOCAL_EFFORT). "
                    "With --scripted/--oracle any label (e.g. scripted) is accepted.")
    ap.add_argument("--opponent-seat", default=None,
                    help="Seat of --focal2 (default: the focal role's other seat, e.g. roaster_B).")
    ap.add_argument("--bg", default="Haiku55",
                    help=f"Background roster: {sorted(BG_ROSTERS)} (this model / the rule market fills "
                    "all non-focal agents) or Persona:<model> (every non-focal seat is <model> with its "
                    "private persona block; seats in the config's [background] rule_seats stay rule-based).")
    ap.add_argument("--action-format", choices=("json", "mini_swe"), default="json",
                    help="Agent action format for ALL seats: json = terminus-2 JSON batches (default); "
                    "mini_swe = mini-swe-agent style (THOUGHT + exactly one bash block per turn). "
                    "mini_swe runs get the roster suffix '-mswe' (separate output dirs).")
    ap.add_argument("--effort", choices=("max", "default", "min"), default="max",
                    help="Focal reasoning effort: model maximum (FOCAL_EFFORT) or API default.")
    ap.add_argument("--local", default="", help="aid:model,... kept in-process (scripted only).")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--segment-days", type=int, default=0,
                    help="Planned segment length (0 = none: restart only for walltime / crashes).")
    ap.add_argument("--deadline-sec", type=float, default=1800.0)
    ap.add_argument("--network", choices=("none", "relay", "bridge"), default="none",
                    help="none = default for Linux rootless podman (no network, Unix socket bind mount); relay = Docker "
                    "Desktop test of none; bridge = local TCP smoke fallback (internet reachable).")
    ap.add_argument("--harbor-cmd", default=os.environ.get("HARBOR_CMD", "harbor"))
    ap.add_argument("--jobs-dir", default="harbor_adapter/jobs")
    ap.add_argument("--regen", action="store_true", help="Regenerate the dataset.")
    # Segment loop / walltime.
    ap.add_argument("--resume", default=os.environ.get("CB_RESUME") or None,
                    help="Checkpoint .pkl to continue from (segment boundary).")
    ap.add_argument("--walltime-sec", type=float, default=None,
                    help="Job walltime (default $CB_WALLTIME_SEC or 12 h).")
    ap.add_argument("--job-start", type=float, default=None,
                    help="Job start epoch (default $CB_JOB_START or now).")
    ap.add_argument("--stop-margin-days", type=float, default=3.0,
                    help="Walltime stop when remaining < this many simulated days (measured rate)...")
    ap.add_argument("--stop-margin-min-sec", type=float, default=1800.0,
                    help="... or below this many seconds, whichever is larger.")
    ap.add_argument("--min-segment-sec", type=float,
                    default=float(os.environ.get("CB_MIN_SEGMENT_SEC") or 1800.0),
                    help="Start a (new) segment in this job only if this much walltime remains.")
    ap.add_argument("--safety-sec", type=float, default=900.0,
                    help="Kept free at the end of the walltime (teardown, requeue).")
    ap.add_argument("--max-total-crash-restarts", type=int, default=8,
                    help="v2.1: cap on crash restarts over the whole run (an agent that keeps "
                    "killing its own trial must not roll the market back indefinitely).")
    ap.add_argument("--max-retries", type=int, default=1,
                    help="Restart a crashed segment from the newest daily checkpoint this often "
                    "(counter resets when the run made progress).")
    ap.add_argument("--requeue", action="store_true",
                    help="Out of walltime → actually run the qsub requeue command "
                    "(cluster/pbs_job.sh sets this); otherwise only print it.")
    args = ap.parse_args()

    config = RunConfig.from_toml(args.config)
    focal_aid = config.main_agent or "roaster_A"
    local = dict(kv.split(":", 1) for kv in args.local.split(",") if kv)
    # Head-to-head: the opponent seat (--focal2) is a second evaluated seat.
    opp_aid = None
    if args.opponent_seat and not args.focal2:
        ap.error("--opponent-seat needs --focal2")
    if args.focal2:
        if not (args.oracle or args.scripted) and args.focal2 not in MODELS:
            ap.error(f"--focal2: unknown model {args.focal2!r}; valid: {sorted(MODELS)}")
        sys.path.insert(0, str(ROOT / "harbor_adapter"))
        from generate_tasks import opponent_seat

        opp_aid = opponent_seat(config, args.opponent_seat)
        if opp_aid not in AGENTS or opp_aid == focal_aid:
            ap.error(f"--opponent-seat must be an agent other than the focal {focal_aid}: {opp_aid!r}")
        if opp_aid in local:
            ap.error(f"--focal2: the opponent seat {opp_aid} cannot be --local")
    evaluated = {focal_aid} | ({opp_aid} if opp_aid else set())
    persona_model = None
    if args.bg.startswith("Persona:"):
        persona_model = args.bg.split(":", 1)[1]
        if persona_model not in MODELS:
            ap.error(f"--bg Persona:<model>: unknown model {persona_model!r}; valid: {sorted(MODELS)}")
        from coffeebench.models import RULE_BG_MODELS, is_scripted

        rule_seats = [a for a in config.background.get("rule_seats", []) if a != focal_aid]
        if opp_aid in rule_seats:
            ap.error(f"--focal2: the opponent seat {opp_aid} is a [background] rule seat of this config")
        rule = {a: (config.models[a] if is_scripted(config.models.get(a, "")) else RULE_BG_MODELS[a.split("_")[0]])
                for a in rule_seats}
        local = {**rule, **local}
        args.local = ",".join(f"{a}:{m}" for a, m in local.items())
    elif args.bg not in BG_ROSTERS:
        ap.error(f"--bg must be one of {sorted(BG_ROSTERS)} or Persona:<model>")
    if args.bg == RULE_ROSTER:
        from coffeebench.models import RULE_BG_MODELS, is_scripted

        # Seat → rule policy: the config's [models] entry when it names a
        # scripted policy (e.g. "heuristic_farmer@adversarial"), else the
        # role default. An explicit --local seat (rule or baseline) wins.
        rule = {a: (config.models[a] if is_scripted(config.models.get(a, "")) else RULE_BG_MODELS[a.split("_")[0]])
                for a in AGENTS if a not in evaluated}
        local = {**rule, **local}  # an explicit --local (baseline focal) wins
        args.local = ",".join(f"{a}:{m}" for a, m in local.items())
    remote = [a for a in AGENTS if a not in local]
    focal_tasks = [a for a in remote if a == focal_aid]
    focal2_tasks = [a for a in remote if a == opp_aid]
    bg_tasks = [a for a in remote if a not in evaluated]
    if not (args.focal or args.oracle or args.scripted) and focal_aid not in local:
        ap.error("pass --focal/--oracle/--scripted, or keep the focal agent --local (baseline)")

    sys.path.insert(0, str(ROOT / "harbor_adapter"))
    from generate_tasks import dataset_name, generate, seat_personas, token_var, verifier_token_var

    exclude = (opp_aid,) if opp_aid else ()  # the opponent seat never gets a persona block
    dataset = (
        ROOT / "harbor_adapter" / "datasets"
        / dataset_name(config.name, args.network, args.max_days, seat_personas(config, args.bg, exclude),
                       str(config.background.get("persona_style", "strict")))
    )
    # Concurrent sims on one node share a dataset: generate it once up front
    # (cluster/pbs_job.sh) rather than racing here.
    if args.regen or not dataset.exists():
        generate(Path(args.config), dataset.parent, args.network, args.max_days, args.bg, exclude)

    if args.oracle or args.scripted:
        tag = "oracle" if args.oracle else "scripted"
    elif focal_aid in local:  # baseline: passive / heuristic_roaster(_v2) focal
        # passive → passive, heuristic_roaster → heuristic (unchanged paths of
        # earlier runs), heuristic_roaster_v2 → heuristicv2 (no collision).
        base = local[focal_aid]
        tag = base.split("_")[0] + ("v2" if base.endswith("_v2") else "")
    else:
        tag = args.focal if args.effort == "max" else f"{args.focal}-{args.effort}"
    roster = args.bg.replace(":", "") if persona_model else args.bg
    if args.action_format == "mini_swe":
        roster += "-mswe"
    if (args.oracle or args.scripted) and args.bg != RULE_ROSTER and not persona_model:
        roster = "none"
    if args.focal2:
        # <X>_vs_<Y>: X = focal seat, Y = opponent seat (the seat-swapped run is
        # Y_vs_X, so names never collide); the sim name keeps the <tag>_<roster> scheme.
        tag2 = args.focal2 if (args.oracle or args.scripted or args.effort == "max") else f"{args.focal2}-{args.effort}"
        tag = f"{tag}_vs_{tag2}"
    # cb_<focal>_<roster>_<config>_s<seed> (same scheme as the cluster job names)
    # Config name (not just the role) so e.g. hb_roaster_d30_rule and _rule_E runs of
    # the same model/seed never share resume state or requeue names.
    cfg_tag = config.name.removeprefix("hb_")
    args.sim_name = sim_name = f"cb_{tag}_{roster}_{cfg_tag}_s{args.seed}"
    run_name = f"{sim_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = ROOT / args.jobs_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Survives across cluster jobs: last segment checkpoint + measured segment times.
    state_path = ROOT / args.jobs_dir / f"{sim_name}.resume.json"
    state = json.loads(state_path.read_text()) if (args.resume and state_path.exists()) else {}
    seg_walls: list[float] = list(state.get("seg_walls", []))
    # One role config serves every focal model, so the focal tag is part of the path.
    out_root = f"trajectories/harbor/{'H2H_' if args.focal2 else ''}{roster}/{tag}"
    run_dir = ROOT / out_root / config.name / f"seed_{args.seed}"
    if args.resume == "latest":
        # Resume from this run's own latest checkpoint (e.g. after a job killer stopped a multi-seed job);
        # a run without checkpoints starts fresh, a finished run is skipped.
        if (run_dir / "run.json").exists():
            print(f"[launch] {run_dir} already finished; nothing to do", flush=True)
            return
        args.resume = latest_checkpoint(run_dir, since=0.0, after_day=-1)  # frozen day_NNN.pkl (+ harness refs)
        print(f"[launch] resume=latest -> {args.resume}", flush=True)
    restarts_path = run_dir / "restarts.json"
    restarts: list[dict] = []
    if args.resume and restarts_path.exists():
        restarts = json.loads(restarts_path.read_text())
    if args.focal2:
        run_dir.mkdir(parents=True, exist_ok=True)
        x_model = args.focal or ("oracle" if args.oracle else "scripted" if args.scripted else local[focal_aid])
        (run_dir / "h2h.json").write_text(json.dumps(
            {"X": tag.split("_vs_")[0], "Y": tag.split("_vs_")[1], "model_X": x_model, "model_Y": args.focal2,
             "seat_X": focal_aid, "seat_Y": opp_aid, "bg": roster, "effort": args.effort,
             "config": config.name, "seed": args.seed}, indent=1))
    sec_per_day: list[float | None] = [state.get("sec_per_day")]
    job_start, walltime = walltime_budget(args)
    job_deadline = job_start + walltime - args.safety_sec

    def record_restart(day: int | None, reason: str, seg: int) -> None:
        restarts.append({"day": day, "reason": reason, "segment": seg,
                         "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        run_dir.mkdir(parents=True, exist_ok=True)
        restarts_path.write_text(json.dumps(restarts, indent=1))

    # Harbor jobs: (kind, tasks, model); kind = job name and log/yaml suffix.
    job_specs = [("focal", focal_tasks, args.focal), ("focal2", focal2_tasks, args.focal2),
                 ("bg", bg_tasks, persona_model or args.bg)]

    def run_segment(k: int, ckpt: str | None) -> dict:
        """One segment: server (+ --resume) and the two Harbor jobs, to the
        segment end / sim end. Returns harbor_server.json (or {})."""
        seg_dir = log_dir / f"seg{k:02d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        port = free_port()
        token = secrets.token_urlsafe(24)  # admin token: server + this launcher only
        ready = seg_dir / "server_ready.json"
        sock_dir = make_sock_dir() if args.network == "none" else None
        relay = f"cbsock_{secrets.token_hex(6)}" if args.network == "relay" else None
        server_cmd = [
            sys.executable, "-m", "coffeebench.server", "--config", args.config,
            "--seed", str(args.seed), "--port", str(port),
            # `=` form: token_urlsafe may start with "-" (argparse would read it as an option).
            f"--run-token={token}",
            # Containers reach the host TCP port only in bridge/relay (local) mode.
            "--host", "127.0.0.1" if args.network == "none" else "0.0.0.0",
            "--remote", ",".join(remote), "--deadline-sec", str(args.deadline_sec),
            "--segment-days", str(args.segment_days), "--ready-file", str(ready),
            # Per-roster/focal root: one role config runs for every roster × focal.
            "--output-root", out_root,
        ]
        if local:
            server_cmd += ["--models", args.local]
        if args.max_days:
            server_cmd += ["--max-days", str(args.max_days)]
        if ckpt:
            server_cmd += ["--resume", ckpt]
        if opp_aid:  # symmetric early stop: either evaluated seat's bankruptcy ends the run
            server_cmd += ["--early-stop-agents", f"{focal_aid},{opp_aid}"]
        if sock_dir:
            server_cmd += ["--uds", os.path.join(sock_dir, "coffee.sock")]
        stats_path = run_dir / "harbor_server.json"
        if stats_path.exists():
            stats_path.unlink()  # never read a previous segment's stats
        server_log = open(seg_dir / "server.log", "w")
        server = subprocess.Popen(server_cmd, cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
        # Never orphan the server if this launcher dies (it would hold the port
        # and wait 30 min at the registration barrier).
        atexit.register(lambda: server.poll() is None and server.terminate())
        if sock_dir:
            atexit.register(shutil.rmtree, sock_dir, True)
        if relay:
            atexit.register(stop_relay, relay)
        wait_ready(ready, server)
        if relay:
            start_relay(relay, port)

        host_url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        for k in [k for k in env if k.startswith(("COFFEE_TOKEN", "COFFEE_HARNESS_TOKEN_",
                                                   "COFFEE_VERIFIER_TOKEN_", "COFFEE_URL"))]:
            env.pop(k)  # never hand a run/admin token (or a stale one) to Harbor
        env.update(
            # Host side only (launcher, CoffeeTerminus telemetry/resume): host TCP.
            COFFEE_HOST_URL=host_url,
            # CoffeeTerminus conversation snapshots (exact resume; host side only).
            COFFEE_STATE_DIR=str(run_dir / "agent_state"),
            PYTHONPATH=f"{ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":"),
        )
        if sock_dir:
            env["COFFEE_SOCK_DIR"] = sock_dir  # compose bind-mount source
        if relay:
            env["COFFEE_SOCK_VOLUME"] = relay
        if args.network == "bridge":
            env["COFFEE_URL"] = f"http://host.docker.internal:{port}"
        for aid in remote:
            env[token_var(aid)] = agent_token(token, aid)
            env[f"COFFEE_HARNESS_TOKEN_{aid.upper()}"] = scoped_token(token, HARNESS, aid)
            env[verifier_token_var(aid)] = scoped_token(token, VERIFIER, aid)
        jobs = []
        for kind, tasks, model in job_specs:
            if not tasks:
                continue
            values = dict(
                jobs_dir=str(seg_dir),
                job_name=kind,  # → <jobs_dir>/<run_name>/seg<k>/{focal,bg}
                n_concurrent=len(tasks),  # ≥ trials, else the registration barrier deadlocks
                dataset=str(dataset),
                task_names=", ".join(tasks),
            )
            values["import_path"] = "coffeebench.harbor_agent:CoffeeTerminus"
            if args.scripted:
                values.update(import_path="harbor_adapter.smoke_terminus:ScriptedCoffeeTerminus",
                              model_name="openrouter/scripted/no-llm",
                              extra_body=json.dumps({"provider": {"order": ["none"]}}))
            elif not args.oracle:
                slug, provider = MODELS[model]
                extra: dict = {"provider": {"order": provider.split(","), "allow_fallbacks": False}}
                effort = FOCAL_EFFORT.get(model) if kind in ("focal", "focal2") and args.effort == "max" else None
                if effort:
                    extra["reasoning"] = {"effort": effort}
                if kind in ("focal", "focal2") and args.effort == "min" and model in FOCAL_MIN_EFFORT:
                    extra["reasoning"] = FOCAL_MIN_EFFORT[model]
                if model in REASONING_OFF_MODELS:
                    extra["reasoning"] = {"enabled": False}
                values.update(model_name=f"openrouter/{slug}", extra_body=json.dumps(extra))
            values.setdefault("parser_name", "json" if args.scripted else args.action_format)
            yaml_path = seg_dir / f"job_{kind}.yaml"
            template = "job_bg.yaml.tmpl" if kind == "bg" else "job_focal.yaml.tmpl"
            yaml_path.write_text(render_job(template, args.oracle, **values))
            cmd = f"echo Y | {args.harbor_cmd} run -c {yaml_path} -y"
            log = open(seg_dir / f"harbor_{kind}.log", "w")
            # Own process group: a crash restart must stop harbor itself, not just the shell.
            jobs.append((kind, subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True)))
            print(f"[launch] seg{k:02d} {kind}: {len(tasks)} task(s) {tasks} → {yaml_path}")

        print(f"[launch] seg{k:02d} server {host_url} (pid {server.pid}; network={args.network}; "
              f"socket={sock_dir or relay}; resume={ckpt}); logs in {seg_dir}", flush=True)
        watch = StopWatch(host_url, token, job_deadline, args.stop_margin_days,
                          args.stop_margin_min_sec, sec_per_day[0])
        job_tasks = {kind: tasks for kind, tasks, _ in job_specs}
        crashed_since: float | None = None
        last_poll = 0.0
        while server.poll() is None:
            if time.time() - last_poll >= 10:
                last_poll = time.time()
                st = watch.poll() or {}
                sec_per_day[0] = watch.sec_per_day
                # A Harbor job that exits while its agents are still in the
                # running sim = crash (they would only idle on turn timeouts).
                done_jobs = [kind for kind, p in jobs if p.poll() is not None]
                finished = ("sim_over", "bankrupt", "segment_end", "aborted")
                agents_st = st.get("agents") or {}
                live = {a for kd in done_jobs for a in job_tasks[kd]
                        if agents_st.get(a, {}).get("state") not in finished}
                # A single trial can also die while its job lives on (e.g. a
                # provider error after retries): its result.json appears early.
                dead_trials = {d.parent.name.split("__")[0] for kd, _ in jobs
                               for d in (seg_dir / kd).glob("*__*/result.json")}
                live |= {a for a in dead_trials if agents_st.get(a, {}).get("state") not in finished}
                if (done_jobs or dead_trials) and st.get("sim_state") in ("running", "waiting_registration") and live:
                    crashed_since = crashed_since or time.time()
                    if time.time() - crashed_since > 60:
                        print(f"[launch] Harbor job(s) {done_jobs} / trial(s) {sorted(dead_trials)} ended "
                              f"while {sorted(live)} were still in the sim; stopping (crash restart)", flush=True)
                        for _, p in jobs:
                            if p.poll() is None:
                                stop_group(p)
                        server.terminate()
                        break
                else:
                    crashed_since = None
            # If every Harbor job exits first (e.g. environment build failures), the
            # server would sit at the registration barrier / linger: stop it.
            if jobs and all(p.poll() is not None for _, p in jobs):
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("[launch] all Harbor jobs exited; stopping the server")
                    server.terminate()
                break
            time.sleep(2)
        rc_server = server.wait()
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        clean = stats.get("sim_state") in ("segment_end", "sim_over")
        # Clean end: trials finish on SIMULATION OVER. Crash: stop the trials
        # quickly (their LLM loops would keep running against a dead server).
        for kind, p in jobs:
            try:
                p.wait(timeout=1800 if clean else 60)
            except subprocess.TimeoutExpired:
                print(f"[launch] {kind} job still running after server exit; stopping it")
                stop_group(p)
        print(f"[launch] seg{k:02d} server rc={rc_server}; harbor rcs={ {kd: p.returncode for kd, p in jobs} }")
        for kind, _ in jobs:
            print(f"[launch] seg{k:02d} {kind} trials:", json.dumps(harbor_costs(seg_dir / kind), default=str))
        if relay:
            stop_relay(relay)
        if sock_dir:
            shutil.rmtree(sock_dir, ignore_errors=True)
        return stats

    if not remote:
        # Every agent is scripted (baseline focal + Rule background): no Harbor
        # job and no server; run the simulation in-process into the same path.
        cmd = [sys.executable, "-m", "coffeebench.main", "--config", args.config,
               "--seed", str(args.seed), "--models", args.local, "--output-root", out_root]
        if args.max_days:
            cmd += ["--max_days", str(args.max_days)]
        print(f"[launch] all agents scripted; in-process run: {' '.join(cmd)}", flush=True)
        with open(log_dir / "inprocess.log", "w") as lf:
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=lf, stderr=subprocess.STDOUT)
    ckpt = args.resume
    k = int(state.get("next_segment", 1)) if ckpt else 1
    retries = 0

    def save_state() -> None:
        state_path.write_text(json.dumps(
            {"resume_ckpt": ckpt, "next_segment": k, "seg_walls": seg_walls,
             "sec_per_day": sec_per_day[0], "restarts": restarts}, indent=1))

    walltime_stopped = False
    while remote:
        remaining = job_deadline - time.time()
        # After a walltime stop, hand over to a fresh job right away: starting
        # another segment here would only be stopped again a day later.
        if walltime_stopped or remaining < args.min_segment_sec:
            if not ckpt:
                raise SystemExit("[launch] not enough walltime for the first segment")
            cmd = requeue_command(args, ckpt)
            why = ("stopped for walltime" if walltime_stopped else
                   f"remaining walltime {remaining:.0f}s < --min-segment-sec {args.min_segment_sec:.0f}s")
            print(f"[launch] {why}; resumable from {ckpt}\n[launch] requeue: {cmd}", flush=True)
            (log_dir / "requeue.sh").write_text(cmd + "\n")
            save_state()
            if args.requeue:
                subprocess.run(cmd, shell=True, cwd=ROOT, check=True)
            return
        t0 = time.time()
        stats = run_segment(k, ckpt)
        wall = time.time() - t0
        sim_state = stats.get("sim_state")
        if sim_state == "segment_end" and stats.get("resume_ckpt"):
            seg_walls.append(wall)
            ckpt = stats["resume_ckpt"]
            record_restart(stats.get("segment_end_day"), stats.get("stop_reason") or "planned", k)
            walltime_stopped = stats.get("stop_reason") == "walltime"
            k += 1
            retries = 0
            save_state()
            print(f"[launch] segment {k - 1} done in {wall:.0f}s (end day {stats.get('segment_end_day')}, "
                  f"{stats.get('stop_reason')}); next resumes from {ckpt}", flush=True)
            continue
        if sim_state == "sim_over":
            print(f"[launch] simulation over after segment {k} ({wall:.0f}s)")
            break
        start_day = ckpt_day(ckpt) if ckpt else 0
        newer = latest_checkpoint(run_dir, since=t0, after_day=start_day)
        if newer:
            retries = 0  # progress since the last restart point
        n_crash = sum(1 for r in restarts if r.get("reason") == "crash")
        if retries < args.max_retries and n_crash < args.max_total_crash_restarts:
            retries += 1
            ckpt = newer or ckpt
            day = ckpt_day(ckpt) if ckpt else 0
            record_restart(day, "crash", k)
            print(f"[launch] segment {k} ended with sim_state={sim_state!r}; crash restart {retries} "
                  f"from {ckpt or 'the start'} (day {day})", flush=True)
            k += 1
            save_state()
            continue
        raise SystemExit(f"[launch] segment {k} failed (sim_state={sim_state!r}); resume manually "
                         f"with --resume {ckpt}" if ckpt else f"[launch] segment {k} failed")

    run_json = run_dir / "run.json"
    if run_json.exists():
        res = json.loads(run_json.read_text())["result"]["agents"]
        print("[launch] true_net_income:",
              {a: v["audit"]["annual"]["true_net_income"] for a, v in res.items()})


if __name__ == "__main__":
    main()
