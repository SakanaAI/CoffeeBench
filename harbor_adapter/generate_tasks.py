"""CoffeeBench config → Harbor dataset (one task per agent id).

    unset VIRTUAL_ENV; uv run python harbor_adapter/generate_tasks.py \
        --config experiments/camera_ready/cr_sonnetbg_roaster_opus47.toml [--network none|relay|bridge]

Output: harbor_adapter/datasets/<config name>[_d<days>][_<network>]/<agent_id>/
    instruction.md          rules prompt (coffeebench.rules_prompt) + CLI guide + tool list
    task.toml               server transport + this agent's own token (COFFEE_TOKEN=
                            ${COFFEE_TOKEN_<AID>}, set by launch_sim on the host only), static
                            COFFEE_AGENT_ID; the verifier gets ${COFFEE_VERIFIER_TOKEN_<AID>}
    environment/            Dockerfile (python:3.11-slim + tmux + curl + coffee), coffee.py,
                            docker-compose.yaml (see network modes)
    tests/test.sh           GET /audit/$COFFEE_AGENT_ID → /logs/verifier/reward.json
    solution/solve.sh       scripted oracle loop (view_trial_balance + wait_for_next_day)
                            + isolation checks (solution/isolation_check.py)

Network modes (`--network`):
    none    DEFAULT (Linux rootless podman). Compose `network_mode: none`: no
            network at all. The env server's Unix socket dir (host path
            $COFFEE_SOCK_DIR, set by launch_sim per segment) is bind-mounted at
            /run/coffee; the CLI and the verifier use COFFEE_SOCKET=/run/coffee/coffee.sock.
    relay   Local Docker Desktop test of `none`: Docker Desktop cannot pass a host Unix
            socket into a Linux container, so launch_sim runs a relay container (Unix
            socket in the docker volume $COFFEE_SOCK_VOLUME → host TCP); the task
            containers are still `network_mode: none`.
    bridge  Local smoke fallback only: default network (internet reachable!) + TCP
            COFFEE_URL=http://host.docker.internal:<port>.
Harbor's own `[environment] network_mode = "no-network"` is NOT used: on the docker
backend (harbor 0.24.0, environments/docker/docker.py `_requires_egress_control`)
every non-public policy starts the egress-control sidecar (NET_ADMIN + nftables,
kernel probe; unsupported → task rejected), which rootless podman is not
expected to support. Compose `network_mode: none` on `main` needs nothing, and
Harbor leaves services with an explicit network_mode alone.

The task set is the same for every seed/run: the per-simulation socket dir and
tokens are injected through the host environment when the job is launched
(harbor_adapter/launch_sim.py).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coffeebench.business_app import BusinessApp  # noqa: E402
from coffeebench.config import RunConfig  # noqa: E402
from coffeebench import mechanics as mech_mod  # noqa: E402
from coffeebench.main import tool_names_for  # noqa: E402
from coffeebench.server.prompts import (  # noqa: E402
    build_instruction,
    build_system_prompts,
    tool_catalog,
)

PREWARM_IMAGE = "cb_prewarm"
NETWORKS = ("none", "relay", "bridge")
SOCK_MOUNT = "/run/coffee"
SOCKET_PATH = f"{SOCK_MOUNT}/coffee.sock"

TASK_TOML = """\
schema_version = "1.0"

[task]
name = "coffeebench/{cfg}-{aid}"
description = "CoffeeBench: operate {aid} ({role}) in a shared 6-agent coffee supply chain ({max_days} simulated days)."
authors = [{{ name = "CoffeeBench" }}]
keywords = ["coffeebench", "multi-agent", "{role}"]

[metadata]
config = "{cfg}"
agent_id = "{aid}"
role = "{role}"
max_days = {max_days}
network = "{network}"

[verifier]
timeout_sec = 600.0

[verifier.env]
{transport}COFFEE_TOKEN = "${{{verifier_var}}}"
COFFEE_AGENT_ID = "{aid}"

[solution.env]
{transport}COFFEE_TOKEN = "${{{token_var}}}"
COFFEE_AGENT_ID = "{aid}"

[agent]
# Above a 48 h cluster walltime: harbor must never end a trial mid-simulation (the
# 11.5 h value from the 12 h-segment design killed agents in long Qwen-roster runs).
timeout_sec = 180000.0

[environment]
build_timeout_sec = 1200.0
cpus = 1
memory_mb = 1024
storage_mb = 4096

[environment.env]
{transport}COFFEE_TOKEN = "${{{token_var}}}"
COFFEE_AGENT_ID = "{aid}"
COFFEE_OUT_DIR = "/workspace/out"
COFFEE_MAX_WAIT = "6900"
"""

DOCKERFILE = """\
FROM docker.io/library/python:3.11-slim
RUN apt-get -o APT::Sandbox::User=root update \\
    && apt-get -o APT::Sandbox::User=root install -y --no-install-recommends tmux curl \\
    && rm -rf /var/lib/apt/lists/*
COPY coffee.py /usr/local/bin/coffee
RUN chmod 755 /usr/local/bin/coffee && mkdir -p /workspace/out \\
    && printf '%s\\n' 'PS1="\\u@coffee:\\w\\$ "' 'set +H' >> /root/.bashrc
# Fixed prompt (no container id): identical terminal output across restarts.
# set +H: no history expansion, so "Available!" in a message is not an error
# ("event not found" aborted the line and its blocking suffix -> 2 h hangs).
WORKDIR /workspace
CMD ["/bin/bash"]
"""

COMPOSE_NONE = f"""\
# No network at all. The env server's Unix socket dir (host, per segment) is
# bind-mounted; COFFEE_SOCK_DIR comes from the Harbor host env (launch_sim).
services:
  main:
    network_mode: none
    volumes:
      - type: bind
        source: ${{COFFEE_SOCK_DIR:?COFFEE_SOCK_DIR is not set (launch_sim sets it)}}
        target: {SOCK_MOUNT}
        # Read-only (v2.1 security fix): connecting to a Unix socket works on a
        # read-only mount, but no agent can rename/replace/delete the shared
        # socket (MITM of the other agents) or fill the host directory.
        read_only: true
"""

COMPOSE_RELAY = f"""\
# Docker Desktop test of the no-network setup: the socket lives in a docker
# volume served by launch_sim's relay container (-> host TCP).
services:
  main:
    network_mode: none
    volumes:
      - type: volume
        source: coffee_sock
        target: {SOCK_MOUNT}
volumes:
  coffee_sock:
    external: true
    name: ${{COFFEE_SOCK_VOLUME:?COFFEE_SOCK_VOLUME is not set (launch_sim sets it)}}
"""

COMPOSE_BRIDGE = """\
# Local smoke fallback (Docker Desktop): reach the host server over TCP as
# host.docker.internal. NOTE: this container can reach the internet.
services:
  main:
    extra_hosts:
      - "host.docker.internal:host-gateway"
"""

TEST_SH = """\
#!/bin/bash
# Score = this agent's truth-ledger net income from the env server (verifier
# token; Unix socket $COFFEE_SOCKET, else TCP $COFFEE_URL).
# reward = true_net_income on the final segment, 0 otherwise (bankrupt agents
# keep their frozen NI here; the leaderboard treats them as DNF from run.json).
mkdir -p /logs/verifier
python3 - <<'EOF'
import http.client, json, os, socket, time, urllib.parse


class UnixConn(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("coffee", timeout=30)
        self.sock_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        self.sock.connect(self.sock_path)


def conn():
    if os.environ.get("COFFEE_SOCKET"):
        return UnixConn(os.environ["COFFEE_SOCKET"]), ""
    u = urllib.parse.urlsplit(os.environ["COFFEE_URL"])
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=30), u.path.rstrip("/")


out = {"reward": 0.0, "error": 1}
for attempt in range(10):
    try:
        c, base = conn()
        c.request("GET", base + "/audit/" + os.environ["COFFEE_AGENT_ID"],
                  headers={"Authorization": "Bearer " + os.environ["COFFEE_TOKEN"]})
        r = c.getresponse()
        body = r.read()
        c.close()
        if r.status != 200:
            raise RuntimeError("HTTP %d %r" % (r.status, body[:200]))
        d = json.loads(body)
        ni = float(d.get("true_net_income") or 0.0)
        out = {
            "reward": ni if d.get("final") else 0.0,
            "true_net_income": ni,
            "final": int(bool(d.get("final"))),
            "bankrupt": int(bool(d.get("bankrupt"))),
        }
        break
    except Exception as e:  # server busy / restarting
        print("audit fetch failed:", e)
        time.sleep(5)
json.dump(out, open("/logs/verifier/reward.json", "w"))
print(json.dumps(out))
EOF
"""

SOLVE_SH = """\
#!/bin/bash
# Scripted oracle ($0): isolation checks, then each turn view the books and
# end the day. Exercises the full CLI/server/verifier path without an LLM.
# Workspace persistence check: a resumed session must find the note written
# in the previous one.
set -u
# Credential / network / server-allowlist isolation (prints *_OK|LEAK lines).
python3 "$(dirname "$0")/isolation_check.py"
out=$(coffee start) || exit 1
echo "$out" | head -c 1500; echo
if echo "$out" | grep -q -- "--- resume digest ---"; then
  if [ -f /workspace/note.txt ]; then echo "WORKSPACE_NOTE_OK: $(cat /workspace/note.txt)"
  else echo "WORKSPACE_NOTE_MISSING"; fi
fi
echo "note from $COFFEE_AGENT_ID, session started $(echo "$out" | head -1 | cut -c1-80)" >> /workspace/note.txt
for i in $(seq 1 5000); do
  echo "$out" | grep -q "SIMULATION OVER" && break
  out=$(coffee view_trial_balance) || exit 1
  echo "$out" | head -c 300; echo
  echo "$out" | grep -q "SIMULATION OVER" && break
  out=$(coffee wait_for_next_day) || exit 1
  echo "$out" | head -1
done
echo "oracle loop done"
"""

ISOLATION_CHECK = '''\
"""Oracle-run isolation checks inside the task container (prints one line each).

CREDENTIALS_OK   no other agent's / harness / verifier token in the env
NET_IFACES       network interfaces (network none: `lo` only)
NET_CURL_*       `curl https://example.com` fails (BLOCKED) — network none/relay
NET_PY_*         python urllib to https://pypi.org fails (BLOCKED)
SERVER_ALLOWLIST everything but the CLI's endpoints → uniform 404, no server header
COFFEE_CLI_OK    `coffee status` works over $COFFEE_SOCKET / $COFFEE_URL
"""
import http.client
import json
import os
import socket
import subprocess
import urllib.parse
import urllib.request

leaked = [k for k in os.environ
          if k.startswith(("COFFEE_TOKEN_", "COFFEE_HARNESS", "COFFEE_VERIFIER"))]
print("CREDENTIALS_OK" if not leaked else "CREDENTIALS_LEAK %s" % leaked)

try:
    ifaces = sorted(os.listdir("/sys/class/net"))
except OSError:
    ifaces = ["?"]
print("NET_IFACES %s" % ",".join(ifaces))  # network_mode none -> lo only
r = subprocess.run(["curl", "-sS", "-m", "8", "-o", "/dev/null", "https://example.com"],
                   capture_output=True, text=True)
print("NET_CURL_BLOCKED" if r.returncode else "NET_CURL_REACHABLE")
try:
    urllib.request.urlopen("https://pypi.org", timeout=8)
    print("NET_PY_REACHABLE")
except Exception as e:  # noqa: BLE001
    print("NET_PY_BLOCKED (%s)" % type(e).__name__)


class UnixConn(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(10)
        self.sock.connect(os.environ["COFFEE_SOCKET"])


def conn():
    if os.environ.get("COFFEE_SOCKET"):
        return UnixConn("coffee", timeout=10)
    u = urllib.parse.urlsplit(os.environ["COFFEE_URL"])
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)


res = {}
aid = os.environ["COFFEE_AGENT_ID"]
probes = [("GET", "/"), ("GET", "/openapi.json"), ("GET", "/docs"), ("GET", "/health"),
          ("GET", "/audit/" + aid), ("POST", "/telemetry"), ("GET", "/resume_info"),
          ("POST", "/resume_ack"), ("POST", "/admin/stop_at_next_morning"), ("DELETE", "/act")]
for method, path in probes:
    c = conn()
    c.request(method, path, body=b"{}" if method != "GET" else None,
              headers={"Authorization": "Bearer " + os.environ["COFFEE_TOKEN"],
                       "Content-Type": "application/json"})
    r = c.getresponse()
    body = r.read()
    res[method + " " + path] = (r.status, body == b'{"error":"not found"}', r.getheader("server"))
    c.close()
ok = all(st == 404 and same and srv is None for st, same, srv in res.values())
print("SERVER_ALLOWLIST_OK" if ok else "SERVER_ALLOWLIST_LEAK %s" % json.dumps(res))
r = subprocess.run(["coffee", "status"], capture_output=True, text=True)
try:
    st = json.loads(r.stdout)
    print("COFFEE_CLI_OK %s" % st.get("sim_state") if "sim_state" in st else "COFFEE_CLI_FAIL %s" % st)
except ValueError:
    print("COFFEE_CLI_FAIL %r" % r.stdout[:300])
'''


def token_var(aid: str) -> str:
    """Host env var holding `aid`'s own token (a container never sees another's)."""
    return f"COFFEE_TOKEN_{aid.upper()}"


def verifier_token_var(aid: str) -> str:
    """Host env var holding `aid`'s verifier-scope token (tests/test.sh only)."""
    return f"COFFEE_VERIFIER_TOKEN_{aid.upper()}"


def transport_env(network: str) -> str:
    """task.toml lines telling the CLI / verifier how to reach the server."""
    if network == "bridge":
        return 'COFFEE_URL = "${COFFEE_URL}"\n'
    return f'COFFEE_SOCKET = "{SOCKET_PATH}"\n'


def role_tools(role: str, mech_spec: dict | None = None) -> list:
    names = tool_names_for(role, mech_spec)
    return [getattr(BusinessApp, n) for n in names if callable(getattr(BusinessApp, n, None))]


AGENT_IDS = ("farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B")


def opponent_seat(config: RunConfig, seat: str | None = None) -> str:
    """Head-to-head opponent seat: `seat` if given, else the focal role's other
    seat (main_agent roaster_A → roaster_B)."""
    if seat:
        return seat
    role, letter = (config.main_agent or "roaster_A").split("_")
    return f"{role}_{'B' if letter == 'A' else 'A'}"


def seat_personas(config: RunConfig, bg: str | None, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    """{aid: profile} of the persona-LLM seats for background roster `bg`
    (`Persona:<model>`): every non-focal seat, profile from its [models] entry
    (`heuristic_<role>@<profile>`, else default). {} for other rosters.
    `exclude`: further evaluated seats without a persona (head-to-head
    opponent seat, launch_sim --focal2)."""
    if not bg or not bg.startswith("Persona"):
        return {}
    from coffeebench.personas import seat_profile

    focal = config.main_agent or "roaster_A"
    return {aid: seat_profile(config.models.get(aid), aid.split("_")[0])
            for aid in AGENT_IDS if aid != focal and aid not in exclude}


def dataset_name(config_name: str, network: str, max_days: int | None,
                 personas: dict[str, str] | None = None, persona_style: str = "strict") -> str:
    """Dataset dir name. Non-default settings (smoke-test horizons, relay /
    bridge networking, persona blocks) get their own suffix so a stale smoke
    dataset can never be picked up by a real run."""
    import hashlib

    name = config_name
    if max_days:
        name += f"_d{max_days}"
    if network != "none":
        name += f"_{network}"
    if personas:
        key = ",".join(f"{a}={p}" for a, p in sorted(personas.items()))
        if persona_style != "strict":
            key += f"|{persona_style}"
        name += "_persona" + hashlib.sha1(key.encode()).hexdigest()[:8]
    return name


def generate(config_path: Path, out_root: Path, network: str, max_days: int | None,
             bg: str | None = None, exclude: tuple[str, ...] = ()) -> Path:
    if network not in NETWORKS:
        raise ValueError(f"network must be one of {NETWORKS}")
    config = RunConfig.from_toml(config_path)
    days = int(max_days or config.max_days)
    prompts = build_system_prompts(config, max_days=days)
    personas = seat_personas(config, bg, exclude)
    persona_style = str(config.background.get("persona_style", "strict"))
    ds = out_root / dataset_name(config.name, network, max_days, personas, persona_style)
    if ds.exists():
        shutil.rmtree(ds)
    compose = {"none": COMPOSE_NONE, "relay": COMPOSE_RELAY, "bridge": COMPOSE_BRIDGE}[network]
    for aid, info in prompts.items():
        t = ds / aid
        (t / "environment").mkdir(parents=True)
        (t / "tests").mkdir()
        (t / "solution").mkdir()
        catalog = tool_catalog(role_tools(info["role"], mech_mod.parse(config.mechanics)))
        text = build_instruction(aid, info["prompt"], catalog, days)
        if aid in personas and persona_style != "none":  # private: this seat's instruction only
            from coffeebench.personas import persona_block

            text += persona_block(info["role"], personas[aid], persona_style)
        (t / "instruction.md").write_text(text)
        (t / "task.toml").write_text(
            TASK_TOML.format(cfg=config.name, aid=aid, role=info["role"], max_days=days,
                             network=network, token_var=token_var(aid),
                             verifier_var=verifier_token_var(aid),
                             transport=transport_env(network))
        )
        (t / "environment" / "Dockerfile").write_text(DOCKERFILE)
        shutil.copy2(ROOT / "coffeebench" / "cli" / "coffee.py", t / "environment" / "coffee.py")
        (t / "environment" / "docker-compose.yaml").write_text(compose)
        (t / "solution" / "isolation_check.py").write_text(ISOLATION_CHECK)
        for rel, body in (("tests/test.sh", TEST_SH), ("solution/solve.sh", SOLVE_SH)):
            (t / rel).write_text(body)
            (t / rel).chmod(0o755)
    print(f"[generate_tasks] {len(prompts)} tasks → {ds} (network={network}, max_days={days})")
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True, help="Experiment TOML (repeatable).")
    ap.add_argument("--out", type=Path, default=ROOT / "harbor_adapter" / "datasets")
    ap.add_argument("--network", choices=NETWORKS, default="none",
                    help="none = default for Linux rootless podman (Unix socket, no network); relay = Docker Desktop "
                    "test of none; bridge = local TCP smoke fallback (internet reachable).")
    ap.add_argument("--max-days", type=int, default=None, help="Override (smoke tests).")
    ap.add_argument("--bg", default=None,
                    help="Background roster; Persona:<model> renders the seats' private persona blocks.")
    ap.add_argument("--h2h", action="store_true",
                    help="Head-to-head run (launch_sim --focal2): the opponent seat gets no persona block.")
    ap.add_argument("--opponent-seat", default=None, help="With --h2h (default: the focal role's other seat).")
    ap.add_argument("--print-name", action="store_true",
                    help="Only print the dataset dir name (for cluster/pbs_job.sh) and exit.")
    args = ap.parse_args()
    for c in args.config:
        cfg = RunConfig.from_toml(c)
        exclude = (opponent_seat(cfg, args.opponent_seat),) if args.h2h else ()
        if args.print_name:
            print(dataset_name(cfg.name, args.network, args.max_days, seat_personas(cfg, args.bg, exclude),
                               str(cfg.background.get("persona_style", "strict"))))
            continue
        generate(Path(c), args.out, args.network, args.max_days, args.bg, exclude)


if __name__ == "__main__":
    main()
