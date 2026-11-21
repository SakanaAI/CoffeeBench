"""$0 local smoke tests for the Harbor harness (no Harbor, no LLM, no docker).

    unset VIRTUAL_ENV; uv run python harbor_adapter/smoke_local.py \
        [--only equiv,deadline,segment,ckpt,digest,resume,auth,walltime]

equiv    in-process run (roaster_A = heuristic_roaster, rest passive) vs
         (1) hybrid server: roaster_A over the real `coffee` CLI driven by the same
             heuristic policy, the 5 passive agents in-process;
         (2) full CLI: all 6 agents over the CLI (5 bash `coffee wait_for_next_day` loops).
         → every agent's true_net_income must match exactly.
deadline server with a 3 s wake deadline; farmer_B stalls 6 s before its first
         action → `turn_timeout` event, its late action is NOT executed, sim completes.
segment  5-day config with --segment-days 2 → every CLI gets segment_end at day 2,
         server writes segment_002.json and exits.
ckpt     in-process 10-day sim (heuristic roaster_A, scripted `TraderModel` farmers /
         retailers / roaster_B that trade, pay, and DM each other): uninterrupted vs
         stop at day 5 → resume in a NEW process → run.json and run.events.jsonl
         identical after dropping wall-clock fields (ts_ms, timestamps, saved_at).
digest   resume digest from the day-5 checkpoint: has lots / deals / DM threads with
         ids, ≤ MAX_CHARS, deterministic, trimmed with omission counts under a
         small bound.
resume   the same 10-day sim through the server, all 6 agents over the CLI:
         uninterrupted vs --segment-days 5 (2 server processes; drivers keep
         their own policy state like an agent's memory) → identical run.json /
         events; every agent gets a digest with its first segment-2 payload.
         Workspace: each agent's note.txt (written in segment 1, its local dir
         wiped before segment 2 = fresh container) is restored in segment 2;
         farmer_A's 0.1 MB cap drops its out/ dump but keeps the note.
auth     scoped tokens + strict allowlist: farmer_A's agent token acting / waiting /
         tools as roaster_B → 403; with it, every non-CLI path (/, /docs,
         /openapi.json, /health, /audit/*, /telemetry, /resume_*, /admin/*,
         unknown paths/methods) → the same 404 {"error":"not found"}; wrong or
         missing token → 404; no `server` header; own /status minimal; harness
         token: telemetry/resume_info only for itself; verifier token: own /audit
         only; admin token: /status (all agents) + /admin, not /register;
         malformed / oversized bodies → 400 / 413; flood → 429; the same over
         the Unix socket (CLI with COFFEE_SOCKET).
Transport: every server also listens on a Unix socket (/tmp/cbsm_<port>.sock);
the `equiv` full-CLI bash loops and all `resume` / `walltime` drivers use it
(COFFEE_SOCKET), the rest TCP.
v2       hybrid server run with roaster_A = heuristic_roaster_v2 in-process
         (as `launch_sim.py --local roaster_A:heuristic_roaster_v2`), farmer_A +
         retailer_A TraderModel over the CLI, rest passive, 8 days → v2 roasts and
         trades, ≤ 20 actions/day. Not in the default --only list.
walltime the 10-day sim with launch_sim.StopWatch on a fake walltime: it calls
         /admin/stop_at_next_morning mid-run → segment_end (stop_reason
         walltime) → resume → identical run.json / events vs uninterrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "coffeebench" / "cli" / "coffee.py"
AGENTS = ["farmer_A", "farmer_B", "roaster_A", "roaster_B", "retailer_A", "retailer_B"]
TOKEN = "smoke-token"
OUT_ROOT = "trajectories/harbor_smoke"

CONFIG = """\
[experiment]
name = "{name}"
description = "Harbor smoke test (scripted agents only, $0)."

[run]
seeds = [0]
max_days = {max_days}
main_agent = "roaster_A"

[models]
default = "passive"
roaster_A = "heuristic_roaster"
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_config(tmp: Path, name: str, max_days: int) -> Path:
    p = tmp / f"{name}.toml"
    p.write_text(CONFIG.format(name=name, max_days=max_days))
    return p


def agent_tok(agent: str) -> str:
    from coffeebench.server.app import agent_token

    return agent_token(TOKEN, agent)


def scoped_tok(scope: str, agent: str) -> str:
    from coffeebench.server.app import scoped_token

    return scoped_token(TOKEN, scope, agent)


def sock_path(port: int) -> str:
    return f"/tmp/cbsm_{port}.sock"


def cli_env(port: int, agent: str, tmp: Path, uds: bool = False) -> dict:
    env = dict(os.environ)
    env.pop("COFFEE_SOCKET", None)
    if uds:  # Unix socket only: a bogus COFFEE_URL proves TCP is not used
        env.update(COFFEE_SOCKET=sock_path(port), COFFEE_URL="http://127.0.0.1:9")
    else:
        env["COFFEE_URL"] = f"http://127.0.0.1:{port}"
    env.update(
        COFFEE_TOKEN=agent_tok(agent),
        COFFEE_AGENT_ID=agent,
        COFFEE_WORKSPACE=str(tmp / "ws" / agent),
        COFFEE_OUT_DIR=str(tmp / "ws" / agent / "out"),
        TMPDIR=str(tmp),
    )
    env["PATH"] = f"{tmp / 'bin'}:{env['PATH']}"
    return env


def install_cli(tmp: Path) -> None:
    (tmp / "bin").mkdir(exist_ok=True)
    link = tmp / "bin" / "coffee"
    if not link.exists():
        link.symlink_to(CLI)


def parse_cli(out: str) -> dict:
    """Inverse of coffee.py's printer: JSON head line + digest + observations."""
    lines = out.splitlines()
    head = json.loads(lines[0])
    obs, cur, digest = [], None, None
    for ln in lines[1:]:
        if ln == "--- resume digest ---":
            digest = []
            continue
        if digest is not None and cur is None and ln != "--- observation ---":
            digest.append(ln)
            continue
        if ln == "--- observation ---":
            if cur is not None:
                obs.append("\n".join(cur))
            cur = []
        elif ln.startswith("SIMULATION OVER") or ln.startswith("(full output also saved"):
            continue
        elif cur is not None:
            cur.append(ln)
    if cur is not None:
        obs.append("\n".join(cur))
    head["observations"] = obs
    if digest is not None:
        head["digest"] = "\n".join(digest)
    return head


def run_cli(env: dict, *args: str) -> dict:
    r = subprocess.run(["coffee", *args], env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"coffee {args} failed rc={r.returncode}: {r.stdout} {r.stderr}")
    return parse_cli(r.stdout)


def heuristic_driver(env: dict, log: list) -> None:
    """HeuristicRoasterModel fed exactly what it would see in-process: the
    last tool result followed by any wake observations."""
    from coffeebench.models.heuristic_roaster_model import HeuristicRoasterModel

    model = HeuristicRoasterModel()
    resp = run_cli(env, "start")
    while True:
        log.append(resp["status"])
        if resp["status"] != "ok":
            return
        msgs = []
        if "result" in resp:
            msgs.append(
                {"role": "tool", "name": resp.get("tool"), "content": json.dumps(resp["result"], default=str)}
            )
        msgs += [{"role": "user", "content": o} for o in resp["observations"]]
        call = model.query(msgs).tool_calls[0]
        resp = run_cli(env, call.name, json.dumps(call.input))


def bash_loop(env: dict, stall_first: float = 0.0) -> subprocess.Popen:
    script = (
        "coffee start >/dev/null || exit 1\n"
        + (f"sleep {stall_first}\n" if stall_first else "")
        + "for i in $(seq 1 400); do\n"
        "  out=$(coffee wait_for_next_day) || exit 1\n"
        "  echo \"$out\" | grep -q 'SIMULATION OVER' && { echo \"$out\" | head -1; exit 0; }\n"
        "done\nexit 1\n"
    )
    return subprocess.Popen(["bash", "-c", script], env=env, stdout=subprocess.PIPE, text=True)


def start_server(cfg: Path, port: int, *extra: str) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "coffeebench.server", "--config", str(cfg), "--seed", "0",
        "--port", str(port), "--run-token", TOKEN, "--output-root", OUT_ROOT,
        "--linger-sec", "5", "--uds", sock_path(port), *extra,
    ]
    log = open(Path(tempfile.gettempdir()) / f"cb_server_{port}.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return proc
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"server died; see {log.name}")
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def fetch_audit(port: int, agent: str) -> None:
    """Verifier stand-in: GET /audit/<agent> with the agent's verifier token."""
    http(port, scoped_tok("verifier", agent), "GET", f"/audit/{agent}")


def http_raw(port: int, token: str | None, method: str, path: str, body: bytes | None = None,
             uds: bool = False, headers: dict | None = None, pace: float = 0.03) -> tuple[int, bytes, dict]:
    """(status, raw body, lower-cased headers) over TCP or the Unix socket.
    `pace` keeps test traffic below the server's per-principal rate limit."""
    import http.client

    time.sleep(pace)
    if uds:
        class Conn(http.client.HTTPConnection):
            def connect(self):
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(30)
                self.sock.connect(sock_path(port))
        c = Conn("coffee", timeout=30)
    else:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    h = {"Content-Type": "application/json", **(headers or {})}
    if token is not None:
        h["Authorization"] = f"Bearer {token}"
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    out = (r.status, data, {k.lower(): v for k, v in r.getheaders()})
    c.close()
    return out


def http(port: int, token: str | None, method: str, path: str, body: dict | None = None,
         uds: bool = False) -> tuple[int, dict]:
    st, data, _ = http_raw(port, token, method, path,
                           json.dumps(body).encode() if body is not None else None, uds=uds)
    try:
        return st, json.loads(data.decode() or "{}")
    except ValueError:
        return st, {"raw": data.decode(errors="replace")}


def ni_of(run_json: Path) -> dict:
    data = json.loads(run_json.read_text())
    return {
        aid: a["audit"]["annual"]["true_net_income"]
        for aid, a in data["result"]["agents"].items()
    }


def events(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def fingerprint(run_dir: Path, agent: str = "roaster_A") -> tuple:
    """(agent's step sequence, every agent's final books) — must be identical
    between the in-process and CLI-driven runs."""
    steps = [
        (e["day"], e["at"], e["action"], json.dumps(e["action_input"], sort_keys=True),
         json.dumps(e["observation"], sort_keys=True, default=str))
        for e in events(run_dir / "run.events.jsonl")
        if e["type"] == "agent_step" and e["agent_id"] == agent
    ]
    books = json.loads((run_dir / "run.json").read_text())["agent_books"]
    books = {a: (round(b["cash"], 6), json.dumps(b["inventory"], sort_keys=True)) for a, b in books.items()}
    # Object ids are uuid4-based (unseeded) → renumber by first appearance.
    ids: dict[str, str] = {}
    norm = re.compile(r"\b([a-z]+)_[0-9a-f]{8}\b")
    steps = [
        tuple(norm.sub(lambda m: ids.setdefault(m.group(0), f"{m.group(1)}#{len(ids)}"), str(x)) for x in st)
        for st in steps
    ]
    return steps, books


def test_equiv(tmp: Path) -> bool:
    cfg = write_config(tmp, "hbsmoke_equiv", 5)
    subprocess.run(
        [sys.executable, "-m", "coffeebench.main", "--config", str(cfg), "--seed", "0"],
        cwd=ROOT, check=True, capture_output=True,
    )
    ref = ni_of(ROOT / "trajectories/hbsmoke_equiv/seed_0/run.json")
    ref_fp = fingerprint(ROOT / "trajectories/hbsmoke_equiv/seed_0")
    print("  in-process NI:", ref)
    ok = True
    for mode in ("hybrid", "all"):
        port = free_port()
        server = start_server(cfg, port, "--remote", "roaster_A" if mode == "hybrid" else "all")
        loops = []
        if mode == "all":
            # bash loops over the Unix socket, roaster_A over TCP (both listeners at once)
            loops = [bash_loop(cli_env(port, a, tmp, uds=True)) for a in AGENTS if a != "roaster_A"]
        log: list = []
        heuristic_driver(cli_env(port, "roaster_A", tmp), log)
        for p in loops:
            p.wait(timeout=600)
        for a in AGENTS:  # verifier stand-in: fetch /audit so the server exits
            if mode == "all" or a == "roaster_A":
                fetch_audit(port, a)
        server.wait(timeout=120)
        got = ni_of(ROOT / OUT_ROOT / "hbsmoke_equiv/seed_0/run.json")
        fp = fingerprint(ROOT / OUT_ROOT / "hbsmoke_equiv/seed_0")
        same = got == ref and fp == ref_fp
        ok &= same and all(p.returncode == 0 for p in loops) and server.returncode == 0
        print(f"  {mode:6s} NI: {got}  match={same}  roaster steps={len(fp[0])} last={log[-1]}")
    return ok


def test_deadline(tmp: Path) -> bool:
    cfg = write_config(tmp, "hbsmoke_deadline", 2)
    port = free_port()
    server = start_server(cfg, port, "--remote", "all", "--deadline-sec", "3")
    loops = {a: bash_loop(cli_env(port, a, tmp), stall_first=6 if a == "farmer_B" else 0)
             for a in AGENTS if a != "roaster_A"}
    log: list = []
    heuristic_driver(cli_env(port, "roaster_A", tmp), log)
    for p in loops.values():
        p.wait(timeout=600)
    for a in AGENTS:
        fetch_audit(port, a)
    server.wait(timeout=120)
    run_dir = ROOT / OUT_ROOT / "hbsmoke_deadline/seed_0"
    ev = events(run_dir / "run.events.jsonl")
    timeouts = [e["agent_id"] for e in ev if e["type"] == "turn_timeout"]
    stats = json.loads((run_dir / "harbor_server.json").read_text())
    steps_b = [e for e in ev if e["type"] == "agent_step" and e["agent_id"] == "farmer_B"]
    first_b = steps_b[0]["observation"] if steps_b else None
    ok = timeouts == ["farmer_B"] and server.returncode == 0 and loops["farmer_B"].returncode == 0
    ok &= first_b is not None and first_b.get("status") == "turn_timeout"
    print(f"  turn_timeout events: {timeouts}; farmer_B first step obs: {first_b}")
    print(f"  server stats timeouts: { {a: s['timeouts'] for a, s in stats['agents'].items()} }")
    return ok


def test_segment(tmp: Path) -> bool:
    cfg = write_config(tmp, "hbsmoke_segment", 5)
    port = free_port()
    server = start_server(cfg, port, "--remote", "all", "--segment-days", "2")
    loops = {a: bash_loop(cli_env(port, a, tmp)) for a in AGENTS if a != "roaster_A"}
    log: list = []
    heuristic_driver(cli_env(port, "roaster_A", tmp), log)
    outs = {a: p.communicate(timeout=600)[0].strip() for a, p in loops.items()}
    for a in AGENTS:
        fetch_audit(port, a)
    server.wait(timeout=120)
    run_dir = ROOT / OUT_ROOT / "hbsmoke_segment/seed_0"
    seg = run_dir / "segment_002.json"
    statuses = {a: json.loads(o.splitlines()[0])["status"] for a, o in outs.items()}
    statuses["roaster_A"] = log[-1]
    ev = events(run_dir / "run.events.jsonl")
    last_day = max(e.get("day", -1) for e in ev if e["type"] == "agent_step")
    ok = seg.exists() and set(statuses.values()) == {"segment_end"} and server.returncode == 0
    ok &= last_day == 1
    print(f"  statuses: {statuses}; last agent_step day={last_day}; {seg.name} exists={seg.exists()}")
    return ok

# ---------------------------------------------------------------------------
# Checkpoint / resume tests
# ---------------------------------------------------------------------------

TRADERS = ("farmer_A", "roaster_B", "retailer_A", "retailer_B")
DM_PARTNER = {"farmer_A": "roaster_A", "roaster_B": "farmer_A", "retailer_A": "roaster_A",
              "retailer_B": "roaster_B"}


class TraderModel:
    """Deterministic scripted policy ($0) that exercises deals, invoices and
    DMs so checkpoint/resume equality is tested on a non-trivial state.
    Daily: view_dashboard → accept incoming offers → pay due AP → read +
    reply to the oldest unread DM → role action (farmer: list green;
    roaster_B: list roasted + bid on green; retailer: bid on roasted) → DM
    its partner on even days → wait_for_next_day. Reads every message after
    the last assistant turn, so the in-process transcript and the CLI payload
    (tool result + observations) drive it identically."""

    checkpointable = True  # coffeebench.server.checkpoint: state = vars(model)

    def __init__(self, agent_id: str):
        self.model = "trader/scripted"
        self.agent_id = agent_id
        self.role = agent_id.split("_")[0]
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = self.total_output_tokens = self.last_input_tokens = 0
        self.system_prompt = ""
        self.day = -1
        self.plan: list[list] = []
        self.seq = 0

    def get_usage_stats(self) -> dict:
        return {"model": self.model, "n_calls": self.n_calls, "cost": 0.0}

    def query(self, messages, tools=None):  # noqa: ARG002
        from coffeebench.models.heuristic_roaster_model import _extract_day
        from coffeebench.models.types import ModelResponse, ToolCall

        self.n_calls += 1
        tail = []
        for m in reversed(messages):
            if m.get("role") == "assistant":
                break
            tail.append(m)
        for m in reversed(tail):
            if m.get("role") == "user":
                d = _extract_day(m.get("content") or "")
                if d is not None and d != self.day:
                    self.day, self.plan = d, [["view_dashboard", {}]]
            elif m.get("role") == "tool":
                try:
                    data = json.loads(m.get("content") or "{}")
                except ValueError:
                    data = {}
                self._ingest(m.get("name"), data if isinstance(data, dict) else {})
        tool, args = self.plan.pop(0) if self.plan else ["wait_for_next_day", {}]
        self.seq += 1
        return ModelResponse(content="(trader)", thinking="", stop_reason="tool_use", cost=0.0, raw=None,
                             tool_calls=[ToolCall(id=f"tr_{self.seq:04d}", name=tool, input=args)])

    def _ingest(self, name: str | None, data: dict) -> None:
        if name == "view_dashboard" and data.get("status") == "success":
            plan = [["accept_offer", {"offer_id": o["id"]}]
                    for o in data["offers_on_my_listings"]["rows"]]
            plan += [["pay_invoice", {"invoice_id": r["id"]}] for r in data["payables_due"]["rows"]]
            unread = [m for m in data["messages"]["rows"] if m.get("unread")]
            if unread:
                m = unread[-1]  # rows are newest first
                plan += [["read_message", {"message_id": m["id"]}],
                         ["send_message", {"recipient": m["from"], "title": f"Re: {m['title']}"[:60],
                                           "body": f"Day {self.day}: noted, {self.agent_id} confirms.",
                                           "reply_to_message_id": m["id"]}]]
            inv = {r["item_id"]: r["qty"] for r in data["inventory"]["by_item"]}
            has_listing = bool(data["my_open_listings"]["rows"])
            if self.role == "farmer" and not has_listing and inv.get("green_coffee_kg", 0) >= 10:
                plan.append(["post_listing", {"item_id": "green_coffee_kg", "qty": min(20, inv["green_coffee_kg"]),
                                              "asking_price": 2.6, "payment_terms_days": 7}])
            if self.role == "roaster":
                if not has_listing and inv.get("roasted_coffee_kg", 0) >= 10:
                    plan.append(["post_listing", {"item_id": "roasted_coffee_kg", "qty": 10,
                                                  "asking_price": 7.5, "payment_terms_days": 7}])
                plan.append(["view_listings", {"item_id": "green_coffee_kg"}])
            if self.role == "retailer":
                plan.append(["view_listings", {"item_id": "roasted_coffee_kg"}])
            if self.day % 2 == 0:
                plan.append(["send_message", {
                    "recipient": DM_PARTNER[self.agent_id], "title": f"Day {self.day} plan ({self.agent_id})",
                    "body": f"Exclusive supply promise #{self.day}: I will trade 10 kg with you "
                            f"on day {self.day + 2} at the listed price."}])
            self.plan = plan + self.plan
        elif name == "view_listings" and data.get("status") == "success":
            rows = sorted((r for r in data.get("listings") or []
                           if r.get("seller_id") != self.agent_id and r.get("qty", 0) > 0),
                          key=lambda r: (float(r["asking_price"]), r["id"]))
            if rows:
                top = rows[0]
                self.plan.insert(0, ["make_offer", {
                    "listing_id": top["id"], "offered_price": float(top["asking_price"]),
                    "qty": min(5 if self.role == "retailer" else 10, int(top["qty"])),
                    "payment_terms_days": 7}])


TRADE_CONFIG = """\
[experiment]
name = "{name}"
description = "Checkpoint/resume smoke test (scripted agents only, $0)."

[run]
seeds = [0]
max_days = {max_days}
main_agent = "roaster_A"

[models]
default = "passive"
roaster_A = "heuristic_roaster"
"""


def child_sim(argv: list[str]) -> None:
    """`smoke_local.py --child-sim CFG [--ckpt-dir D] [--stop-day N] [--resume P]`:
    one in-process run (a fresh process per call) with TraderModel agents."""
    import asyncio

    from coffeebench.agent import Agent
    from coffeebench.main import INITIAL_OBSERVATION, build_run
    from coffeebench.server import checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument("--child-sim", required=True)
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--stop-day", type=int)
    ap.add_argument("--resume")
    a = ap.parse_args(argv)

    def factory(aid, model_id, tools, prompt):
        if aid in TRADERS:
            return Agent(model=TraderModel(aid), tools=tools, instruct_prompt=prompt,
                         initial_observation=INITIAL_OBSERVATION, name=aid)
        return None

    run_args = argparse.Namespace(config=a.child_sim, model=None, models=None, max_days=None, seed=0,
                                  main_agent=None, output=None, output_root="trajectories")
    env, out, events = build_run(run_args, agent_factory=factory, open_events=not a.resume)
    env.verbose = False
    if a.resume:
        env = checkpoint.restore(env, checkpoint.load(a.resume), events)
    if a.ckpt_dir or a.stop_day is not None:
        env.checkpoint_hook = checkpoint.make_hook(a.ckpt_dir, a.stop_day)
    asyncio.run(env.run(resume=bool(a.resume)))
    if env.stopped_at_day is None:
        env.save_trajectory(out)
    env.event_logger.close()


WALL_CLOCK = re.compile(r'"timestamp": "\d{8}_\d{6}"')


def normalized(run_dir: Path) -> tuple[str, list[str]]:
    """run.json + events without wall-clock fields (ts_ms, timestamps, saved_at)."""
    data = json.loads((run_dir / "run.json").read_text())
    data.pop("saved_at", None)
    run = WALL_CLOCK.sub('"timestamp": "T"', json.dumps(data, sort_keys=True, indent=0))
    evs = []
    for e in events(run_dir / "run.events.jsonl"):
        e.pop("ts_ms", None)
        evs.append(json.dumps(e, sort_keys=True))
    return run, evs


def compare(ref: tuple, got: tuple, label: str) -> bool:
    same_run, same_ev = ref[0] == got[0], ref[1] == got[1]
    print(f"  {label}: run.json identical={same_run}; events identical={same_ev} "
          f"({len(ref[1])} vs {len(got[1])} lines)")
    for i, (x, y) in enumerate(zip(ref[1], got[1])):
        if x != y:
            print(f"    first event diff @{i}:\n    {x[:300]}\n    {y[:300]}")
            break
    if not same_run:
        a, b = ref[0].splitlines(), got[0].splitlines()
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"    first run.json diff @line {i}: {x[:200]!r} vs {y[:200]!r}")
                break
    return same_run and same_ev


def run_child(*args: str) -> None:
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], cwd=ROOT, check=True,
                   capture_output=True)


def snapshot_dir(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("run.json", "run.events.jsonl"):
        if (src / f).exists():
            (dst / f).write_bytes((src / f).read_bytes())
    return dst


def test_ckpt(tmp: Path) -> bool:
    cfg = tmp / "hbsmoke_ckpt.toml"
    cfg.write_text(TRADE_CONFIG.format(name="hbsmoke_ckpt", max_days=10))
    run_dir = ROOT / "trajectories/hbsmoke_ckpt/seed_0"
    run_child("--child-sim", str(cfg))
    full = snapshot_dir(run_dir, tmp / "ckpt_full")
    ref = normalized(full)
    ckpts = tmp / "ckpt_pkls"
    run_child("--child-sim", str(cfg), "--ckpt-dir", str(ckpts), "--stop-day", "5")
    stopped = not (run_dir / "run.json").exists() or (run_dir / "run.json").stat().st_mtime < (
        ckpts / "day_005.pkl").stat().st_mtime
    run_child("--child-sim", str(cfg), "--resume", str(ckpts / "day_005.pkl"))
    got = normalized(run_dir)
    ok = compare(ref, got, "uninterrupted vs stop@5+resume") and stopped
    ni_ref, ni_got = ni_of(full / "run.json"), ni_of(run_dir / "run.json")
    summary = json.loads((run_dir / "run.json").read_text())["result"]["marketplace_summary"]
    print(f"  NI identical={ni_ref == ni_got}: {ni_got}")
    print(f"  marketplace: {summary}; checkpoints: {sorted(p.name for p in ckpts.glob('*.pkl'))}")
    ok &= ni_ref == ni_got and summary["total_deals"] > 0 and summary["total_messages"] > 0
    return ok


def test_digest(tmp: Path) -> bool:
    from coffeebench.server import checkpoint
    from coffeebench.server.digest import MAX_CHARS, build_digests

    ckpt_path = tmp / "ckpt_pkls" / "day_005.pkl"
    if not ckpt_path.exists():
        test_ckpt(tmp)
    ck = checkpoint.load(str(ckpt_path))
    env = ck["env"]
    events_file = tmp / "ckpt_digest_events.jsonl"
    full_events = (tmp / "ckpt_full" / "run.events.jsonl").read_bytes()
    events_file.write_bytes(full_events[: ck["events_offset"]])
    d1 = build_digests(env, AGENTS, str(events_file))
    d2 = build_digests(env, AGENTS, str(events_file))
    small = build_digests(env, AGENTS, str(events_file), max_chars=6000)
    ok = d1 == d2
    for aid in AGENTS:
        txt = d1[aid]
        checks = {
            "size": len(txt) <= MAX_CHARS,
            "dm": "## DM threads" in txt and ("[msg_" in txt or "no messages yet" in txt),
            "lots": "lot_" in txt or "(no stock)" in txt,
            "actions": "## Your actions since day" in txt and " → " in txt,
            "workspace": "/workspace" in txt,
            "small": len(small[aid]) <= 6000,
        }
        ok &= all(checks.values())
        print(f"  {aid}: {len(txt)} chars (small bound: {len(small[aid])}); "
              f"{'ok' if all(checks.values()) else checks}")
    ok &= any("dl_" in t for t in d1.values()) and any("omitted" in t for t in small.values())
    # DM bodies (commitments) are carried verbatim, with thread ids.
    ok &= "Exclusive supply promise" in d1["roaster_A"] and "thread msg_" in d1["roaster_A"]
    print("  --- retailer_A digest excerpt ---")
    print("\n".join("  | " + ln for ln in d1["retailer_A"].splitlines()[:40]))
    return ok


def policy_driver(env: dict, make_model, state_file: Path | None, log: list, digests: list) -> None:
    """Drive one agent over the CLI with a scripted policy. Its state is
    pickled at a segment end and reloaded on the next start (an agent with
    perfect memory) so segmented and uninterrupted runs act identically."""
    if state_file is not None and state_file.exists():
        model = pickle.loads(state_file.read_bytes())
    else:
        model = make_model()
    resp = run_cli(env, "start")
    digests.append(resp.get("digest"))
    while True:
        log.append(resp["status"])
        if resp["status"] != "ok":
            if state_file is not None:
                state_file.write_bytes(pickle.dumps(model))
            return
        msgs = []
        if "result" in resp:
            msgs.append({"role": "tool", "name": resp.get("tool"),
                         "content": json.dumps(resp["result"], default=str)})
        msgs += [{"role": "user", "content": o} for o in resp["observations"]]
        calls = model.query(msgs).tool_calls
        name, args = (calls[0].name, calls[0].input) if calls else ("wait_for_next_day", {})
        resp = run_cli(env, name, json.dumps(args))


def server_session(cfg: Path, tmp: Path, state_dir: Path, *extra: str, env_extra: dict | None = None,
                   watch=None) -> tuple[dict, dict, int]:
    """One server process + 6 CLI drivers until sim_over / segment_end.
    `watch(port)` (optional) runs in a thread alongside (walltime stop test)."""
    from coffeebench.models.heuristic_roaster_model import HeuristicRoasterModel

    port = free_port()
    server = start_server(cfg, port, "--remote", "all", *extra)
    if watch is not None:
        threading.Thread(target=watch, args=(port,), daemon=True).start()
    envs = {a: {**cli_env(port, a, tmp, uds=True), **(env_extra or {}).get(a, {})} for a in AGENTS}
    state_dir.mkdir(parents=True, exist_ok=True)
    logs = {a: [] for a in AGENTS}
    digests = {a: [] for a in AGENTS}
    threads = []
    for a in AGENTS:
        if a == "farmer_B":
            make = None
        elif a == "roaster_A":
            make = HeuristicRoasterModel
        else:
            make = (lambda aid: (lambda: TraderModel(aid)))(a)
        if make is None:
            p = bash_loop(envs[a])
            threads.append(p)
            continue
        t = threading.Thread(target=policy_driver,
                             args=(envs[a], make, state_dir / f"{a}.pkl", logs[a], digests[a]))
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=900) if isinstance(t, threading.Thread) else t.wait(timeout=900)
    for a in AGENTS:
        fetch_audit(port, a)
    server.wait(timeout=120)
    return logs, digests, server.returncode


def test_resume(tmp: Path) -> bool:
    cfg = tmp / "hbsmoke_resume.toml"
    cfg.write_text(TRADE_CONFIG.format(name="hbsmoke_resume", max_days=10))
    run_dir = ROOT / OUT_ROOT / "hbsmoke_resume/seed_0"
    logs, _, rc = server_session(cfg, tmp, tmp / "state_full")
    full = snapshot_dir(run_dir, tmp / "resume_full")
    ref = normalized(full)
    ok = rc == 0
    print(f"  uninterrupted: rc={rc}; final statuses { {a: v[-1] for a, v in logs.items() if v} }")

    # Workspace persistence: notes written in segment 1; farmer_A's out/ dump
    # exceeds its 0.1 MB cap and must be dropped (the note kept).
    ws = tmp / "ws"
    for a in AGENTS:
        (ws / a).mkdir(parents=True, exist_ok=True)
        (ws / a / "note.txt").write_text(f"note of {a}\n")
    (ws / "farmer_A" / "out").mkdir(parents=True, exist_ok=True)
    (ws / "farmer_A" / "out" / "big.json").write_bytes(os.urandom(300_000))
    cap = {"farmer_A": {"COFFEE_WS_CAP_MB": "0.1"}}
    logs1, _, rc1 = server_session(cfg, tmp, tmp / "state_seg", "--segment-days", "5", env_extra=cap)
    stats = json.loads((run_dir / "harbor_server.json").read_text())
    saved = stats.get("workspaces_saved", {})
    print(f"  workspaces saved: { {a: (v['day'], v['files'], v['dropped']) for a, v in saved.items()} }")
    ok &= set(saved) == set(AGENTS) and saved["farmer_A"]["dropped"] == 1
    for a in AGENTS:  # fresh containers
        for f in sorted((ws / a).rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
    ckpt = stats.get("resume_ckpt")
    print(f"  segment 1: rc={rc1}; end day={stats['segment_end_day']}; ckpt={ckpt and Path(ckpt).name}; "
          f"statuses { {a: v[-1] for a, v in logs1.items() if v} }")
    ok &= rc1 == 0 and stats["sim_state"] == "segment_end" and bool(ckpt)
    ok &= all(v[-1] == "segment_end" for v in logs1.values() if v)
    logs2, digests2, rc2 = server_session(cfg, tmp, tmp / "state_seg", "--segment-days", "5",
                                          "--resume", ckpt)
    have_digest = {a: bool(d and d[0] and "RESUMED SESSION" in d[0]) for a, d in digests2.items() if d}
    print(f"  segment 2: rc={rc2}; statuses { {a: v[-1] for a, v in logs2.items() if v} }; digest={have_digest}")
    ok &= rc2 == 0 and all(have_digest.values()) and len(have_digest) == 5
    restored = {a: (ws / a / "note.txt").exists() and (ws / a / "note.txt").read_text() == f"note of {a}\n"
                for a in AGENTS}
    out_dropped = not (ws / "farmer_A" / "out" / "big.json").exists()
    print(f"  workspace restored in segment 2: {restored}; farmer_A out/ dropped={out_dropped}")
    ok &= all(restored.values()) and out_dropped
    got = normalized(run_dir)
    ok &= compare(ref, got, "uninterrupted vs 2 segments")
    ni_ref, ni_got = ni_of(full / "run.json"), ni_of(run_dir / "run.json")
    print(f"  NI identical={ni_ref == ni_got}: {ni_got}")
    return ok and ni_ref == ni_got


def test_auth(tmp: Path) -> bool:
    cfg = write_config(tmp, "hbsmoke_auth", 2)
    port = free_port()
    server = start_server(cfg, port, "--remote", "all")
    me, other = agent_tok("farmer_A"), "roaster_B"
    harness, verifier = scoped_tok("harness", "farmer_A"), scoped_tok("verifier", "farmer_A")
    nf = b'{"error":"not found"}'

    def is404(token, method, path, body=None, uds=False) -> bool:
        st, data, hdr = http_raw(port, token, method, path, body, uds=uds)
        return st == 404 and data == nf and "server" not in hdr

    checks = {
        # agent token: the CLI endpoints, only as itself
        "register_as_other": http(port, me, "POST", "/register", {"agent": other})[0] == 403,
        "act_as_other": http(port, me, "POST", "/act", {"agent": other, "tool": "view_dashboard",
                                                        "args": {}})[0] == 403,
        "wait_as_other": http(port, me, "GET", f"/wait?agent={other}&timeout=1")[0] == 403,
        "tools_as_other": http(port, me, "GET", f"/tools?agent={other}")[0] == 403,
        "status_as_other": http(port, me, "GET", f"/status?agent={other}")[0] == 403,
        "workspace_get_other": http(port, me, "GET", f"/workspace?agent={other}")[0] == 403,
        "tools_own": http(port, me, "GET", "/tools")[0] == 200,
        "workspace_get_own": http(port, me, "GET", "/workspace")[0] == 200,
        "status_own_minimal": set(http(port, me, "GET", "/status")[1])
        == {"sim_state", "time", "day", "agent", "agent_done"},
        "status_own_uds": http(port, me, "GET", "/status", uds=True)[0] == 200,
        # agent token: everything else is a uniform 404 (TCP and Unix socket)
        "agent_404_paths": all(is404(me, m, p, b"{}" if m != "GET" else None, uds=u)
                               for u in (False, True)
                               for m, p in [("GET", "/"), ("GET", "/docs"), ("GET", "/openapi.json"),
                                            ("GET", "/health"), ("GET", "/redoc"),
                                            ("GET", "/audit/farmer_A"), (
                                                "GET", f"/audit/{other}"),
                                            ("POST", "/telemetry"), ("GET", "/resume_info"),
                                            ("POST", "/resume_ack"),
                                            ("POST", "/admin/stop_at_next_morning"),
                                            ("GET", "/register"), ("DELETE", "/act"),
                                            ("PUT", "/workspace"), ("FOO", "/status"),
                                            ("GET", "/status/"), ("GET", "/../etc/passwd")]),
        "wrong_token_404": is404("nope", "GET", "/status") and is404("nope", "POST", "/act", b"{}"),
        "no_token_404": is404(None, "GET", "/tools") and is404(None, "GET", "/openapi.json"),
        # harness token (CoffeeTerminus): telemetry / resume_info / status for itself only
        "harness_telemetry_own": http(port, harness, "POST", "/telemetry",
                                      {"agent": "farmer_A", "cost_so_far": 1})[0] == 200,
        "harness_telemetry_other": http(port, harness, "POST", "/telemetry",
                                        {"agent": other, "cost_so_far": 9})[0] == 403,
        "harness_resume_info": http(port, harness, "GET", "/resume_info")[0] == 200,
        "harness_status": "bankrupt" in http(port, harness, "GET", "/status")[1],
        "harness_no_act": is404(harness, "POST", "/act", b'{"tool": "view_dashboard"}'),
        "harness_no_audit": is404(harness, "GET", "/audit/farmer_A"),
        # verifier token: own /audit only
        "verifier_audit_own": http(port, verifier, "GET", "/audit/farmer_A")[0] == 200,
        "verifier_audit_other": is404(verifier, "GET", f"/audit/{other}"),
        "verifier_no_register": is404(verifier, "POST", "/register", b"{}"),
        "verifier_no_status": is404(verifier, "GET", "/status"),
        # admin (launcher) token
        "admin_status_all": "agents" in http(port, TOKEN, "GET", "/status")[1],
        "admin_audit_any": http(port, TOKEN, "GET", f"/audit/{other}")[0] == 200,
        "admin_not_agent": is404(TOKEN, "POST", "/register", b'{"agent": "farmer_A"}'),
        "admin_still_404_docs": is404(TOKEN, "GET", "/openapi.json"),
        # malformed / oversized bodies
        "bad_json_400": http_raw(port, me, "POST", "/act", b"{not json")[0] == 400,
        "non_object_400": http_raw(port, me, "POST", "/act", b"[1, 2]")[0] == 400,
        "oversized_413": http_raw(port, me, "POST", "/act", b"{" + b" " * 300_000 + b"}")[0] == 413,
        "missing_tool_400": http(port, me, "POST", "/act", {"args": {}})[0] == 400,
        "bad_timeout_400": http(port, me, "GET", "/wait?timeout=abc")[0] == 400,
    }
    # A CLI that relabels itself is refused (never executed as the other agent).
    env = cli_env(port, "farmer_A", tmp)
    env["COFFEE_AGENT_ID"] = other
    r = subprocess.run(["coffee", "view_dashboard"], env=env, capture_output=True, text=True, timeout=60)
    checks["cli_relabel"] = r.returncode != 0 and "403" in r.stdout
    # The CLI over the Unix socket (bogus COFFEE_URL) works.
    r = subprocess.run(["coffee", "help"], env=cli_env(port, "farmer_B", tmp, uds=True),
                       capture_output=True, text=True, timeout=60)
    checks["cli_uds_help"] = r.returncode == 0 and "coffee view_dashboard" in r.stdout
    tel = http(port, TOKEN, "GET", "/status")[1]["agents"]
    checks["no_foreign_telemetry"] = tel[other]["cost_usd"] == 0 and tel["farmer_A"]["cost_usd"] == 1
    # Flooding one principal → 429 (the CLI never comes close to the limit).
    codes = [http_raw(port, scoped_tok("harness", "retailer_A"), "GET", "/resume_info", pace=0)[0]
             for _ in range(80)]
    checks["rate_limit_429"] = 429 in codes and codes[0] == 200
    server.terminate()
    server.wait(timeout=30)
    bad = [k for k, v in checks.items() if not v]
    print(f"  {len(checks) - len(bad)}/{len(checks)} auth checks ok; failed: {bad}")
    return not bad


def test_walltime(tmp: Path) -> bool:
    """Walltime-driven stop through launch_sim.StopWatch (fake 1 h walltime
    with a huge per-day margin → fires once the first simulated day has been
    timed), then resume; must equal the uninterrupted run."""
    sys.path.insert(0, str(ROOT / "harbor_adapter"))
    from launch_sim import StopWatch

    cfg = tmp / "hbsmoke_resume.toml"
    cfg.write_text(TRADE_CONFIG.format(name="hbsmoke_resume", max_days=10))
    run_dir = ROOT / OUT_ROOT / "hbsmoke_resume/seed_0"
    if (tmp / "resume_full" / "run.json").exists():
        full = tmp / "resume_full"
    else:
        server_session(cfg, tmp, tmp / "state_full")
        full = snapshot_dir(run_dir, tmp / "resume_full")
    ref = normalized(full)
    fired: list = []

    def watch(port: int) -> None:
        w = StopWatch(f"http://127.0.0.1:{port}", TOKEN, deadline=time.time() + 3600,
                      margin_days=1e6, min_margin_sec=0)
        while not w.requested:
            st = w.poll()
            if st is not None and st.get("sim_state") not in ("waiting_registration", "running"):
                return
            time.sleep(0.2)
        fired.append(w.last_status.get("day"))

    logs1, _, rc1 = server_session(cfg, tmp, tmp / "state_wt", watch=watch)
    stats = json.loads((run_dir / "harbor_server.json").read_text())
    ckpt = stats.get("resume_ckpt")
    print(f"  stop requested at day {fired}; segment end day={stats.get('segment_end_day')} "
          f"reason={stats.get('stop_reason')}; ckpt={ckpt and Path(ckpt).name}")
    ok = rc1 == 0 and stats["sim_state"] == "segment_end" and stats.get("stop_reason") == "walltime"
    ok &= bool(fired) and 0 < stats["segment_end_day"] < 10
    logs2, _, rc2 = server_session(cfg, tmp, tmp / "state_wt", "--resume", ckpt)
    ok &= rc2 == 0 and compare(ref, normalized(run_dir), "uninterrupted vs walltime stop + resume")
    return ok


V2_CONFIG = """\
[experiment]
name = "{name}"
description = "heuristic_roaster_v2 through the env server (scripted agents only, $0)."

[run]
seeds = [0]
max_days = {max_days}
main_agent = "roaster_A"

[models]
default = "passive"
"""


def test_v2(tmp: Path) -> bool:
    """Hybrid server run like a baseline launch (`launch_sim.py --local
    roaster_A:heuristic_roaster_v2`): roaster_A in-process on the server,
    farmer_A + retailer_A over the CLI (TraderModel: lists green / bids on
    roasted / DMs roaster_A → reactive wakes), the rest passive."""
    cfg = tmp / "hbsmoke_v2.toml"
    cfg.write_text(V2_CONFIG.format(name="hbsmoke_v2", max_days=8))
    port = free_port()
    remote = ("farmer_A", "retailer_A")
    server = start_server(cfg, port, "--remote", ",".join(remote),
                          "--models", "roaster_A:heuristic_roaster_v2")
    logs = {a: [] for a in remote}
    threads = [threading.Thread(target=policy_driver,
                                args=(cli_env(port, a, tmp, uds=True),
                                      (lambda aid: (lambda: TraderModel(aid)))(a), None, logs[a], []))
               for a in remote]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)
    for a in remote:
        fetch_audit(port, a)
    server.wait(timeout=120)
    run_dir = ROOT / OUT_ROOT / "hbsmoke_v2/seed_0"
    data = json.loads((run_dir / "run.json").read_text())
    ev = events(run_dir / "run.events.jsonl")
    steps = [e for e in ev if e["type"] == "agent_step" and e["agent_id"] == "roaster_A"]
    acts: dict = {}
    for e in steps:
        acts[e["action"]] = acts.get(e["action"], 0) + 1
    errs = [(e["day"], e["action"], e["observation"].get("message")) for e in steps
            if e["observation"].get("status") == "error"]
    deals = [d for d in data["marketplace"]["deals"] if "roaster_A" in (d["buyer_id"], d["seller_id"])]
    per_day: dict = {}
    for e in steps:
        if e["action"] != "wait_for_next_day":
            per_day[e["day"]] = per_day.get(e["day"], 0) + 1
    ni = data["result"]["agents"]["roaster_A"]["audit"]["annual"]["true_net_income"]
    print(f"  models: {data['models']}")
    print(f"  roaster_A actions: {acts}; max/day={max(per_day.values())}; errors={errs[:5]}")
    print(f"  roaster_A deals: {[(d['deal_at'] // 1440, d['item_id'], d['qty'], d['unit_price']) for d in deals]}")
    print(f"  roaster_A NI={ni:.2f}; drivers ended {[logs[a][-1] for a in remote]}; server rc={server.returncode}")
    return (server.returncode == 0 and data["models"]["roaster_A"] == "heuristic_roaster_v2"
            and max(steps[-1]["day"], 0) == 7 and max(per_day.values()) <= 20 and len(deals) > 0
            and acts.get("roast", 0) > 0 and all(logs[a][-1] == "sim_over" for a in remote))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="equiv,deadline,segment,ckpt,digest,resume,auth,walltime")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    results = {}
    with tempfile.TemporaryDirectory(prefix="cb_hb_smoke_") as td:
        tmp = Path(td)
        install_cli(tmp)
        for name in args.only.split(","):
            print(f"[{name}]")
            t0 = time.time()
            results[name] = {"equiv": test_equiv, "deadline": test_deadline, "segment": test_segment,
                             "ckpt": test_ckpt, "digest": test_digest, "resume": test_resume,
                             "auth": test_auth, "walltime": test_walltime, "v2": test_v2}[name](tmp)
            print(f"  -> {'PASS' if results[name] else 'FAIL'} ({time.time() - t0:.1f}s)")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    threading.excepthook = lambda a: print(a)
    if "--child-sim" in sys.argv:
        sys.path.insert(0, str(ROOT))
        child_sim(sys.argv[1:])
    else:
        main()
