"""CoffeeBench env server for the Harbor harness.

    uv run python -m coffeebench.server --config experiments/x.toml --seed 0 \
        --port 8765 --run-token T [--remote roaster_A] [--models farmer_A:passive] \
        [--segment-days N] [--deadline-sec 1800] [--resume <run_dir>/checkpoints/day_015.pkl]

Builds the run exactly like `coffeebench.main` (same `build_run`), but agents
marked remote become `RemoteAgent`s driven over HTTP by the `coffee` CLI.
Default remote set = every agent whose resolved model is not a scripted
baseline (passive / heuristic_roaster); scripted agents stay in-process
(hybrid mode). An in-process LLM agent is refused (no paid calls from here).

Lifecycle: registration barrier (all remote agents ran `coffee start`) →
`env.run()` → sim_over | segment_end | aborted → write trajectories → linger
until every remote agent's verifier fetched `/audit` (or `--linger-sec`).

Checkpoints / segments (DESIGN §6): `Environment.checkpoint_hook` fires at
the start of every MORNING_OPEN bucket (before that day's mechanics; every
agent parked). The server writes `<run_dir>/checkpoints/latest.pkl` there each
day (crash recovery). With `--segment-days N` the run stops at the first day
that is a multiple of N after the start day: the checkpoint is kept as
`checkpoints/day_NNN.pkl`, all remote agents get `segment_end`, the server
writes `segment_NNN.json` (+ `resume_ckpt` in harbor_server.json) and exits.
The launcher can also end the segment at the next MORNING_OPEN via
`POST /admin/stop_at_next_morning` (run token = admin token; walltime stop),
recorded as `stop_reason` (planned | walltime) in harbor_server.json /
segment_NNN.json. Agents authenticate with per-agent tokens derived from the
run token (`app.agent_token`); at a segment end each CLI uploads its
/workspace (stored as workspaces/<agent>/day_NNN.tar.gz) and a resumed run
hands back the newest one taken at or before the checkpoint day.
`--resume <ckpt>` (same config/seed/models/max-days/remote set) restores the
env, re-creates the agents with their checkpointed state, reopens
run.events.jsonl truncated to the checkpoint offset, and hands every remote
agent a deterministic digest (`coffeebench.server.digest`) with its first
payload. The final segment's run.json / run.events.jsonl equal those of an
uninterrupted run (modulo wall-clock timestamps).

Turn timeouts are logged as separate `turn_timeout` events (the agent_step
that follows has observation.status == "turn_timeout") and counted per agent
in harbor_server.json, so timeout-idle is distinguishable from voluntary
`wait_for_next_day`. Every `coffee` tool call costs `tool_cost_minutes`
(uniform 30 min, read-only tools included); shell commands are free.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import socket
import sys
import time
from argparse import ArgumentParser, Namespace

from coffeebench.models import is_scripted
from coffeebench.server import agent_state, checkpoint
from coffeebench.server.app import Hub, build_app
from coffeebench.server.digest import build_digests
from coffeebench.server.remote_agent import RemoteAgent

# sun_path is 104 bytes on macOS, 108 on Linux (incl. the NUL terminator).
UDS_MAX_PATH = 100


def listen_sockets(host: str, port: int, uds: str | None) -> list[socket.socket]:
    """TCP listener (+ Unix socket listener) for `uvicorn.Server.serve(sockets=…)`."""
    tcp = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp.bind((host, port))
    tcp.listen(2048)
    tcp.setblocking(False)
    socks = [tcp]
    if uds:
        if len(os.fsencode(uds)) > UDS_MAX_PATH:
            raise SystemExit(f"[server] --uds path too long ({len(uds)} > {UDS_MAX_PATH}): {uds}")
        if os.path.exists(uds):
            os.unlink(uds)
        os.makedirs(os.path.dirname(os.path.abspath(uds)), exist_ok=True)
        us = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        us.bind(uds)
        # Containers run as (mapped) root or another uid: let anyone connect;
        # the per-agent bearer tokens are the access control.
        os.chmod(uds, 0o666)
        us.listen(2048)
        us.setblocking(False)
        socks.append(us)
    return socks


def segment_stop_day(start_day: int, segment_days: int, max_days: int) -> int | None:
    """First multiple of `segment_days` after `start_day` (fixed simulated-day
    boundaries for every model), or None if the run ends first."""
    if segment_days <= 0:
        return None
    stop = (start_day // segment_days + 1) * segment_days
    return stop if stop < max_days else None


def parse_args(argv: list[str] | None = None) -> Namespace:
    p = ArgumentParser(prog="python -m coffeebench.server")
    p.add_argument("--config", required=True, help="Experiment TOML (as for coffeebench.main).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_days", "--max-days", dest="max_days", type=int, default=None)
    p.add_argument("--models", default=None, help='Per-agent overrides "aid:model,..." (e.g. farmer_A:passive).')
    p.add_argument("--main_agent", "--main-agent", dest="main_agent", default=None)
    p.add_argument("--early-stop-agents", dest="early_stop_agents", default=None,
                   help="Comma list overriding the config's [run] early_stop_agents (head-to-head: both "
                   "evaluated seats, so the early stop is symmetric).")
    p.add_argument(
        "--remote",
        default=None,
        help="Comma list of agent ids driven over HTTP, or 'all'. Default: every "
        "agent whose resolved model is not scripted (passive/heuristic_roaster).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--uds", default=None,
                   help="Also listen on this Unix socket (bind-mounted into network-less task "
                   "containers; dir + socket chmod 0777, the tokens authenticate). "
                   f"Path ≤ {UDS_MAX_PATH} bytes.")
    p.add_argument("--run-token", default=os.environ.get("COFFEE_TOKEN"),
                   help="Admin token; agent tokens are derived from it (app.agent_token).")
    p.add_argument("--deadline-sec", type=float, default=1800.0, help="Wake deadline per turn (wall clock).")
    p.add_argument("--registration-timeout", type=float, default=1800.0)
    p.add_argument(
        "--segment-days",
        type=int,
        default=0,
        help="Segment length: stop at the next day that is a multiple of N (0 = run to the end).",
    )
    p.add_argument("--resume", default=None, help="Checkpoint .pkl to continue from.")
    p.add_argument(
        "--no-daily-checkpoint",
        action="store_true",
        help="Do not write checkpoints/latest.pkl every simulated day.",
    )
    p.add_argument("--workspace-timeout", type=float, default=60.0,
                   help="Max wait for the parked agents' /workspace uploads at a checkpoint.")
    p.add_argument("--linger-sec", type=float, default=1800.0, help="Max wait for verifiers after the end.")
    p.add_argument("--output-root", default="trajectories/harbor")
    p.add_argument("--ready-file", default=None, help="Write {url, token, pid} JSON here once listening.")
    return p.parse_args(argv)


def build(args: Namespace, open_events: bool = True):
    from coffeebench.main import build_run

    explicit = None
    if args.remote and args.remote != "all":
        explicit = {a.strip() for a in args.remote.split(",") if a.strip()}
    remote: dict[str, RemoteAgent] = {}

    def factory(aid: str, model_id: str, tools: list, prompt: str):
        scripted = is_scripted(model_id)
        if args.remote == "all" or (explicit is not None and aid in explicit) or (
            explicit is None and not scripted
        ):
            from coffeebench.main import INITIAL_OBSERVATION

            ra = RemoteAgent(
                name=aid,
                tools=tools,
                instruct_prompt=prompt,
                initial_observation=INITIAL_OBSERVATION,
                model_label=model_id,
                deadline_sec=args.deadline_sec,
            )
            remote[aid] = ra
            print(f"[server] {aid} → REMOTE (CLI over HTTP; config model {model_id})")
            return ra
        if not scripted:
            raise ValueError(
                f"{aid} would run in-process with LLM model {model_id!r}; the server "
                "never makes LLM calls. Mark it remote or override with --models "
                f"{aid}:passive."
            )
        return None

    run_args = Namespace(
        config=args.config,
        model=None,
        models=args.models,
        max_days=args.max_days,
        seed=args.seed,
        main_agent=args.main_agent,
        early_stop_agents=args.early_stop_agents,
        output=None,
        output_root=args.output_root,
    )
    env, output_path, events_path = build_run(
        run_args, agent_factory=factory, open_events=open_events
    )
    if not remote:
        raise SystemExit("[server] no remote agents — use coffeebench.main instead")
    return env, remote, output_path, events_path


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


async def serve(args: Namespace) -> int:
    import uvicorn

    env, remote, output_path, events_path = build(args, open_events=not args.resume)
    run_dir = os.path.dirname(output_path)
    start_day = 0
    if args.resume:
        ckpt = checkpoint.load(args.resume)
        env = checkpoint.restore(env, ckpt, events_path)
        start_day = int(ckpt["day"])
        digests = build_digests(env, sorted(remote), events_path)
        for aid, ra in remote.items():
            ra.resume_digest = digests[aid]
            if aid in env._bankrupt_agents:
                ra.on_end("bankrupt")
        print(f"[server] resumed from {args.resume} at day {start_day}", flush=True)
    token = args.run_token or secrets.token_urlsafe(24)
    hub = Hub(env, remote, token, run_dir=run_dir, resume_day=start_day if args.resume else None)
    for aid, ra in remote.items():
        ra.restore_workspace = bool(hub.restore_ws.get(aid))
    hub.segment_end_day = segment_stop_day(start_day, args.segment_days, env.max_days)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    resume_ckpt: dict[str, str] = {}

    async def _checkpoint_hook(env_, day: int) -> bool:
        planned = hub.segment_end_day is not None and day >= hub.segment_end_day
        stop = planned or hub.stop_requested is not None
        if day == 0 or (args.no_daily_checkpoint and not stop):
            return False
        # Every parked agent's CLI uploads /workspace as of this checkpoint
        # (before pickling: an over-cap note lands in the checkpointed outbox).
        await hub.collect_workspaces(day, timeout=args.workspace_timeout)
        latest = os.path.join(ckpt_dir, "latest.pkl")
        checkpoint.save(env_, day, latest, extra={"segment_start_day": start_day})
        # Exact resume: which harness snapshot each parked agent continues from.
        refs = {"day": day, "calls": {aid: (ra.pending_call if ra._open_ticket is not None else None)
                                      for aid, ra in remote.items()}}
        agent_state.update_refs(ckpt_dir, "latest.pkl", refs)
        if stop:
            path = os.path.join(ckpt_dir, f"day_{day:03d}.pkl")
            shutil.copyfile(latest, path)
            agent_state.update_refs(ckpt_dir, os.path.basename(path), refs)
            resume_ckpt["path"] = path
            hub.stop_reason = "planned" if planned else hub.stop_requested
            hub.stopped_day = day
        agent_state.gc_snapshots(os.path.join(run_dir, "agent_state"), ckpt_dir)
        agent_state.gc_workspaces(os.path.join(run_dir, "workspaces"), ckpt_dir)
        return stop

    env.checkpoint_hook = _checkpoint_hook

    app = build_app(hub)
    # No `server:` / `date:` headers: nothing reveals the stack to a probing client.
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="warning", access_log=False,
                       server_header=False, date_header=False, lifespan="on",
                       limit_concurrency=1024, timeout_keep_alive=5)
    )
    socks = listen_sockets(args.host, args.port, args.uds)
    server_task = asyncio.create_task(server.serve(sockets=socks))
    while not server.started:
        if server_task.done():
            await server_task
            return 3
        await asyncio.sleep(0.05)
    url = f"http://{args.host}:{args.port}"
    print(f"[server] listening on {url}" + (f" + unix:{args.uds}" if args.uds else "")
          + f"; remote agents: {sorted(remote)}", flush=True)
    if args.ready_file:
        _write_json(args.ready_file, {"url": url, "socket": args.uds, "token": token,
                                      "pid": os.getpid()})

    rc = 0
    try:
        await asyncio.wait_for(hub.all_registered.wait(), args.registration_timeout)
    except TimeoutError:
        missing = sorted(a for a, ra in remote.items() if not ra.registered)
        print(f"[server] registration timeout; missing: {missing}", flush=True)
        hub.sim_state = "aborted"
        hub.end_all("aborted")
        env._emit("harbor_aborted", reason="registration_timeout", missing=missing)
        rc = 2
    else:
        hub.sim_state = "running"
        t0 = time.time()
        try:
            result = await env.run(resume=bool(args.resume))
        except BaseException as exc:  # noqa: BLE001 — save what we have, then re-raise
            print(f"[server] env.run() crashed: {type(exc).__name__}: {exc}", flush=True)
            hub.sim_state = "aborted"
            hub.end_all("aborted")
            try:
                env.save_trajectory(output_path)
            except Exception as save_exc:  # noqa: BLE001
                print(f"[server] partial save failed: {save_exc}")
            server.should_exit = True
            raise
        else:
            if env.stopped_at_day is not None:
                day = env.stopped_at_day
                hub.sim_state = "segment_end"
                hub.end_all("segment_end")
                # Written after the checkpoint offset → dropped on resume.
                env._emit("segment_end", day=day, resume_ckpt=resume_ckpt.get("path"))
                seg_path = os.path.join(run_dir, f"segment_{day:03d}.json")
                _write_json(
                    seg_path,
                    {
                        "segment_start_day": start_day,
                        "segment_end_day": day,
                        "stop_reason": hub.stop_reason,
                        "resume_ckpt": resume_ckpt.get("path"),
                        "audit": {aid: env._compute_audit_for(aid) for aid in env.business_apps},
                        "bankrupt_day": dict(env._bankrupt_day),
                        "models": {aid: a.model.model for aid, a in env.agents.items()},
                    },
                )
                print(f"[server] segment end at day {day}; wrote {seg_path}", flush=True)
            else:
                hub.final_result = result
                hub.sim_state = "sim_over"
                hub.end_all("sim_over")
                env.save_trajectory(output_path)
        print(f"[server] simulation wall time {time.time() - t0:.1f}s", flush=True)

    def _stats() -> dict:
        return {
            "sim_state": hub.sim_state,
            "segment_start_day": start_day,
            "segment_end_day": hub.stopped_day if hub.stopped_day is not None else hub.segment_end_day,
            "stop_reason": hub.stop_reason,
            "resume_ckpt": resume_ckpt.get("path"),
            "deadline_sec": args.deadline_sec,
            "wall_sec": round(time.time() - hub.started_at, 1),
            "agents": {
                aid: {
                    "turns": ra.n_turns,
                    "timeouts": ra.n_timeouts,
                    "conflicts": ra.n_conflicts,
                    "ended": ra.ended,
                    "telemetry": ra.telemetry,
                }
                for aid, ra in remote.items()
            },
            "workspaces_saved": hub.saved_ws,
            "resume_mode": hub.resume_mode,
            "workspaces_restored": {a: p and os.path.basename(p) for a, p in hub.restore_ws.items()},
        }

    def _write_stats() -> None:
        stats = _stats()
        _write_json(os.path.join(run_dir, "harbor_server.json"), stats)
        _write_json(os.path.join(run_dir, f"harbor_server_d{start_day:03d}.json"), stats)

    _write_stats()
    if env.event_logger is not None:
        env.event_logger.close()

    # Linger so verifiers (tests/test.sh → /audit) can read the final score.
    deadline = time.time() + args.linger_sec
    while time.time() < deadline and hub.audit_fetched != set(remote):
        hub.audit_event.clear()
        try:
            await asyncio.wait_for(hub.audit_event.wait(), max(0.1, deadline - time.time()))
        except TimeoutError:
            break
    await asyncio.sleep(1.0)  # let the last responses flush
    _write_stats()  # + workspace uploads received while lingering
    server.should_exit = True
    await server_task
    if args.uds and os.path.exists(args.uds):
        os.unlink(args.uds)
    print(f"[server] exit ({hub.sim_state})", flush=True)
    return rc


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sys.exit(asyncio.run(serve(args)))


if __name__ == "__main__":
    main()
