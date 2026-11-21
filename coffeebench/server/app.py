"""HTTP front-end of the CoffeeBench env server (Harbor harness).

Transport: the server listens on host-only TCP 127.0.0.1 (launcher,
CoffeeTerminus) and on a Unix socket that is bind-mounted into the task
containers (`network_mode: none`; the CLI uses $COFFEE_SOCKET).

Auth: `Authorization: Bearer <token>`. Tokens are scoped, all derived from the
run (admin) token with HMAC-SHA256 (`scoped_token`):

    agent     agent_token(run, aid)            container COFFEE_TOKEN (the `coffee` CLI)
    harness   scoped_token(run, "harness", aid)  Harbor host only (CoffeeTerminus)
    verifier  scoped_token(run, "verifier", aid) tests/test.sh only ([verifier.env])
    admin     the run token itself               launcher only

Strict allowlist (agent tokens see only what the CLI needs):

    POST /register   {agent}                 → {ticket}     agent  (`coffee start`)
    POST /act        {agent, tool, args}     → {ticket}     agent  (409 if a ticket is pending)
    GET  /wait?agent&ticket&timeout=50       → payload      agent  (| {"status": "pending"})
    GET  /tools?agent                        → tool catalog agent
    POST /workspace  {data: base64 tar.gz}   → {}           agent  (checkpoint upload)
    GET  /workspace                          → {data|null}  agent  (resumed `coffee start`)
    GET  /status[?agent]                     → agent: own minimal view; harness: own
                                               (+ bankrupt list …); admin: all agents
    POST /telemetry  {agent, cost_so_far, …} → {}           harness
    GET  /resume_info                        → {resumed, day, pending_call}   harness
    POST /resume_ack {mode: exact|fallback}  → {}           harness: exact = context restored,
                                                            no digest, silent workspace restore
    GET  /audit/{agent}                      → own truth-ledger score   verifier (own) / admin
    POST /admin/stop_at_next_morning {reason} → {}          admin (launcher walltime stop)

Everything else — unknown paths (incl. /, /docs, /openapi.json, /health),
wrong methods, unknown tokens, and endpoints outside the token's scope — gets
the same `404 {"error": "not found"}`. A client-sent `agent` that differs from
the token's agent → 403. Bodies are capped (413) and must be JSON objects
(400); non-admin principals are rate-limited (429).

/register, /act and /wait accept an optional `call` id (`<seq>.<k>`, see
coffeebench.server.agent_state) naming the harness snapshot taken before the
CLI call; the id parked on the open ticket is checkpointed.

The `Hub` owns the sim lifecycle state; `RemoteAgent`s own per-agent tickets.
Everything runs on one asyncio loop together with `Environment.run()`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import os
import time
import traceback
from collections import deque

from starlette.requests import Request
from starlette.responses import JSONResponse

from coffeebench.event_loop import format_min
from coffeebench.server.prompts import tool_catalog
from coffeebench.server.remote_agent import Conflict, RemoteAgent

MAX_WAIT_POLL_SEC = 55.0
MAX_WORKSPACE_BYTES = 32 * 1024 * 1024  # compressed upload cap (the CLI caps at 20 MB raw)
MAX_WORKSPACE_BODY_BYTES = MAX_WORKSPACE_BYTES * 4 // 3 + 64 * 1024  # base64 + JSON envelope
MAX_BODY_BYTES = 256 * 1024  # every other JSON body
RATE_PER_SEC = 40  # requests per second per non-admin principal (outside UNLIMITED)
MAX_INFLIGHT_PER_PRINCIPAL = 16  # concurrent requests per non-admin principal (v2.1)
UNLIMITED = {("POST", "/register"), ("POST", "/act"), ("GET", "/wait"), ("POST", "/workspace")}

# Token scopes.
AGENT, HARNESS, VERIFIER, ADMIN = "agent", "harness", "verifier", "admin"

_NOT_FOUND = JSONResponse({"error": "not found"}, status_code=404)
_BAD = JSONResponse({"error": "bad request"}, status_code=400)
_TOO_LARGE = JSONResponse({"error": "request too large"}, status_code=413)


def agent_token(run_token: str, agent_id: str) -> str:
    """Per-agent bearer token derived from the run (admin) token (the CLI's)."""
    return hmac.new(run_token.encode(), agent_id.encode(), hashlib.sha256).hexdigest()


def scoped_token(run_token: str, scope: str, agent_id: str) -> str:
    """Harness / verifier token of `agent_id` (never visible to the agent)."""
    return agent_token(run_token, f"{scope}:{agent_id}")


def workspace_path(run_dir: str, aid: str, day: int) -> str:
    return os.path.join(run_dir, "workspaces", aid, f"day_{day:03d}.tar.gz")


def latest_workspace(run_dir: str, aid: str, max_day: int) -> str | None:
    """Newest stored workspace of `aid` taken at or before `max_day` (the
    resume checkpoint's day), or None."""
    d = os.path.join(run_dir, "workspaces", aid)
    best: tuple[int, str] | None = None
    try:
        names = os.listdir(d)
    except OSError:
        return None
    for n in names:
        if n.startswith("day_") and n.endswith(".tar.gz"):
            try:
                day = int(n[4:-7])
            except ValueError:
                continue
            if day <= max_day and (best is None or day > best[0]):
                best = (day, os.path.join(d, n))
    return best[1] if best else None


class Hub:
    """Shared state between the HTTP layer and the server main loop."""

    def __init__(
        self,
        env,
        remote_agents: dict[str, RemoteAgent],
        token: str,
        run_dir: str | None = None,
        resume_day: int | None = None,
    ):
        self.env = env
        self.remote = remote_agents
        self.token = token
        # token → (scope, agent id | None)
        self.principals: dict[str, tuple[str, str | None]] = {token: (ADMIN, None)}
        for aid in remote_agents:
            self.principals[agent_token(token, aid)] = (AGENT, aid)
            self.principals[scoped_token(token, HARNESS, aid)] = (HARNESS, aid)
            self.principals[scoped_token(token, VERIFIER, aid)] = (VERIFIER, aid)
        self.run_dir = run_dir
        # Workspace restored on a resumed `coffee start` (as of the checkpoint day).
        self.restore_ws: dict[str, str | None] = {
            aid: (latest_workspace(run_dir, aid, resume_day) if run_dir and resume_day else None)
            for aid in remote_agents
        }
        self.saved_ws: dict[str, dict] = {}
        self.resume_day = resume_day
        # Resumed run: per agent "exact" (CoffeeTerminus restored its context)
        # or "fallback_digest" (fresh context + digest; default).
        self.resume_mode: dict[str, str] = (
            {aid: "fallback_digest" for aid in remote_agents} if resume_day is not None else {}
        )
        # Launcher walltime stop: checkpoint + segment_end at the next MORNING_OPEN.
        self.stop_requested: str | None = None
        self.stop_reason: str | None = None
        self.stopped_day: int | None = None
        # waiting_registration → running → (sim_over | segment_end | aborted)
        self.sim_state = "waiting_registration"
        self.final_result: dict | None = None
        self.segment_end_day: int | None = None
        self.all_registered = asyncio.Event()
        self.audit_fetched: set[str] = set()
        self.audit_event = asyncio.Event()
        self.started_at = time.time()
        for a in remote_agents.values():
            a.env = env

    # ----- lifecycle -----

    async def collect_workspaces(self, day: int, timeout: float) -> None:
        """Ask every parked agent whose CLI is polling to upload /workspace
        for checkpoint `day`; wait for all (≤ timeout)."""
        if not self.run_dir:
            return
        now = time.time()
        waits = [ra.request_workspace(day) for ra in self.remote.values()
                 if ra._open_ticket is not None and not ra.ended and now - ra.last_poll_at < 70]
        if not waits:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in waits)), timeout)
        except TimeoutError:
            missing = [ra.name for ra in self.remote.values() if ra.ws_request is not None]
            print(f"[server] day {day}: no workspace upload from {missing} within {timeout:.0f}s",
                  flush=True)
        for ra in self.remote.values():
            ra.ws_request = None

    def mark_registered(self, aid: str) -> None:
        if all(a.registered for a in self.remote.values()):
            self.all_registered.set()

    def end_all(self, reason: str) -> None:
        for a in self.remote.values():
            a.on_end(reason)

    def sim_done(self) -> bool:
        return self.sim_state in ("sim_over", "segment_end", "aborted")

    # ----- views -----

    def status(self, aid: str | None = None, admin: bool = False) -> dict:
        env = self.env
        vt = env.time_manager.get_virtual_min()
        out = {
            "sim_state": self.sim_state,
            "time": format_min(vt),
            "day": env.time_manager.get_current_day(),
            "max_days": env.max_days,
            "segment_end_day": self.segment_end_day,
            "stop_requested": self.stop_requested,
            "bankrupt": sorted(env._bankrupt_agents),
        }
        if admin:
            out["agents"] = {
                a: {
                    "state": ra.state(),
                    "turns": ra.n_turns,
                    "timeouts": ra.n_timeouts,
                    "cost_usd": round(ra.model.cost, 4),
                }
                for a, ra in self.remote.items()
            }
        if aid is not None:
            ra = self.remote.get(aid)
            out["agent"] = aid
            out["agent_done"] = bool(self.sim_done() or (ra is not None and ra.is_done()))
        return out

    def agent_status(self, aid: str) -> dict:
        """Minimal own view for the agent's CLI (`coffee status`)."""
        env = self.env
        ra = self.remote.get(aid)
        return {
            "sim_state": self.sim_state,
            "time": format_min(env.time_manager.get_virtual_min()),
            "day": env.time_manager.get_current_day(),
            "agent": aid,
            "agent_done": bool(self.sim_done() or (ra is not None and ra.is_done())),
        }

    def audit(self, aid: str) -> dict:
        env = self.env
        bankrupt = aid in env._bankrupt_agents
        final = self.sim_state == "sim_over" and self.final_result is not None
        if final:
            per = self.final_result["agents"][aid]
            annual = per["audit"]["annual"]
        else:
            annual = env._compute_audit_for(aid)["annual"]
        return {
            "agent": aid,
            "sim_state": self.sim_state,
            "final": final,
            "bankrupt": bankrupt,
            "true_net_income": annual.get("true_net_income"),
            "annual": annual,
        }


def build_app(hub: Hub):
    """Plain ASGI app (no framework routing): a fixed (method, path) → handler
    table, each entry with the token scopes allowed to use it. Anything else —
    unknown path, wrong method, missing/unknown token, or a scope the endpoint
    does not serve — gets the same `404 {"error": "not found"}`, so nothing
    about the server can be discovered from inside a container."""
    rate: dict[str, deque] = {}

    def _scope(request: Request) -> tuple[str, str | None] | None:
        header = request.headers.get("authorization", "")
        got = (header[7:] if header.startswith("Bearer ") else "").encode()
        for tok, principal in hub.principals.items():
            if hmac.compare_digest(got, tok.encode()):
                return principal
        return None

    def _check_claim(aid: str, claimed) -> JSONResponse | None:
        """A client-sent agent id is only a label: a different one is refused."""
        if claimed and claimed != aid:
            return JSONResponse(
                {"error": f"credentials belong to {aid}; cannot act as {claimed!r}"},
                status_code=403,
            )
        return None

    def _call(value) -> str | None:
        return str(value)[:40] if value else None

    async def _read_body(request: Request, cap: int) -> dict | JSONResponse:
        """JSON object body, at most `cap` bytes; 413 / 400 otherwise."""
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            return _BAD
        if declared > cap:
            return _TOO_LARGE
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > cap:
                return _TOO_LARGE
        if not buf:
            return {}
        try:
            data = json.loads(buf)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return _BAD
        return data if isinstance(data, dict) else _BAD

    # ----- agent scope (the `coffee` CLI) -----

    async def register(request: Request, ra: RemoteAgent, data: dict):
        if hub.sim_state == "aborted":
            ra.on_end("aborted")
        ticket = ra.register(call=_call(data.get("call")))
        hub.mark_registered(ra.name)
        return JSONResponse({"ticket": ticket, "sim_state": hub.sim_state})

    async def act(request: Request, ra: RemoteAgent, data: dict):
        tool = data.get("tool")
        args = data.get("args") or {}
        if not isinstance(tool, str) or not tool:
            return JSONResponse({"error": "missing 'tool'"}, status_code=400)
        if not isinstance(args, dict):
            return JSONResponse({"error": "'args' must be an object"}, status_code=400)
        if not ra.registered:
            return JSONResponse(
                {"error": "not started; run `coffee start` first"}, status_code=400
            )
        try:
            ticket = ra.submit(tool, args, call=_call(data.get("call")))
        except Conflict as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"ticket": ticket})

    async def wait(request: Request, ra: RemoteAgent, data: dict):
        q = request.query_params
        try:
            ticket = int(q["ticket"]) if q.get("ticket") else None
            timeout = max(0.0, min(float(q.get("timeout", 50)), MAX_WAIT_POLL_SEC))
        except ValueError:
            return JSONResponse({"error": "bad ticket/timeout"}, status_code=400)
        return JSONResponse(await ra.wait_ticket(ticket, timeout, call=_call(q.get("call"))))

    async def tools(request: Request, ra: RemoteAgent, data: dict):
        return JSONResponse({"agent": ra.name, "tools": tool_catalog(ra.tools)})

    async def workspace_put(request: Request, ra: RemoteAgent, data: dict):
        """Workspace tarball for checkpoint day D (requested via /wait while
        the agent is parked), stored as <run_dir>/workspaces/<agent>/day_DDD.tar.gz."""
        day = data.get("day")
        if ra.ws_request is None or day != ra.ws_request or not hub.run_dir:
            return JSONResponse({"error": "no workspace upload requested"}, status_code=409)
        try:
            blob = base64.b64decode(str(data.get("data") or ""), validate=True)
        except (binascii.Error, ValueError):
            return JSONResponse({"error": "bad base64"}, status_code=400)
        if len(blob) > MAX_WORKSPACE_BYTES:
            return JSONResponse({"error": "workspace too large"}, status_code=413)
        path = workspace_path(hub.run_dir, ra.name, int(day))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
        dropped = [str(x) for x in (data.get("dropped") or [])][:200]
        files = data.get("files")
        hub.saved_ws[ra.name] = {
            "day": int(day),
            "bytes": len(blob),
            "files": files if isinstance(files, int) else None,
            "dropped": len(dropped),
        }
        ra.workspace_uploaded(dropped)
        return JSONResponse({"saved": os.path.basename(path), "bytes": len(blob)})

    async def workspace_get(request: Request, ra: RemoteAgent, data: dict):
        path = hub.restore_ws.get(ra.name)
        if not path or not os.path.exists(path):
            return JSONResponse({"data": None})
        with open(path, "rb") as f:
            blob = f.read()
        return JSONResponse({"data": base64.b64encode(blob).decode(), "source": os.path.basename(path)})

    # ----- harness scope (CoffeeTerminus, Harbor host process) -----

    async def telemetry(request: Request, ra: RemoteAgent, data: dict):
        ra.note_telemetry(data)
        return JSONResponse({})

    async def resume_info(request: Request, ra: RemoteAgent, data: dict):
        return JSONResponse({
            "resumed": hub.resume_day is not None,
            "day": hub.resume_day,
            "pending_call": ra.resume_call,
        })

    async def resume_ack(request: Request, ra: RemoteAgent, data: dict):
        if hub.resume_day is None or ra.registered:
            return JSONResponse({"error": "not a fresh resumed session"}, status_code=409)
        mode = "exact" if data.get("mode") == "exact" else "fallback_digest"
        hub.resume_mode[ra.name] = mode
        if mode == "exact":
            ra.resume_digest = None
            if ra.restore_workspace:
                ra.restore_workspace = "quiet"
        print(f"[server] {ra.name}: resume mode {mode}"
              + ("" if mode == "exact" else " (resume_fallback_digest)"), flush=True)
        return JSONResponse({"mode": mode})

    # ----- shared / admin -----

    async def status(request: Request, scope: str, aid: str | None):
        if scope == ADMIN:
            return JSONResponse(hub.status(request.query_params.get("agent"), admin=True))
        deny = _check_claim(aid, request.query_params.get("agent"))
        if deny is not None:
            return deny
        if scope == HARNESS:
            return JSONResponse(hub.status(aid))
        return JSONResponse(hub.agent_status(aid))

    async def audit(request: Request, scope: str, aid: str | None, target: str):
        if scope != ADMIN and target != aid:
            return _NOT_FOUND
        if target not in hub.env.business_apps:
            return _NOT_FOUND
        out = hub.audit(target)
        ra = hub.remote.get(target)
        # Only the agent's own verifier counts for the end-of-run linger.
        if scope == VERIFIER and ra is not None and (hub.sim_done() or ra.is_done()):
            hub.audit_fetched.add(target)
            hub.audit_event.set()
        return JSONResponse(out)

    async def admin_stop(request: Request, data: dict):
        hub.stop_requested = str(data.get("reason") or "walltime")[:40]
        print(f"[server] stop requested ({hub.stop_requested}): segment end at the next morning",
              flush=True)
        return JSONResponse({"stop_requested": hub.stop_requested, "day": hub.status()["day"]})

    # (method, path) → (handler, scopes, body cap or None for no body)
    agent_routes = {
        ("POST", "/register"): (register, {AGENT}, MAX_BODY_BYTES),
        ("POST", "/act"): (act, {AGENT}, MAX_BODY_BYTES),
        ("GET", "/wait"): (wait, {AGENT}, None),
        ("GET", "/tools"): (tools, {AGENT}, None),
        ("POST", "/workspace"): (workspace_put, {AGENT}, MAX_WORKSPACE_BODY_BYTES),
        ("GET", "/workspace"): (workspace_get, {AGENT}, None),
        ("POST", "/telemetry"): (telemetry, {HARNESS}, MAX_BODY_BYTES),
        ("GET", "/resume_info"): (resume_info, {HARNESS}, None),
        ("POST", "/resume_ack"): (resume_ack, {HARNESS}, MAX_BODY_BYTES),
    }

    def _rate_ok(key: str) -> bool:
        now = time.monotonic()
        q = rate.setdefault(key, deque())
        while q and now - q[0] > 1.0:
            q.popleft()
        if len(q) >= RATE_PER_SEC:
            return False
        q.append(now)
        return True

    inflight: dict[str, int] = {}

    async def dispatch(request: Request):
        principal = _scope(request)
        if principal is None:
            return _NOT_FOUND
        # v2.1: cap concurrent requests per principal (incl. the exempt long-poll
        # /wait and /act), so one agent cannot exhaust the server's connections.
        key = f"{principal[0]}:{principal[1]}"
        if principal[0] != ADMIN and inflight.get(key, 0) >= MAX_INFLIGHT_PER_PRINCIPAL:
            return JSONResponse({"error": "too many concurrent requests"}, status_code=429)
        inflight[key] = inflight.get(key, 0) + 1
        try:
            return await _dispatch(request, principal)
        finally:
            inflight[key] -= 1

    async def _dispatch(request: Request, principal):
        method, path = request.method, request.url.path
        scope, aid = principal
        route = agent_routes.get((method, path))
        # Sim calls are serialized by tickets (a second one → 409) and may be
        # fast for scripted drivers: no limit. Everything else (status, tools,
        # telemetry, probes of unknown paths) is rate-limited per principal.
        if (scope != ADMIN and (method, path) not in UNLIMITED
                and not _rate_ok(f"{scope}:{aid}")):
            return JSONResponse({"error": "too many requests"}, status_code=429)
        if route is not None:
            handler, scopes, cap = route
            if scope not in scopes:
                return _NOT_FOUND
            data: dict = {}
            if cap is not None:
                body = await _read_body(request, cap)
                if isinstance(body, JSONResponse):
                    return body
                data = body
            claimed = data.get("agent") if cap is not None else request.query_params.get("agent")
            deny = _check_claim(aid, claimed)
            if deny is not None:
                return deny
            return await handler(request, hub.remote[aid], data)
        if method == "GET" and path == "/status" and scope in (AGENT, HARNESS, ADMIN):
            return await status(request, scope, aid)
        if method == "GET" and path.startswith("/audit/") and scope in (VERIFIER, ADMIN):
            return await audit(request, scope, aid, path[len("/audit/"):])
        if method == "POST" and path == "/admin/stop_at_next_morning" and scope == ADMIN:
            body = await _read_body(request, MAX_BODY_BYTES)
            return body if isinstance(body, JSONResponse) else await admin_stop(request, body)
        return _NOT_FOUND

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return  # websockets etc.: connection dropped
        request = Request(scope, receive)
        try:
            response = await dispatch(request)
        except Exception:  # noqa: BLE001 — uniform error, details only in the server log
            traceback.print_exc()
            response = JSONResponse({"error": "internal error"}, status_code=500)
        await response(scope, receive, send)

    return app
