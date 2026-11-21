#!/usr/bin/env python3
"""`coffee` — stdlib-only CLI for the CoffeeBench env server (Harbor harness).

    coffee start                         join the simulation (blocks until it opens)
    coffee <tool> --key value ...        one simulation turn (blocks until your next turn)
    coffee <tool> '{"key": value}'       same, JSON arguments
    coffee <tool> - <<'EOF' ... EOF      JSON object on stdin (text with `$` stays verbatim)
    coffee wait                          keep waiting for a pending turn
    coffee status                        simulation status (no simulated time)
    coffee help [tool]                   list tools / show one tool's schema

Environment: COFFEE_SOCKET (Unix socket of the env server; the task
containers have no network) or else COFFEE_URL (http://host:port, TCP),
COFFEE_TOKEN (per-agent credential: the server derives your identity from
it), COFFEE_AGENT_ID (label only); optional
COFFEE_OUT_DIR (default /workspace/out), COFFEE_MAX_WAIT (seconds, default
6900), COFFEE_WORKSPACE (default /workspace), COFFEE_WS_CAP_MB (default 20).

Workspace persistence: at every daily checkpoint (all agents parked in a
`coffee` call) the server answers the CLI's /wait poll with
`upload_workspace: <day>`; the CLI uploads a tar.gz of $COFFEE_WORKSPACE
($COFFEE_OUT_DIR is dropped first, then the largest files, if the total
exceeds the cap) and keeps waiting. The first payload of a resumed session
carries `workspace_restore`; the CLI downloads and extracts it before printing.

Exact resume: the harness (CoffeeTerminus) writes $COFFEE_CALL_FILE (default
/tmp/.coffee_call, JSON {seq, sub, resume}) before each blocking `coffee`
command; every sim call sends the id "<seq>.<sub>" and bumps sub. With
`resume` = k set (restarted run), invocation k continues the call that was
parked at the checkpoint (register + wait: prints what it would have
printed), invocations < k were already executed and are skipped.

Output: one compact JSON line (status, time, tool result, notes), then each
observation as plain text. Outputs > 60 KB are also saved to
$COFFEE_OUT_DIR/<ticket>.json. Must stay Python 3.8+ / stdlib-only: it is
copied into the task containers as /usr/local/bin/coffee.
"""

import base64
import http.client
import io
import json
import os
import socket
import sys
import tarfile
import time
import urllib.parse

END_STATUSES = ("sim_over", "bankrupt", "segment_end", "aborted")
END_LINE = "SIMULATION OVER — set task_complete true"
LARGE_OUTPUT_BYTES = 60_000
POLL_SEC = 50


def _env(name, default=None):
    val = os.environ.get(name, default)
    if val is None:
        _die(f"{name} is not set", code=2)
    return val


def _die(msg, code=1):
    print(json.dumps({"status": "error", "error": msg}), flush=True)
    sys.exit(code)


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP over a Unix domain socket (stdlib only)."""

    def __init__(self, path, timeout):
        super().__init__("coffee", timeout=timeout)
        self._sock_path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self._sock_path)
        except OSError:
            s.close()
            raise
        self.sock = s


def _connection(timeout):
    """(connection, base path, label) for $COFFEE_SOCKET, else $COFFEE_URL."""
    sock_path = os.environ.get("COFFEE_SOCKET")
    if sock_path:
        return _UnixHTTPConnection(sock_path, timeout), "", "unix:" + sock_path
    url = _env("COFFEE_URL")
    u = urllib.parse.urlsplit(url)
    if u.scheme != "http" or not u.hostname:
        _die("COFFEE_URL must be http://host:port", code=2)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout), u.path.rstrip("/"), url


def _request(method, path, body=None, timeout=70.0, retries=30):
    """HTTP with retry on connection errors (server busy / restarting)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Authorization": "Bearer " + _env("COFFEE_TOKEN")}
    if data is not None:
        headers["Content-Type"] = "application/json"
    last_err = None
    label = None
    for attempt in range(retries):
        conn, base, label = _connection(timeout)
        try:
            conn.request(method, base + path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode() or "{}"
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {"error": raw[:200]}
            return resp.status, payload
        except (OSError, http.client.HTTPException) as e:
            last_err = e
            time.sleep(min(10.0, 1.0 + attempt))
        finally:
            conn.close()
    _die("cannot reach CoffeeBench server at %s: %s" % (label, last_err), code=3)


def _agent():
    return _env("COFFEE_AGENT_ID")


def _next_call():
    """(call id, mode) from the harness marker; mode in None|"resume"|"skip"."""
    path = os.environ.get("COFFEE_CALL_FILE", "/tmp/.coffee_call")
    try:
        with open(path) as f:
            d = json.load(f)
        sub = int(d.get("sub", 0))
        d["sub"] = sub + 1
        with open(path, "w") as f:
            json.dump(d, f)
    except (OSError, ValueError, TypeError, AttributeError):
        return None, None
    rk = d.get("resume")
    mode = None
    if rk is not None:
        mode = "skip" if sub < int(rk) else ("resume" if sub == int(rk) else None)
    return "%s.%d" % (d.get("seq"), sub), mode


def _wait(ticket, call=None):
    max_wait = float(os.environ.get("COFFEE_MAX_WAIT", "6900"))
    t0 = time.time()
    while True:
        poll = max(1, min(POLL_SEC, int(max_wait - (time.time() - t0)) + 1))
        q = {"agent": _agent(), "timeout": str(poll)}
        if call:
            q["call"] = call
        if ticket is not None:
            q["ticket"] = str(ticket)
        code, resp = _request("GET", "/wait?" + urllib.parse.urlencode(q))
        if code != 200:
            _die(f"HTTP {code}: {resp.get('error', resp)}")
        if resp.get("status") == "pending" and resp.get("upload_workspace") is not None:
            _save_workspace(resp["upload_workspace"])
            continue
        if resp.get("status") != "pending":
            return resp
        if time.time() - t0 > max_wait:
            return {
                "status": "still_waiting",
                "ticket": resp.get("ticket", ticket),
                "hint": "The other participants are still acting. Run `coffee wait` to keep waiting.",
            }


def _out_dir():
    for d in (os.environ.get("COFFEE_OUT_DIR"), "/workspace/out", os.path.expanduser("~/.coffee/out")):
        if not d:
            continue
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except OSError:
            continue
    return None


def _ws_dir():
    return os.environ.get("COFFEE_WORKSPACE", "/workspace")


def _save_workspace(day):
    """tar.gz of the workspace → POST /workspace (checkpoint `day`)."""
    root = _ws_dir()
    cap = int(float(os.environ.get("COFFEE_WS_CAP_MB", "20")) * 1024 * 1024)
    out_dir = os.path.realpath(os.environ.get("COFFEE_OUT_DIR") or os.path.join(root, "out"))
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            files.append((path, size))
    dropped = []
    if sum(sz for _, sz in files) > cap:
        keep = []
        for path, sz in files:
            if os.path.realpath(path).startswith(out_dir + os.sep):
                dropped.append(os.path.relpath(path, root))
            else:
                keep.append((path, sz))
        files = keep
    total = sum(sz for _, sz in files)
    for path, sz in sorted(files, key=lambda x: -x[1]):
        if total <= cap:
            break
        files.remove((path, sz))
        dropped.append(os.path.relpath(path, root))
        total -= sz
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, _ in files:
            try:
                tar.add(path, arcname=os.path.relpath(path, root), recursive=False)
            except OSError:
                dropped.append(os.path.relpath(path, root))
    body = {"agent": _agent(), "day": day, "data": base64.b64encode(buf.getvalue()).decode(),
            "files": len(files), "dropped": sorted(dropped)}
    _request("POST", "/workspace", body, timeout=120.0, retries=10)


def _safe_member(m, root):
    if not (m.isfile() or m.isdir()):
        return False
    dest = os.path.realpath(os.path.join(root, m.name))
    return dest == root or dest.startswith(root + os.sep)


def _restore_workspace(quiet=False):
    code, resp = _request("GET", "/workspace?" + urllib.parse.urlencode({"agent": _agent()}),
                          timeout=120.0, retries=10)
    if code != 200 or not resp.get("data"):
        return None
    root = os.path.realpath(_ws_dir())
    os.makedirs(root, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(resp["data"])), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not _safe_member(m, root):
                continue
            m.mode = (m.mode or 0o644) | 0o600
            if hasattr(tarfile, "data_filter"):
                tar.extract(m, root, filter="data")
            else:
                tar.extract(m, root)
            n += m.isfile()
    if quiet:
        return None
    return "%s restored from your previous session (%d files)" % (_ws_dir(), n)


def _print_payload(resp):
    restore = resp.pop("workspace_restore", None)
    if restore:
        note = _restore_workspace(quiet=restore == "quiet")
        if note:
            resp["notes"] = list(resp.get("notes") or []) + [note]
    head = {k: resp[k] for k in ("status", "time", "tool", "result", "notes", "message", "ticket", "hint", "error") if resp.get(k) not in (None, [], "")}
    lines = [json.dumps(head, separators=(",", ":"), ensure_ascii=False, default=str)]
    if resp.get("digest"):
        lines.append("--- resume digest ---")
        lines.append(str(resp["digest"]).rstrip())
    for obs in resp.get("observations") or []:
        lines.append("--- observation ---")
        lines.append(str(obs).rstrip())
    text = "\n".join(lines)
    if len(text.encode("utf-8")) > LARGE_OUTPUT_BYTES:
        d = _out_dir()
        if d:
            path = os.path.join(d, "%s.json" % resp.get("ticket", int(time.time())))
            with open(path, "w") as f:
                json.dump(resp, f, ensure_ascii=False, indent=1, default=str)
            text += "\n(full output also saved to %s)" % path
    print(text, flush=True)
    if resp.get("status") in END_STATUSES:
        print(END_LINE, flush=True)


def _tools():
    cache = os.path.join(os.environ.get("TMPDIR", "/tmp"), "coffee_tools_%s.json" % _agent())
    try:
        with open(cache) as f:
            return json.load(f)
    except (OSError, ValueError):
        pass
    code, resp = _request("GET", "/tools?" + urllib.parse.urlencode({"agent": _agent()}))
    if code != 200:
        _die(f"HTTP {code}: {resp.get('error', resp)}")
    tools = {t["name"]: t for t in resp.get("tools", [])}
    try:
        with open(cache, "w") as f:
            json.dump(tools, f)
    except OSError:
        pass
    return tools


def _coerce(value, schema_type):
    if schema_type == "string":
        return value
    if schema_type == "integer":
        try:
            return int(value)
        except ValueError:
            try:
                f = float(value)
                return int(f) if f.is_integer() else f
            except ValueError:
                return value
    if schema_type == "number":
        try:
            return float(value)
        except ValueError:
            return value
    if schema_type == "boolean":
        return value.lower() in ("1", "true", "yes", "y")
    try:
        return json.loads(value)
    except ValueError:
        return value


def _parse_args(tool, argv):
    # `coffee <tool> -` reads one JSON object from stdin (quoted heredoc), so
    # message text with `$`, quotes or newlines reaches the server verbatim.
    if argv == ["-"]:
        argv = [sys.stdin.read()]
    if len(argv) == 1 and argv[0].lstrip().startswith("{"):
        try:
            args = json.loads(argv[0])
        except ValueError as e:
            _die(f"invalid JSON arguments: {e}", code=2)
        if not isinstance(args, dict):
            _die("JSON arguments must be an object", code=2)
        return args
    props = (tool.get("parameters") or {}).get("properties") or {}
    args = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            _die(f"unexpected argument {tok!r}; use --key value or one JSON object", code=2)
        key = tok[2:]
        if "=" in key:
            key, val = key.split("=", 1)
            i += 1
        else:
            if i + 1 >= len(argv):
                _die(f"missing value for --{key}", code=2)
            val = argv[i + 1]
            i += 2
        key = key.replace("-", "_")
        if key not in props:
            _die(f"unknown argument --{key} for {tool['name']}; valid: {sorted(props)}", code=2)
        args[key] = _coerce(val, props[key].get("type", "string"))
    return args


USAGE = """\
coffee start                      join the simulation (blocks until it opens)
coffee <tool> --key value ...     one simulation action (blocks until your next turn)
coffee <tool> '{"key": value}'    same, JSON arguments
coffee <tool> - <<'EOF'           same, one JSON object on stdin (sent verbatim)
coffee wait                       keep waiting for a pending turn (after `still_waiting`)
coffee status                     simulation status (no simulated time)
coffee help [tool]                full descriptions of your tools / one tool
"""


def _usage_text(t):
    if t.get("usage"):
        return t["usage"]
    props = (t.get("parameters") or {}).get("properties") or {}
    req = set((t.get("parameters") or {}).get("required") or [])
    sig = " ".join(("--%s <%s>" if k in req else "[--%s <%s>]") % (k, v.get("type", "string")) for k, v in props.items())
    return "- `coffee %s %s`\n  %s" % (t["name"], sig, " ".join((t.get("description") or "").split()))


def _help(argv):
    tools = _tools()
    if argv:
        t = tools.get(argv[0])
        if t is None:
            _die(f"unknown tool {argv[0]!r}; run `coffee help`", code=2)
        print(_usage_text(t))
        return
    for t in tools.values():
        print(_usage_text(t))
    print("\nBuilt-ins: coffee start | coffee wait | coffee status | coffee help [tool]")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "help":
        _help(rest)
        return
    if cmd == "status":
        code, resp = _request("GET", "/status?" + urllib.parse.urlencode({"agent": _agent()}))
        print(json.dumps(resp, separators=(",", ":")), flush=True)
        return
    if cmd != "start" and cmd != "wait":
        tools = _tools()
        if cmd not in tools:
            _die(f"unknown tool {cmd!r}; run `coffee help` (valid: {sorted(tools)})", code=2)
        args = _parse_args(tools[cmd], rest)
    call, mode = _next_call()
    if mode == "skip":
        print(json.dumps({"status": "skipped", "note": "already executed before the run was paused"}), flush=True)
        return
    if cmd == "start" or mode == "resume":
        code, resp = _request("POST", "/register", {"agent": _agent(), "call": call})
        if code != 200:
            _die(f"HTTP {code}: {resp.get('error', resp)}")
        _print_payload(_wait(resp["ticket"], call))
        return
    if cmd == "wait":
        ticket = int(rest[0]) if rest else None
        _print_payload(_wait(ticket, call))
        return
    code, resp = _request("POST", "/act", {"agent": _agent(), "tool": cmd, "args": args, "call": call})
    if code != 200:
        _die(f"HTTP {code}: {resp.get('error', resp)}")
    _print_payload(_wait(resp["ticket"], call))


if __name__ == "__main__":
    main()
