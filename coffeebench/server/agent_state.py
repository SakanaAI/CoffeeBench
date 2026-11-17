"""Exact-resume bookkeeping shared by the server, the launcher and CoffeeTerminus.

CoffeeTerminus (Harbor host process) writes one snapshot of its full LLM
conversation state just before every blocking `coffee` command:

    <run_dir>/agent_state/<agent>/call_<SEQ>.json

and tells the CLI the call id `<SEQ>.<k>` (k = k-th `coffee` invocation in
that command) through a marker file in the container. The CLI sends the call
id with /register, /act and /wait; the server keeps the id of the call that
is parked on each agent's open ticket (`RemoteAgent.pending_call`) and stores
it in every checkpoint. On `--resume <ckpt>` the restored agent's
`resume_call` names the snapshot to continue from.

`checkpoints/refs.json` = {checkpoint file name: {"day": D, "calls": {agent:
call id | None}}} (sidecar, so GC and the launcher need not unpickle
checkpoints). Garbage collection at each daily checkpoint: an agent's
snapshot is kept if a retained checkpoint references it or it is not older
than the call referenced by latest.pkl (later calls may still become
checkpoint points); a workspace tarball (workspaces/<agent>/day_NNN.tar.gz)
is kept if a retained checkpoint is at that day or it is not older than
latest.pkl.
"""

from __future__ import annotations

import json
import os

REFS_FILE = "refs.json"


def call_seq(call: str | None) -> int | None:
    """`"<seq>.<sub>"` → seq (None if absent / malformed)."""
    if not call:
        return None
    try:
        return int(str(call).split(".", 1)[0])
    except ValueError:
        return None


def call_sub(call: str | None) -> int:
    try:
        return int(str(call).split(".", 1)[1])
    except (IndexError, ValueError):
        return 0


def snapshot_path(state_dir: str, aid: str, seq: int) -> str:
    return os.path.join(state_dir, aid, f"call_{seq:06d}.json")


def load_refs(ckpt_dir: str) -> dict[str, dict[str, str | None]]:
    try:
        with open(os.path.join(ckpt_dir, REFS_FILE)) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def update_refs(ckpt_dir: str, name: str, entry: dict) -> None:
    """entry = {"day": D, "calls": {agent: call id | None}}."""
    os.makedirs(ckpt_dir, exist_ok=True)
    data = load_refs(ckpt_dir)
    data[name] = {"day": int(entry["day"]), "calls": dict(entry["calls"])}
    path = os.path.join(ckpt_dir, REFS_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def _listdir(d: str) -> list[str]:
    try:
        return os.listdir(d)
    except OSError:
        return []


def gc_workspaces(ws_dir: str, ckpt_dir: str) -> int:
    refs = load_refs(ckpt_dir)
    if "latest.pkl" not in refs:
        return 0
    floor = int(refs["latest.pkl"]["day"])
    keep = {int(r["day"]) for r in refs.values()}
    n = 0
    for aid in _listdir(ws_dir):
        d = os.path.join(ws_dir, aid)
        for fn in _listdir(d):
            if not (fn.startswith("day_") and fn.endswith(".tar.gz")):
                continue
            try:
                day = int(fn[4:-7])
            except ValueError:
                continue
            if day < floor and day not in keep:
                try:
                    os.remove(os.path.join(d, fn))
                    n += 1
                except OSError:
                    pass
    return n


def gc_snapshots(state_dir: str, ckpt_dir: str) -> int:
    """Delete snapshots no retained checkpoint can need. Returns #deleted."""
    refs = load_refs(ckpt_dir)
    latest = (refs.get("latest.pkl") or {}).get("calls") or {}
    n = 0
    for aid in _listdir(state_dir):
        floor = call_seq(latest.get(aid))
        if floor is None:
            continue  # unknown (timed-out agent / no snapshots yet): keep everything
        keep = {call_seq((r.get("calls") or {}).get(aid)) for r in refs.values()}
        d = os.path.join(state_dir, aid)
        for fn in _listdir(d):
            if not (fn.startswith("call_") and fn.endswith(".json")):
                continue
            try:
                seq = int(fn[5:-5])
            except ValueError:
                continue
            if seq < floor and seq not in keep:
                try:
                    os.remove(os.path.join(d, fn))
                    n += 1
                except OSError:
                    pass
    return n
