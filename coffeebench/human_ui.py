"""Human-in-the-loop UI for CoffeeBench."""

from __future__ import annotations

import argparse
import inspect
import queue
import sys
import threading
import time
import traceback
from typing import Any

import streamlit as st

from coffeebench.main import build_run


# ---------------------------------------------------------------------------
# Argv parsing — Streamlit puts script args after `--`.
# ---------------------------------------------------------------------------
def _parse_argv() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--human",
        default="roaster_A",
        help="agent_id to human-control (default roaster_A).",
    )
    p.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="LLM driving the OTHER (non-human) agents.",
    )
    p.add_argument(
        "--models",
        default=None,
        help='Per-agent overrides "agent_id:model_name" '
        "(only applies to non-human agents).",
    )
    p.add_argument("--max_days", type=int, default=90)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    return p.parse_args(sys.argv[1:])


# ---------------------------------------------------------------------------
# Env lifecycle — `@st.cache_resource` guarantees this body executes EXACTLY
# ONCE per (args_key) across all Streamlit reruns and browser sessions.
# Without this, the module-level `_GLOBAL = {...}` line would be re-executed
# on every rerun, spawning a new env thread each time.
# ---------------------------------------------------------------------------
def _args_key(args: argparse.Namespace) -> tuple:
    """Hashable key so cache_resource dedupes only across compatible runs."""
    return (
        args.human,
        args.model,
        args.models,
        args.max_days,
        args.seed,
        args.output,
    )


@st.cache_resource(show_spinner=False)
def _start_env(_args_key_tuple: tuple) -> dict[str, Any]:
    """Build the env, kick off the runner thread, return shared handles.

    The returned dict is THE singleton for this session — Streamlit caches
    it by `_args_key_tuple`. Mutating fields (`done`, `crash`, `latest_request`)
    is how the page communicates with the runner thread across reruns.
    """
    args = _parse_argv()
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    state: dict[str, Any] = {
        "args": args,
        "human_id": args.human,
        "in_q": in_q,
        "out_q": out_q,
        "env": None,
        "output_path": None,
        "events_path": None,
        "thread": None,
        "done": False,
        "crash": None,
        "latest_request": None,  # last out_q item the page rendered
    }

    print(
        f"[human_ui] building env (human={args.human}, max_days={args.max_days})...",
        flush=True,
    )
    try:
        env, output_path, events_path = build_run(
            args, human_agents={args.human: (in_q, out_q)}
        )
    except BaseException as exc:  # noqa: BLE001
        state["crash"] = f"build_run: {type(exc).__name__}: {exc}"
        print(f"[human_ui] build_run crashed: {exc}", flush=True)
        traceback.print_exc()
        return state

    state["env"] = env
    state["output_path"] = output_path
    state["events_path"] = events_path
    print(
        f"[human_ui] env built, agents={list(env.agents.keys())}, "
        f"trader class={type(env.agents[args.human]).__name__}",
        flush=True,
    )

    def _runner() -> None:
        import asyncio

        try:
            print("[human_ui] env.run() starting…", flush=True)
            asyncio.run(env.run())
            print("[human_ui] env.run() returned normally.", flush=True)
        except BaseException as exc:  # noqa: BLE001
            state["crash"] = f"env.run: {type(exc).__name__}: {exc}"
            print(f"[human_ui] env crashed: {exc}", flush=True)
            traceback.print_exc()
        finally:
            try:
                env.save_trajectory(output_path)
            except Exception as save_exc:  # noqa: BLE001
                print(f"[human_ui] save failed: {save_exc}", flush=True)
            state["done"] = True
            out_q.put({"type": "done"})

    t = threading.Thread(target=_runner, daemon=True, name="coffeebench-env")
    t.start()
    state["thread"] = t
    return state


def _drain_out_q(state: dict[str, Any]) -> None:
    """Move queued requests from out_q into state['latest_request']."""
    out_q: queue.Queue | None = state.get("out_q")
    if out_q is None:
        return
    while True:
        try:
            msg = out_q.get_nowait()
        except queue.Empty:
            break
        if msg.get("type") == "done":
            state["done"] = True
        else:
            state["latest_request"] = msg


# ---------------------------------------------------------------------------
# Tool-form rendering
# ---------------------------------------------------------------------------
_LONG_TEXT_PARAMS = {"body", "memo", "notes", "message", "reason"}


def _is_optional(tool_fn, param_name: str) -> bool:
    try:
        sig = inspect.signature(tool_fn)
        return sig.parameters[param_name].default is not inspect._empty
    except (KeyError, ValueError):
        return False


def _render_param_widget(form_key: str, pname: str, ptype: str, optional: bool):
    label = pname + ("" if not optional else "  (optional)")
    widget_key = f"{form_key}__{pname}"
    if ptype == "integer":
        return st.number_input(label, value=0, step=1, key=widget_key)
    if ptype == "number":
        return st.number_input(label, value=0.0, key=widget_key)
    if ptype == "boolean":
        return st.checkbox(label, key=widget_key)
    if pname in _LONG_TEXT_PARAMS:
        return st.text_area(label, key=widget_key, height=80)
    return st.text_input(label, key=widget_key)


def _render_action_form(state: dict[str, Any], req: dict) -> None:
    env = state["env"]
    human_id = state["human_id"]
    tools = req["tools"]
    tool_names = [t["function"]["name"] for t in tools]

    ba = env.business_apps[human_id]
    fn_lookup = {n: getattr(ba, n, None) for n in tool_names}

    chosen = st.selectbox(
        "Tool",
        tool_names,
        index=tool_names.index("view_listings") if "view_listings" in tool_names else 0,
        key="tool_picker",
    )
    chosen_idx = tool_names.index(chosen)
    schema = tools[chosen_idx]["function"]
    st.caption(schema.get("description", "").strip().split("\n")[0][:300])

    params = schema["parameters"]["properties"]
    required = set(schema["parameters"]["required"])

    form_key = f"form__{chosen}"
    with st.form(form_key, clear_on_submit=True):
        inputs: dict[str, Any] = {}
        for pname, pinfo in params.items():
            optional = pname not in required
            inputs[pname] = _render_param_widget(
                form_key, pname, pinfo.get("type", "string"), optional
            )
        thought = st.text_area(
            "Thought (optional, recorded in trajectory for analysis)",
            key=f"thought__{chosen}",
            height=80,
        )
        submitted = st.form_submit_button(f"Submit `{chosen}`")

    if not submitted:
        return

    action_input: dict[str, Any] = {}
    fn = fn_lookup.get(chosen)
    for pname, val in inputs.items():
        if pname in required:
            action_input[pname] = val
            continue
        if isinstance(val, str) and val == "":
            if fn is None or _is_optional(fn, pname):
                continue
        action_input[pname] = val

    state["in_q"].put(
        {"action": chosen, "action_input": action_input, "thought": thought}
    )
    # Optimistically clear the rendered request so the UI shows a
    # "processing" state until the env emits the next observation.
    state["latest_request"] = None
    time.sleep(0.2)
    st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CoffeeBench — Human Baseline",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _format_role_message(m: dict) -> tuple[str, str]:
    role = m.get("role", "?")
    content = m.get("content") or ""
    if role == "user":
        return ("Observation (env -> you)", content)
    if role == "assistant":
        return ("Your action", content)
    return (role, content)


def main() -> None:
    args = _parse_argv()
    state = _start_env(_args_key(args))
    _drain_out_q(state)

    human_id = state["human_id"]
    req = state["latest_request"]

    # ----- Sidebar -----
    with st.sidebar:
        st.markdown("### Human-baseline run")
        st.markdown(f"**You control**: `{human_id}`")
        st.caption(f"max_days = {args.max_days}")
        st.caption(f"non-human model = `{args.model}`")
        if args.seed is not None:
            st.caption(f"seed = {args.seed}")
        st.caption(f"events: `{state.get('events_path')}`")
        st.caption(f"trajectory: `{state.get('output_path')}`")
        st.divider()
        st.markdown(
            "**Live dashboard**: open another tab and run "
            "`uv run streamlit run coffeebench/web.py` to watch the events file."
        )
        st.divider()
        st.markdown("### debug")
        thr = state.get("thread")
        st.caption(f"env_thread alive = {thr.is_alive() if thr else 'no thread'}")
        out_q = state.get("out_q")
        st.caption(f"out_q size = {out_q.qsize() if out_q is not None else '—'}")
        st.caption(f"crash = {state.get('crash') or '—'}")
        st.caption(f"done = {state['done']}")
        st.caption(f"latest_request? = {req is not None}")
        if st.button("Reset cached env (dev only)"):
            st.cache_resource.clear()
            st.rerun()

    # ----- Header -----
    cols = st.columns([3, 2, 2, 2])
    with cols[0]:
        st.markdown(f"### `{human_id}` — human in the loop")
    if req is not None:
        msgs = req.get("messages", [])
        day_str = "—"
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content", "")
                if c.startswith("Observation: Day "):
                    day_str = c.split("\n", 1)[0].replace("Observation: ", "")
                    break
        with cols[1]:
            st.metric("Day", day_str)
        with cols[2]:
            st.metric("Messages", len(msgs))
    with cols[3]:
        if state["done"]:
            st.success("FINISHED")
        elif state["crash"]:
            st.error("CRASHED")
        elif req is not None:
            st.info("YOUR TURN")
        else:
            st.warning("ENV PROCESSING…")

    if state["crash"]:
        st.error(f"Env thread crashed: {state['crash']}. See terminal logs.")
        st.stop()

    if state["done"]:
        st.success(
            f"Run finished. Trajectory saved to `{state['output_path']}`. "
            "Run `uv run python -m coffeebench.show --latest --plot` to inspect."
        )
        st.stop()

    if req is None:
        st.info(
            "Waiting for the next observation. The other agents are taking "
            "their turns and end-of-day mechanics are running."
        )
        time.sleep(1.0)
        st.rerun()
        return

    # ----- Latest observation -----
    st.markdown("## Latest observation")
    st.code(req["observation"], language="markdown")

    # ----- Message history -----
    with st.expander(
        f"Message history ({len(req['messages'])} messages — same view an LLM agent would have)",
        expanded=False,
    ):
        for m in req["messages"][-40:]:
            role_label, content = _format_role_message(m)
            st.markdown(f"**{role_label}**  · _{m.get('timestamp', '')}_")
            st.code(content[:4000] + ("…" if len(content) > 4000 else ""))

    st.divider()

    # ----- Action form -----
    st.markdown("## Take action")
    _render_action_form(state, req)


if __name__ == "__main__":
    main()
