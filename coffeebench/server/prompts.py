"""Per-agent instructions for the Harbor (CLI-driven) harness.

instruction.md = rules prompt sections 1-9 (`coffeebench.rules_prompt`, the
same text the in-process harness uses) + section 10 `CLI_GUIDE` (terminal
interface) + section 11 (complete tool list, rendered by `tool_usage`; the
same text `coffee help` prints, served by `GET /tools` as `usage`).
"""

from __future__ import annotations

import inspect
import re

from coffeebench import mechanics as mech_mod
from coffeebench.config import RunConfig
from coffeebench.event_loop import tool_cost_minutes
from coffeebench.main import _build_score_framing, _seed_world
from coffeebench.models.types import ToolSpec
from coffeebench.rules_prompt import build_rules_prompt

# Uniform context budget of every Harbor agent (job templates:
# `model_info.max_input_tokens`); CoffeeTerminus summarizes at 85% of it
# (`proactive_summarization_threshold: 150000`).
CONTEXT_TOKENS = 1_000_000


def build_system_prompts(config: RunConfig, max_days: int | None = None) -> dict[str, dict]:
    """`{agent_id: {"role", "display_name", "prompt"}}` for every agent
    (rules sections 1-9)."""
    max_days = int(max_days if max_days is not None else config.max_days)
    items, endowments = _seed_world(config.economy.get("initial_cash"))
    if "p_res" in config.economy:
        for it in items:
            if it.retail_reservation_price is not None:
                it.retail_reservation_price = float(config.economy["p_res"])
    out: dict[str, dict] = {}
    for e in endowments:
        prompt = build_rules_prompt(
            e,
            endowments,
            items,
            max_days,
            _build_score_framing(config.kpi.get(e.agent_id), max_days=max_days),
            mechanics=mech_mod.parse(config.mechanics),
        )
        out[e.agent_id] = {"role": e.role, "display_name": e.display_name, "prompt": prompt}
    return out


def _signature(name: str, schema: dict) -> str:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    args = " ".join(
        f"--{k} <{v.get('type', 'string')}>" if k in required else f"[--{k} <{v.get('type', 'string')}>]"
        for k, v in props.items()
    )
    return f"coffee {name}{(' ' + args) if args else ''}"


def tool_usage(name: str, description: str, schema: dict) -> str:
    """Complete human-readable entry for one tool: signature, description
    (one paragraph) and one line per argument (from the docstring's Args)."""
    doc = inspect.cleandoc(description or "")
    body, _, args_block = doc.partition("\nArgs:")
    lines = [f"- `{_signature(name, schema)}`", "  " + " ".join(body.split())]
    args: list[list[str]] = []
    for raw in args_block.splitlines():
        m = re.match(r"^\s*(\w+)(\s*\([^)]*\))?:\s*(.*)$", raw)
        if m and m.group(1) in schema.get("properties", {}):
            args.append([m.group(1), m.group(3)])
        elif args and raw.strip():
            args[-1][1] += " " + raw.strip()
    lines += [f"  --{k}: {' '.join(v.split())}" for k, v in args]
    return "\n".join(lines)


def tool_catalog(tools: list) -> list[dict]:
    """JSON-able tool list (name, description, parameters, cost, usage) —
    served by `GET /tools` and embedded in the task instruction."""
    out = []
    for t in tools:
        spec = ToolSpec.from_function(t)
        out.append(
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
                "cost_minutes": tool_cost_minutes(spec.name),
                "usage": tool_usage(spec.name, spec.description, spec.input_schema),
            }
        )
    return out


def format_tool_list(catalog: list[dict]) -> str:
    return "\n".join(t["usage"] for t in catalog)


CLI_GUIDE = """\
## 10. Interface: the `coffee` command
You work in a Linux terminal without network access. You are `{agent_id}`; every simulation action is a shell command `coffee <tool> [--arg value ...]`.
1. `coffee start` joins; it blocks until all six businesses have joined and prints your day-0 observation.
2. Each `coffee <tool>` is one action. It blocks until your next turn (often minutes of wall time), then prints a JSON line with the result and every observation since (`--- observation ---`, wake reasons). On `still_waiting`, `coffee wait` keeps waiting for the same turn.
3. A response may contain several `coffee` commands; they run in order, one action each. Never run two concurrently (HTTP 409).
4. Arguments: `--key value` or one JSON object (`coffee make_offer '{{"listing_id": "L3", "offered_price": 6.5, "qty": 20}}'`), or JSON on stdin via a quoted heredoc (`coffee send_message - <<'EOF'` … `EOF`, sent verbatim). Bash expands `$` inside double quotes. `coffee help [tool]` shows the tool descriptions.
5. Other shell commands cost no simulated time. `/workspace` persists for the whole run (20 MB cap, `/workspace/out` dropped first); `coffee` output over 60 KB is also saved in `/workspace/out/`.
6. A response without a `coffee` command does not advance simulated time; with no `coffee` command within 30 minutes of wall time after an observation, the turn counts as `wait_for_next_day`.
7. Your context is limited to {context_tokens:,} tokens; when nearly full, you summarize the session and continue from that summary and the current screen.
8. The run may be paused and resumed; you keep your context and `/workspace`, but background processes and shell variables may be lost. A `--- resume digest ---` means your context was lost; it summarizes your books, open deals, invoices, message threads and recent actions.
9. When a `coffee` command prints `SIMULATION OVER — set task_complete true`, your part has ended: set `task_complete` true. Before that, `task_complete` is ignored; there is nothing to submit.
10. Act on the market only through `coffee`. Inspecting the evaluation harness (the `coffee` client's internals, the server, the host) is not allowed.
"""


def build_instruction(agent_id: str, prompt: str, catalog: list[dict], max_days: int) -> str:
    """Harbor `instruction.md`: rules (1-9) + CLI interface (10) + tools (11).
    `max_days` is part of the rules text already (kept for the call signature)."""
    return (
        prompt.strip()
        + "\n\n"
        + CLI_GUIDE.format(agent_id=agent_id, context_tokens=CONTEXT_TOKENS)
        + "\n## 11. Tools\n"
        + format_tool_list(catalog)
        + "\n"
    )
