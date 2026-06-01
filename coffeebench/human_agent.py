"""HumanAgent — drop-in replacement for `Agent` that defers to a human."""

from __future__ import annotations

import json
import queue
from typing import Any

from coffeebench.agent import Agent
from coffeebench.util import function_to_json


class _StubModel:
    """Mimics coffeebench.models.Model just enough for save_trajectory + Environment.run().

    `Environment.run` records `models[aid] = a.model.model` into the events
    stream so post-hoc analysis knows what was driving each agent. We tag
    human runs with `human(<agent_id>)` so they show up clearly in
    dashboards.
    """

    def __init__(self, label: str):
        self.model = f"human({label})"
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        # Anthropic-style attribute used in Agent.init(); harmless on our path.
        self.system_prompt = ""

    def query(self, messages):  # noqa: ARG002
        raise RuntimeError("HumanAgent does not call a model.")

    def get_usage_stats(self) -> dict[str, Any]:
        return {"calls": 0, "cost": 0.0}


class HumanAgent(Agent):
    """Same step() contract as Agent, but the body is a human.

    out_queue carries snapshots from agent → UI:
        {
          "type": "request",
          "agent_id": str,
          "observation": str,        # latest user-role message
          "tools": list[dict],       # JSON schemas
          "messages": list[dict],    # full message history so far
        }
    in_queue carries human → agent:
        {"action": str, "action_input": dict, "thought": str | None}
    """

    def __init__(
        self,
        *,
        name: str,
        tools: list,
        in_queue: "queue.Queue[dict]",
        out_queue: "queue.Queue[dict]",
        initial_observation: str = "Observation: Day 0.",
    ):
        # Skip Agent.__init__ — it expects a real Model and builds a system
        # prompt we don't need. We re-implement just the bits Environment
        # depends on.
        self.name = name
        self.tools = tools
        self.tools_json = [function_to_json(t) for t in tools]
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.messages: list[dict] = []
        self.system_prompt = ""
        self.user_prompt = initial_observation
        self.model = _StubModel(name)

    def init(self) -> None:
        """Seed the user prompt — same role/content shape an LLM agent receives."""
        self.add_message("user", self.user_prompt)

    def step(self) -> dict:
        """Push state to the UI, block on the human's reply, execute the tool.

        We do NOT retry on FormatError (the UI sends structured input that's
        already a valid action dict — there's no parse step to fail on).
        """
        last_user = next(
            (m for m in reversed(self.messages) if m["role"] == "user"),
            None,
        )
        observation_text = last_user["content"] if last_user else ""
        self.out_queue.put(
            {
                "type": "request",
                "agent_id": self.name,
                "observation": observation_text,
                "tools": self.tools_json,
                "messages": list(self.messages),
            }
        )

        payload = self.in_queue.get()
        action = {
            "action": payload["action"],
            "action_input": payload.get("action_input") or {},
        }
        thought = (payload.get("thought") or "").strip()

        rendered = (
            (f"Thought: {thought}\n\n" if thought else "")
            + "Action:\n"
            + json.dumps(action, indent=2)
            + "<end_action>"
        )
        self.add_message("assistant", content=rendered)
        return self.get_observation(action)
