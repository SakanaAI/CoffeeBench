"""Shared machinery of the rule-based background agents (heuristic_farmer,
heuristic_retailer).

Same contract as every model and as `heuristic_roaster_v2`: one tool call per
`query`, decisions only from the rules prompt (`system_prompt`, set by
Agent.init) and the tool results / observations in the agent's own message
history (never env internals). State is plain data, so `vars(model)` is the
checkpoint (coffeebench.server.checkpoint) and pickles.

A day is a queue of entries: ["call", tool, args, meta] or ["phase", name, {}];
a phase is expanded lazily (`_phase_<name>()`) when it reaches the head of the
queue, so it decides on the freshest view results. A new "Day N" observation
starts the morning routine; a mid-day wake never restarts it. At most
`ACTION_BUDGET` actions per day (each costs 30 virtual minutes; 20 fit in a day).
"""

from __future__ import annotations

import json
import re
from typing import Any

from coffeebench.models.types import ModelResponse, ToolCall

ACTION_BUDGET = 19
# Days before a festival in which the agents prepare (larger stock targets).
FESTIVAL_RAMP_DAYS = 3


def extract_day(text: str) -> int | None:
    m = re.search(r"\bDay\s+(\d+)\b", text or "")
    return int(m.group(1)) if m else None


def festival_phase(day: int, windows: list, ramp_days: int = FESTIVAL_RAMP_DAYS) -> str | None:
    """'festival' inside a public window, 'ramp' in the `ramp_days` before it."""
    for start, end in windows:
        if start <= day <= end:
            return "festival"
        if start - ramp_days <= day < start:
            return "ramp"
    return None


def parse_money(s: str) -> float:
    return float(s.replace(",", ""))


class RuleAgentBase:
    """Base class; subclasses set ROLE / TAG and implement `_morning_queue`
    and the phases."""

    DEFAULT_MAX_INPUT_TOKENS = 1_000_000
    ROLE = ""
    TAG = "rule"

    def __init__(self, model: str):
        self.model = model
        self.cost = 0.0
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.system_prompt = ""
        self._prompt_parsed = False

        # Public rules (parsed from the rules prompt).
        self.me: str | None = None
        self.others_by_role: dict[str, list[str]] = {}
        self.festivals: list[tuple[int, int]] = []
        self.reservation: dict[str, float] = {}
        self.start_retail: dict[str, float] = {}
        self.prod_cost: dict[str, float] = {}
        self.prod_cap: dict[str, int] = {}
        self.opex = 0.0
        self.inv_cap = 0
        self.max_days = 90

        # Weather advisories seen in observations: [start, end] windows.
        self.advisories: list[list[int]] = []
        # Published weather forecast: str(day) → demand factor.
        self.forecast: dict[str, float] = {}

        self.day = -1
        self.queue: list[list] = []
        self.actions_today = 0
        self._seq = 0
        self._last_call: dict | None = None
        self.cash = 0.0
        self.ap_total = 0.0
        self.payables: dict[str, dict] = {}
        self.paid: list[str] = []
        self.notes: list[str] = []

    # ---------- Model protocol ----------------------------------------------

    def query(self, messages, tools=None) -> ModelResponse:  # noqa: ARG002
        self.n_calls += 1
        if not self._prompt_parsed:
            self._parse_rules(self.system_prompt or "")
        tail = []
        for m in reversed(messages or []):
            if m.get("role") == "assistant":
                break
            tail.append(m)
        for m in reversed(tail):
            role = m.get("role")
            if role == "tool":
                try:
                    data = json.loads(m.get("content") or "{}")
                except (TypeError, ValueError):
                    data = {}
                self._on_result(m.get("name") or "", data if isinstance(data, dict) else {})
            elif role == "user":
                self._on_observation(m.get("content") or "")
        name, args, meta = self._next_call()
        self._seq += 1
        self._last_call = {"name": name, "args": dict(args), "meta": meta}
        if name != "wait_for_next_day":
            self.actions_today += 1
        return ModelResponse(
            content=f"({self.model})", thinking="",
            tool_calls=[ToolCall(id=f"{self.TAG}_{self._seq:05d}", name=name, input=args)],
            stop_reason="tool_use", cost=0.0, raw=None,
        )

    def get_usage_stats(self) -> dict[str, Any]:
        return {"model": self.model, "n_calls": self.n_calls, "cost": 0.0,
                "total_input_tokens": 0, "total_output_tokens": 0, "last_input_tokens": 0}

    def summarize(self, instructions: str, content: str, max_tokens: int = 4096) -> str:  # noqa: ARG002
        return content

    # ---------- rules prompt -------------------------------------------------

    def _parse_rules(self, text: str) -> None:
        from coffeebench import environment as env_mod

        self._prompt_parsed = True
        m = re.search(r"^# CoffeeBench — .*\(`(\w+)`, (\w+)\)", text, re.M)
        self.me = m.group(1) if m else None
        self.others_by_role = {}
        m = re.search(r"The other five are run by [^:]*: ([^\n]*?)\. The simulation", text)
        for aid, role in re.findall(r"(\w+) \((\w+)\)", m.group(1) if m else ""):
            self.others_by_role.setdefault(role, []).append(aid)
        m = re.search(r"The simulation lasts (\d+) days", text)
        if m:
            self.max_days = int(m.group(1))
        fest = re.findall(r"`[\w-]+` days (\d+)[–-](\d+)", text)
        self.festivals = [(int(a), int(b)) for a, b in fest] or [
            (s, e) for s, e, _m, _n in env_mod.FESTIVALS
        ]
        m = re.search(r"consumers never pay more than the reservation price \(([^)]*)\)", text)
        for item, dflt in (("roasted_coffee_kg", 30.0), ("roasted_specialty_kg", 80.0)):
            res = re.search(rf"{item} \$([\d,.]+)/kg", m.group(1)) if m else None
            self.reservation[item] = parse_money(res.group(1)) if res else dflt
            sr = re.search(rf"starting retail prices[^\n]*?{item} \$([\d,.]+)/kg", text)
            if sr:
                self.start_retail[item] = parse_money(sr.group(1))
        for item, cost, cap in re.findall(r"(green_\w+) \(\$([\d,.]+)/kg, ≤ (\d+) kg/day\)", text):
            self.prod_cost[item] = parse_money(cost)
            self.prod_cap[item] = int(cap)
        for item, dc, dcap in (("green_coffee_kg", 2.0, 30), ("green_specialty_kg", 10.0, 10)):
            self.prod_cost.setdefault(item, dc)
            self.prod_cap.setdefault(item, dcap)
        row = re.search(rf"^\| {self.ROLE} \|.*\| \$([\d,.]+) \| (\d+) kg \|$", text, re.M)
        self.opex = parse_money(row.group(1)) if row else float(
            env_mod.DAILY_OPEX_BY_ROLE.get(self.ROLE, 30.0))
        self.inv_cap = int(row.group(2)) if row else int(
            env_mod.ROLE_INVENTORY_CAP_KG.get(self.ROLE, 100))

    # ---------- observations -------------------------------------------------

    def _on_observation(self, text: str) -> None:
        day = extract_day(text)
        m = re.search(r"Cash \$(-?[\d.]+)", text)
        if m:
            self.cash = float(m.group(1))
        for a, b in re.findall(r"Weather advisory \(public\): [^\n]*? days (\d+)[–-](\d+)", text):
            w = [int(a), int(b)]
            if w not in self.advisories:
                self.advisories.append(w)
        fc = re.search(r"Weather forecast \(public[^)]*\): ([^\n]*)", text)
        if fc:
            for a, b, pct in re.findall(r"days (\d+)[–-](\d+): ([+-]\d+)% consumer", fc.group(1)):
                for d in range(int(a), int(b) + 1):
                    self.forecast[str(d)] = 1.0 + int(pct) / 100.0
        wake = re.search(r"\[Wake reason: ([^\]]*)\]", text)
        if day is not None and day != self.day:
            self.day = day
            self.actions_today = 0
            self._start_day()
            return
        if wake:
            reason = wake.group(1)
            self.notes.append(f"d{self.day}: {reason[:80]}")
            self.notes = self.notes[-50:]
            self._on_wake(reason)

    def _on_wake(self, reason: str) -> None:  # noqa: ARG002
        """Mid-day wake (offer received, deal accepted, DM, ...). Default: no
        action (the next morning's views refresh the state); DMs are ignored."""

    def _start_day(self) -> None:
        raise NotImplementedError

    def _on_result(self, name: str, data: dict) -> None:
        raise NotImplementedError

    # ---------- queue --------------------------------------------------------

    def _next_call(self) -> tuple[str, dict, dict]:
        while self.queue:
            if self.actions_today >= ACTION_BUDGET:
                self.queue = []
                break
            entry = self.queue.pop(0)
            kind, name, args = entry[0], entry[1], entry[2]
            if kind == "call":
                return name, dict(args), dict(entry[3]) if len(entry) > 3 else {}
            self.queue[0:0] = getattr(self, f"_phase_{name}")()
        return "wait_for_next_day", {}, {}

    @staticmethod
    def _call(name: str, args: dict, **meta) -> list:
        return ["call", name, args, meta]

    def _last_args(self, name: str) -> dict:
        lc = self._last_call or {}
        return lc.get("args", {}) if lc.get("name") == name else {}

    def _last_meta(self) -> dict:
        return (self._last_call or {}).get("meta") or {}

    # ---------- shared phases -----------------------------------------------

    def _phase_pay(self) -> list:
        """Pay every invoice due today, tomorrow or overdue while cash covers
        it (on time = no late fee); later invoices wait."""
        out = []
        cash = self.cash
        for r in sorted(self.payables.values(), key=lambda r: (int(r.get("due_day", 0)), str(r["id"]))):
            iid = str(r["id"])
            amt = float(r.get("outstanding") or 0)
            if iid in self.paid or amt <= 0 or int(r.get("due_day", self.day)) > self.day + 1:
                continue
            if cash >= amt:
                cash -= amt
                out.append(["call", "pay_invoice", {"invoice_id": iid}])
        return out

    def _ingest_payables(self, data: dict) -> None:
        pay = data.get("payables_due") or {}
        self.payables = {str(r["id"]): r for r in pay.get("rows") or []}

    def _on_pay_result(self, data: dict) -> None:
        args = self._last_args("pay_invoice")
        self.cash = float(data.get("remaining_cash", self.cash - float(data.get("paid_amount", 0))))
        self.paid.append(str(args.get("invoice_id")))
        self.paid = self.paid[-300:]

    def _phase_name(self) -> str | None:
        return festival_phase(self.day, self.festivals)
