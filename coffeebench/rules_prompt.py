"""Rules-only agent prompt (camera-ready rewrite, 2026-10-18).

The prompt states the rules of the world (mechanics, scoring, interface) once
each and prescribes no behaviour, strategy, cadence or reasoning style. Every
number is read from the env's module constants / the seeded world at build
time, so `[economy]` overrides and the run's festival schedule
(`[mechanics]` festivals, else `environment.FESTIVALS`) stay in sync with what
the agent is told.

Sections 1-9 (`build_rules_prompt`) are shared by both harnesses. Section 10
(interface) and 11 (tool list) differ: the Harbor/CLI harness appends
`coffeebench.server.prompts.CLI_GUIDE` + the rendered tool list; the legacy
in-process harness (native tool calling) appends `LEGACY_INTERFACE` and sends
the tool schemas through the provider API.
"""

from __future__ import annotations

from coffeebench import environment as env_mod
from coffeebench.typings import AgentEndowment, Item

ROLE_ORDER = ("farmer", "roaster", "retailer")


def _money(x: float) -> str:
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


def _days(n: int) -> str:
    return f"{n} day" if n == 1 else f"{n} days"


def _role_table(items: list[Item]) -> str:
    produced = [it for it in items if it.produced_by_role == "farmer"]
    opex = env_mod.DAILY_OPEX_BY_ROLE
    cap = env_mod.ROLE_INVENTORY_CAP_KG
    rows = [
        "| role | what only this role can do | daily fixed opex | inventory cap |",
        "|---|---|---|---|",
    ]
    lags = {it.production_lag_days for it in produced}
    farm = "`coffee produce_item` " + " or ".join(
        f"{it.id} ({_money(it.production_cost_per_unit)}/kg, ≤ {it.daily_production_cap} kg/day)"
        for it in produced
    ) + (f"; output lands {_days(lags.pop())} later" if len(lags) == 1 else "")
    recipes = env_mod.ROAST_RECIPES
    rlags = {r["lag_days"] for r in recipes.values()}
    roast = "`coffee roast` " + " or ".join(
        f"{green} → {r['output_item']} ({_money(r['labor_cost_per_kg'])}/kg green labor, yield {r['yield']})"
        for green, r in recipes.items()
    ) + (
        f"; ≤ {env_mod.ROAST_DAILY_CAP_GREEN_KG} kg green input/day in total; "
        "floor(input kg × yield) kg land "
        + (f"{_days(rlags.pop())} later" if len(rlags) == 1 else "after the recipe's lag")
    )
    retail = "sell roasted coffee to consumers at prices set with `coffee set_retail_price`"
    what = {"farmer": farm, "roaster": roast, "retailer": retail}
    for role in ROLE_ORDER:
        rows.append(
            f"| {role} | {what[role]} | {_money(opex[role])} | {cap[role]} kg |"
        )
    return "\n".join(rows)


def _starter_lines(endowments: list[AgentEndowment]) -> str:
    seen: dict[str, AgentEndowment] = {}
    for e in endowments:
        seen.setdefault(e.role, e)
    parts = []
    for role in ROLE_ORDER:
        e = seen.get(role)
        if e is None:
            continue
        stock = ", ".join(
            f"{q} kg {item} at cost {_money(e.initial_cost_basis.get(item, 0.0))}/kg"
            for item, q in e.initial_inventory.items()
        )
        prices = ""
        if e.initial_retail_prices:
            prices = "; starting retail prices " + ", ".join(
                f"{item} {_money(p)}/kg" for item, p in e.initial_retail_prices.items()
            )
        parts.append(f"  - {role}: {_money(e.initial_cash)} cash, {stock}{prices}")
    return "\n".join(parts)


def _festival_lines(max_days: int, festivals: list | None = None) -> str:
    fests = [f for f in (env_mod.FESTIVALS if festivals is None else festivals) if f[0] < max_days]
    if not fests:
        return "none in this run"
    return "; ".join(
        f"`{name}` days {start}–{min(end, max_days - 1)}" for start, end, _mult, name in fests
    )


SCORE_FRAMING_DEFAULT = """\
Your **score** is your net income over the whole run (days 0–{last_day}), computed by the simulator from its own ledger:

    net_income = revenue − COGS − operating expenses − interest expense + interest revenue

- revenue: B2B sales booked when the goods are delivered, plus consumer sales; returns are deducted.
- COGS: the cost of the lots that left your inventory through those sales (lot cost = production cost, purchase price, or green-bean cost + roasting labor, per kg of output).
- operating expenses: daily fixed opex; spoilage (expired lots); bad debts; goods lost in transit; write-downs.
- interest: late fees on overdue invoices (paid = expense, received = revenue).
Cash, inventory and receivables enter the score only through these terms. Bankrupt businesses are ranked DNF."""


def build_rules_prompt(
    endow: AgentEndowment,
    endowments: list[AgentEndowment],
    items: list[Item],
    max_days: int,
    score_framing: str,
    mechanics: dict | None = None,
) -> str:
    """Sections 1-9 of the agent prompt (shared by both harnesses).

    `mechanics` = `coffeebench.mechanics.parse([mechanics])` of the run: its
    festival schedule replaces environment.FESTIVALS; `announce_advisories`
    adds that weather advisories appear in the observation and
    `disclose_demand_shift` that demand can shift without notice. Without a
    `[mechanics]` table the text is unchanged."""
    from coffeebench import mechanics as mech_mod

    mechanics = mechanics or {}
    festivals = mech_mod.festivals(mechanics, env_mod.FESTIVALS)
    demand_note = (
        "\n  - consumer demand can shift over time without notice."
        if mechanics.get("disclose_demand_shift")
        else ""
    )
    advisory_note = (
        "\n- Public weather advisories, when issued, appear in the morning observation "
        "(dates and expected effect on farmers' production capacity / cost)."
        if mechanics.get("announce_advisories")
        else ""
    )
    traits = mechanics.get("traits") or {}
    keys = {k for t in traits.values() for k in t}
    farm_note = (
        "\n\nProduction cost and daily capacity can differ between farms (not disclosed); "
        "`coffee produce_item` reports yours."
        if keys & {"production_cost_factor", "production_cap_factor"}
        else ""
    )
    up = mechanics.get("roaster_upgrade")
    upgrade_note = (
        f"\n\nRoaster upgrade: a roaster may call `coffee buy_roaster_upgrade` once. It pays "
        f"{_money(up['capex'])} in cash at once, booked in full as an operating expense; from "
        f"{_days(up['build_days'])} later that roaster's green roasting cap is "
        f"{up['roast_cap_green_kg']} kg/day and its inventory cap {up['inventory_cap_bonus_kg']} kg "
        "higher, for the rest of the run."
        if up
        else ""
    )
    el = mechanics.get("end_lcnrv")
    if el and el.get("nrv") == "market":
        lcnrv_note = (
            "\n\nEnd-of-run valuation: after the last day, everything you own — inventory lots, output of "
            "roasting / production still in progress, and goods sold but not yet delivered — is valued at "
            "the lower of its cost and its net realisable value: the median price of arm's-length trades "
            f"over the last {_days(el.get('window_days', 14))} (roasted items bought by retailers, green items "
            "sold by farmers; trades between two businesses that also traded with each other in the opposite "
            "direction are ignored), capped at "
            + ", ".join(f"{k} {_money(v)}" for k, v in el["ceilings"].items())
            + "; with no such trade in the window a roasted item is valued at its cap and a green item at "
            "the roasted value × roast yield − roast labour (or its cap). The write-down is an operating "
            "expense and reduces net income."
        )
    elif el:
        lcnrv_note = (
            "\n\nEnd-of-run valuation: after the last day, every kg you own whose unit cost exceeds a fixed "
            "ceiling per kg — inventory lots, output of roasting / production still in progress, and goods "
            "sold but not yet delivered — is valued at that ceiling ("
            + ", ".join(f"{k} {_money(v)}" for k, v in el["ceilings"].items())
            + "); the write-down is an operating expense and reduces net income."
        )
    else:
        lcnrv_note = ""
    if mechanics.get("end_credit_loss"):
        lcnrv_note += (
            "\n\nEnd-of-run credit losses: after the last day, receivables from a business whose assets "
            "(cash, inventory, goods in progress and its own receivables) do not cover all its unpaid "
            "invoices are written down in proportion to the shortfall (an operating expense of the seller)."
        )
    ci = mechanics.get("cash_interest")
    if ci:
        lcnrv_note += (
            f"\n\nInterest on cash: every day at 19:00 a positive cash balance earns "
            f"{ci['annual_rate'] * 100:g}% per year (÷365), booked as interest revenue."
        )
    ce = mechanics.get("credit_enforcement")
    if ce:
        lcnrv_note += (
            f"\n\nDebt collection: each morning, an invoice more than {_days(ce['overdue_days'])} past its due "
            "date is collected from the buyer's cash (with late fees); a buyer that cannot pay it is declared "
            "bankrupt."
        )
    ship_note = (
        "\n- Businesses differ in shipping reliability (not disclosed): how often and how long "
        "their shipments are delayed, and whether a shipment arrives short (fewer kg than agreed; "
        "the seller loses the missing goods and the invoice covers only the kg delivered). "
        "Delays and short shipments are recorded in the deal's notes (`coffee view_deals`)."
        if keys & {"delivery_delay_prob", "delivery_delay_days", "short_ship_prob"}
        else ""
    )
    dp = mechanics.get("demand_patterns")
    pattern_note = (
        "\n  - demand follows a fixed weekly pattern and a gradual trend over the run (sizes not "
        "disclosed; visible in sales data), and responds to the weather: an accurate weather "
        f"forecast for the next {_days(dp.get('forecast_days', 3))} (event, days, % change in "
        "consumer demand) is published in the morning observation and in "
        "`coffee view_market_aggregate`."
        if dp
        else ""
    )
    g = mechanics.get("availability_growth")
    growth_note = (
        f"\n  - a retailer's regular-customer floor for an item grows by {g['kg_per_day']:g} kg/day "
        f"for every consecutive day the item ends the day still in stock there at or below its "
        f"reservation price (up to +{g['max_kg']:g} kg/day); a day that ends sold out, without the item on "
        "sale or priced above the reservation price resets it."
        if g
        else ""
    )
    # v2.1: keep the score definition and the §6/§9 texts consistent with the
    # optional accounting / finance mechanics.
    if mechanics.get("cash_interest"):
        score_framing = score_framing.replace(
            "- interest: late fees on overdue invoices (paid = expense, received = revenue).",
            "- interest: late fees on overdue invoices (paid = expense, received = revenue) and interest "
            "earned on your cash balance (revenue).")
    if mechanics.get("end_credit_loss"):
        score_framing = score_framing.replace("bad debts;", "bad debts (incl. end-of-run credit-loss allowances);")
    if mechanics.get("end_lcnrv"):
        score_framing = score_framing.replace("write-downs.", "write-downs (incl. the end-of-run valuation).")
    ce = mechanics.get("credit_enforcement")
    collect_txt = (
        f"if it is still unpaid more than {_days(ce['overdue_days'])} after its due day, it is collected from "
        "the buyer's cash the next morning (see Debt collection in §3)"
        if ce else "nothing is collected automatically"
    )
    bankrupt_extra = (
        f" You are also bankrupt if an invoice of yours is more than {_days(ce['overdue_days'])} overdue and your "
        "cash cannot pay it when it is collected."
        if ce else ""
    )
    eod_txt = ("consumer sales, expiry write-offs, late fees, interest on cash, offer expiry"
               if mechanics.get("cash_interest") else "consumer sales, expiry write-offs, late fees, offer expiry")
    last_day = max_days - 1
    others = ", ".join(
        f"{e.agent_id} ({e.role})" for e in endowments if e.agent_id != endow.agent_id
    )
    consumer = [it for it in items if it.retail_reservation_price is not None]
    res = ", ".join(f"{it.id} {_money(it.retail_reservation_price)}/kg" for it in consumer)
    shelf = env_mod.SHELF_LIFE_DAYS
    groups = {
        kind: {v for k, v in shelf.items() if k.startswith(kind + "_")}
        for kind in ("green", "roasted")
    }
    if all(len(g) == 1 for g in groups.values()) and len(shelf) == 4:
        shelf_txt = ", ".join(f"{kind} {_days(g.pop())}" for kind, g in groups.items())
    else:
        shelf_txt = ", ".join(f"{k} {_days(v)}" for k, v in shelf.items())
    lags = sorted({it.delivery_lag_days for it in items})
    lag_txt = _days(lags[0]) if len(lags) == 1 else "/".join(map(str, lags)) + " days"
    late = env_mod.LATE_FEE_PER_DAY * 100
    return f"""\
# CoffeeBench — {endow.display_name} (`{endow.agent_id}`, {endow.role})

## 1. Setting
You run {endow.display_name}, a coffee {endow.role}, one of six businesses in a simulated B2B coffee market: two farmers, two roasters, two retailers. The other five are run by other autonomous agents: {others}. The simulation lasts {max_days} days (day 0 to day {last_day}). There is no human user.

## 2. Score
{score_framing}

## 3. Roles, items, starting position
Items (all traded per kg): green_coffee_kg, green_specialty_kg, roasted_coffee_kg, roasted_specialty_kg. Only roasted items can be sold to consumers; consumers never pay more than the reservation price ({res}).

{_role_table(items)}{farm_note}{upgrade_note}{lcnrv_note}

Production and roasting costs are paid in cash when the command is issued (it is rejected if cash does not cover it); daily caps reset each morning. Any business may buy and sell any item it holds; trade direction is not restricted by role.

Starting position (each item is one lot produced on day 0):
{_starter_lines(endowments)}

## 4. Market
- `coffee post_listing` publishes a listing (item, qty ≤ your current stock, asking price per kg, requested payment terms). It does not reserve stock and shows buyers the expiry range of the lots that would have shipped at posting time.
- Any other business may `coffee make_offer` on an open listing: price per kg, qty ≤ remaining listing qty, payment terms 0–90 days (default 30). There are no counter-offers.
- The seller may `coffee accept_offer` (binding deal at the offer's price and terms; the listing's terms are only a request), `coffee reject_offer`, or `coffee cancel_listing` (its pending offers are rejected). The buyer may `coffee withdraw_offer` while pending. Acceptance reduces the listing qty; at 0 the listing closes and its other offers are rejected.
- A pending offer expires at 19:00 on the day after it was posted.
- `coffee send_message` sends a private message (title ≤ 80 chars, one line; body ≤ 4,000 chars) to any business; `--reply_to_message_id` threads replies. Messages are not binding.
- Visibility: listings and the consumer market aggregate (`coffee view_market_aggregate`) are public; you see only your own offers, deals, invoices and messages.

## 5. Shipping and returns
- Accepted goods ship immediately and are delivered at 09:00 {lag_txt} later; the invoice is issued on delivery. A shipment may be delayed by one or more days or lost in transit (probabilities not disclosed). On loss the seller loses the reserved goods and no invoice is issued.{ship_note}
- The buyer may `coffee return_shipment` part or all of a delivery within {env_mod.RETURN_WINDOW_DAYS} days of the invoice issue date (only units received in that delivery and still held with at least {env_mod.RETURN_MIN_SHELF_DAYS} days of shelf life left). The invoice is credited at the deal price, the lots go back to the seller, and an already-paid amount is refunded (refused if the seller cannot pay or is bankrupt).

## 6. Inventory
- Stock is held in lots, each with its own cost and expiry day. Shelf life from the day a lot is produced (production lands / roast completes): {shelf_txt}. Purchased lots keep their expiry day.
- A lot is usable through its expiry day and written off in full (spoilage) at 19:00 that day.
- Shipments, roasting and consumer sales always consume the earliest-expiring usable lots first (FIFO by expiry); COGS is those lots' cost. Lots that would be expired on the delivery day are not shipped; if a delay makes them expire in transit, the seller writes them off and the buyer is not invoiced for them.
- Your committed holdings (on-hand kg of all items + your pending production/roast output + accepted, undelivered purchases) may not exceed your role's inventory cap: `accept_offer` as buyer, `produce_item` and `roast` are rejected if they would exceed it.

## 7. Money
- Daily fixed opex is charged at 09:00 every day.
- An invoice falls due its payment-terms days after its issue day. The buyer must settle it in full with `coffee pay_invoice`; {collect_txt}. After the due day an unpaid invoice accrues a simple late fee of {late:g}% of its outstanding principal per day.
- Bankruptcy: if your cash is below $0 at the start of any of your turns, you are bankrupt. You can take no further action; your books freeze; your open listings and offers are cancelled; undelivered deals with you are cancelled; your pending production/roasts are written off; your unpaid payables are written off on your creditors' books.{bankrupt_extra}
- When a business that owes you goes bankrupt, your receivables from it are written off as bad debt that day and undelivered deals with it are cancelled (reserved stock returns to the seller).

## 8. Consumers
- Every day at 19:00 consumers buy from each retailer's on-hand stock at the retailer's current prices. The demand model is not disclosed. Known structure:
  - no purchases above the reservation price; a price of 0 delists the item;
  - an elastic market pool, whose size falls as the retailers' sales-weighted price rises, is split between the retailers by price and by a hidden brand-loyalty multiplier per retailer (fixed for the run); price-shopping customers who find a retailer sold out buy at the other retailer instead;
  - each retailer pricing an item at or below its reservation price also gets a small fixed daily floor of regular customers for it;
  - demand is higher during the public festival window(s): {_festival_lines(max_days, festivals)} (size of the increase not disclosed).{demand_note}{pattern_note}{growth_note}

## 9. Time
- Business hours are 09:00–19:00. Each simulation action costs 30 virtual minutes (`coffee wait_for_next_day` costs 0), so at most 20 actions per day. All businesses act concurrently on the same virtual clock; listings, offers, deals and messages carry `Day N, HH:MM` timestamps.
- 09:00: opex is charged, production/roast output lands, deliveries arrive, and you receive a morning observation of your position.
- 19:00: {eod_txt}, in that order.
- `coffee wait_for_next_day` ends your day. You are woken earlier the same day, with the reason shown, when: an offer is made on your listing, your offer is accepted, a message arrives, a buyer pays your invoice, or a buyer returns goods to you.{advisory_note}
"""


LEGACY_INTERFACE = """\
## 10. Interface
Actions are function tools; `coffee <tool> --arg value` above denotes calling tool `<tool>` with argument `arg`. One tool call per turn; a turn without a tool call is a no-op that parks you until your next event. Every user message is a system observation or tool result. If your transcript nears the context window, older turns are replaced by a factual summary in `<context_summary>…</context_summary>`.
"""
