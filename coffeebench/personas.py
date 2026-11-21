"""Private persona blocks for LLM background seats ("persona-LLM" background).

A seat's policy profile (`heuristic_farmer@adversarial`, `heuristic_retailer@negotiator`,
...) means the same intent whether a rule bot or an LLM plays it: the rule bot
executes the profile's parameters (coffeebench.models.*_model.PROFILES); an LLM
seat gets the same intent as a short private block appended to ITS instruction
only. The shared rules sections stay identical for every seat; environment
traits ([mechanics.traits]) apply regardless of who controls the seat.

The numbers are rendered from the rule profiles' parameters, so the two stay in
sync.
"""

from __future__ import annotations


def seat_profile(model_id: str | None, role: str) -> str:
    """Profile name of a seat from its [models] entry (`<rule model>@<profile>`);
    'default' when the entry has no profile or names a non-rule model."""
    if model_id and "@" in model_id and model_id.partition("@")[0].startswith("heuristic_"):
        return model_id.partition("@")[2] or "default"
    return "default"


def _farmer(profile: str) -> str:
    from coffeebench.models.heuristic_farmer_model import DEFAULT_PARAMS, PROFILES

    p = {**DEFAULT_PARAMS, **PROFILES[profile]}
    lines = [
        f"- Pricing: open your green-coffee listings at about {p['start_markup']:g}× production cost, "
        f"adjust by about {p['price_step'] * 100:g}% a day with demand (up when stock sells out, down "
        f"after a day without sales), never below {p['floor_markup']:g}× your lot cost.",
        (f"- Discounts: at most {p['max_concession'] * 100:g}% below your list price."
         if p["max_concession"] > 0 else "- Discounts: none; you sell at your list price or not at all."),
        f"- Credit: payment terms up to {p['free_terms_days']} days at the agreed price; longer terms only "
        f"with a premium, never beyond {p['max_terms_days']} days.",
        f"- Production: aim for about {p['cover_days']} days of expected sales on hand plus the production "
        f"lag; keep at least {p['cash_buffer_days']} days of opex in cash.",
    ]
    if p["negotiate"]:
        lines.append(
            "- Negotiation: when an offer is below your price, reject it and reply with a counter-quote; on "
            f"each improved offer from the same buyer concede about {p['neg_step'] * 100:g}% of the gap, "
            f"never below max({p['floor_markup']:g}× lot cost, {(1 - p['neg_max_concession']) * 100:g}% of list)."
        )
    if p["scarcity_messages"]:
        lines.append(
            "- Communication: every few days tell the roasters that supply is short and prices are about to "
            "rise, whatever your actual stock, to push them to buy now."
        )
    return "\n".join(lines)


def _retailer(profile: str) -> str:
    from coffeebench.models.heuristic_retailer_model import DEFAULT_PARAMS, PROFILES

    p = {**DEFAULT_PARAMS, **PROFILES[profile]}
    lines = [
        f"- Buying: pay at most {p['max_buy_ratio'] * 100:g}% of your own retail price for roasted coffee; "
        "buy from the cheapest fresh listings.",
        f"- Stock: keep about {p['cover_days']} days of expected sales (about {p['ramp_factor']:g}× that just "
        "before a festival); pay invoices on time.",
        f"- Retail prices: raise about {p['price_step'] * 100:g}% after a sell-out, cut about "
        f"{p['price_step'] * 100:g}% when most stock is left over; never above "
        f"{p['cap_of_reservation'] * 100:g}% of the reservation price.",
    ]
    if p["negotiate"]:
        lines.append(
            f"- Negotiation: open bids at about {p['open_bid_frac'] * 100:g}% of the ask and raise by about "
            f"{p['bid_raise_step'] * 100:g}% of the ask after each rejection, never above your buying limit."
        )
    return "\n".join(lines)


def _roaster(profile: str) -> str:  # noqa: ARG001 — one profile (heuristic_roaster_v2)
    from coffeebench.models import heuristic_roaster_v2_model as v2

    return "\n".join([
        f"- Pricing: anchor roasted prices at about {v2.RETAILER_MARGIN * 100:g}% of the consumer price, "
        f"never below {v2.FLOOR_MARKUP:g}× lot cost; adjust about {v2.PRICE_STEP * 100:g}% a day with "
        "sell-through.",
        f"- Buying: bid for green only up to the price at which your roasted output still clears "
        f"{v2.FLOOR_MARKUP:g}× cost; keep about {v2.CASH_BUFFER_DAYS} days of opex in cash.",
    ])


BUILDERS = {"farmer": _farmer, "retailer": _retailer, "roaster": _roaster}

# "soft" style: qualitative character only (no numbers, no step-by-step policy),
# so the LLM decides its own strategy within a recognisable business identity.
SOFT = {
    ("farmer", "default"): "a family coffee farm that wants steady, profitable sales. You care about "
        "your margins and dislike deep discounts, but you value reliable long-term buyers.",
    ("farmer", "negotiator"): "a coffee farm with a firm commercial style. You protect your margins, "
        "bargain when buyers push for lower prices, and prefer short payment terms.",
    ("farmer", "adversarial"): "an opportunistic, aggressive coffee farm. You compete on low list prices, "
        "rarely give discounts, and are willing to talk up scarcity to make buyers commit sooner.",
    ("retailer", "default"): "a neighbourhood café chain that wants full shelves and satisfied customers "
        "at a healthy profit.",
    ("retailer", "negotiator"): "a price-conscious café chain. You bargain for good wholesale prices, "
        "keep your shelves stocked (especially around festivals) and pay suppliers on time.",
    ("roaster", "default"): "an independent coffee roastery that wants to grow a profitable business.",
}


def persona_block(role: str, profile: str = "default", style: str = "strict") -> str:
    """Private persona section for an LLM seat of `role` with `profile`.
    style "strict": the rule profile's policy with its numbers; "soft": a short
    qualitative character (the LLM chooses its own strategy); "none" is handled by
    the caller (no block at all)."""
    if style == "soft":
        char = SOFT.get((role, profile)) or SOFT[(role, "default")]
        return (
            "\n## Private business profile (only you see this)\n"
            f"Your company is {char} How you act on this is up to you.\n"
        )
    body = BUILDERS[role](profile)
    return (
        "\n## Private business profile (only you see this)\n"
        f"Your company's standing policy ({role}, profile `{profile}`). Follow it unless the "
        "rules above make it impossible:\n" + body + "\n"
    )
