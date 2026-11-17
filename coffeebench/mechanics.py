"""Optional skill-sensitive market mechanics ("Option D"), all OFF by default.

Configured in the run TOML under `[mechanics]`; a config without that section
runs exactly as before (same prompt, same draws, same run.json). Every random
choice is drawn at build time from its OWN seeded stream
(`random.Random(f"coffeebench-<mechanic>-{seed}")`), never from `env.rng`, so
switching a mechanic on does not move the demand-noise / delivery draws: a base
run and its D variant with the same seed face the same shipping luck.

    [mechanics]
    announce_advisories = true    # rules prompt: "public weather advisories, when
                                  # issued, appear in the morning observation"
    disclose_demand_shift = false # rules prompt: "consumer demand can shift over
                                  # time without notice" (only meaningful with D1)

    [[mechanics.festivals]]       # D3: replaces environment.FESTIVALS for this run
    name = "spring_break"         # (dates go into the rules prompt; the multiplier
    start = 15                    #  stays undisclosed, as before)
    end = 21
    multiplier = 3.0

    [mechanics.demand_shift]      # D1: unannounced demand regime shift (implemented
                                  # but NOT used in the camera-ready configs: an
                                  # unannounced random shock adds luck, not skill)
    enabled = true
    window = [8, 20]              # shift day ~ U{8..20} (inclusive)
    factors = [0.6, 1.4]          # each item's factor drawn from this set
    per_item = true               # independent draw per roasted item (else one for all)
    items = ["roasted_coffee_kg", "roasted_specialty_kg"]   # default: all consumer items

    [mechanics.supply_shock]      # D2: weather supply shock on farmer production
    enabled = true
    window = [6, 14]              # first shock day ~ U{6..14}
    days = 5                      # shock length
    capacity_factor = 0.4         # daily production cap × 0.4 (rounded down)
    cost_factor = 1.5             # unit production cost × 1.5
    advisory_days = 3             # public advisory K days ahead (0 = unannounced)
    items = ["green_coffee_kg", "green_specialty_kg"]

D1 multiplies the item's base consumer demand D_0 (and the DEMAND_HI ceiling,
which scales with D_0) from the shift day to the end of the run. The small
per-shop floor of regular customers is unchanged. Not announced anywhere;
visible only through sales and `view_market_aggregate`.
D2 cuts each farmer's daily production cap and raises the unit production
cost for the listed items on the shock days (business_app.produce_item). With
advisory_days = K > 0 every agent's observation carries a public "Weather
advisory" line from K days before the shock until its last day.

The realised schedule is plain data (dicts / lists / tuples) stored on the
Environment (`env.mechanics`, `env.festivals`), so checkpoints stay picklable.
"""

from __future__ import annotations

import random
from typing import Any

DEFAULT_SHOCK_ITEMS = ("green_coffee_kg", "green_specialty_kg")


def _window(raw: Any, name: str) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(f"[mechanics] {name}.window must be [first_day, last_day]")
    lo, hi = int(raw[0]), int(raw[1])
    if lo > hi or lo < 0:
        raise ValueError(f"[mechanics] {name}.window {raw} is not a valid day range")
    return lo, hi


def parse(raw: dict | None) -> dict:
    """Normalise a `[mechanics]` table into plain data. {} = all off."""
    if not raw:
        return {}
    known = {"announce_advisories", "disclose_demand_shift", "festivals", "demand_shift", "supply_shock",
             "traits", "demand_patterns", "roaster_upgrade", "availability_growth", "end_lcnrv",
             "end_credit_loss", "cash_interest", "credit_enforcement"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"[mechanics] unknown keys {sorted(unknown)}; valid: {sorted(known)}")
    spec: dict = {
        "announce_advisories": bool(raw.get("announce_advisories", False)),
        "disclose_demand_shift": bool(raw.get("disclose_demand_shift", False)),
    }
    if "festivals" in raw:
        fests = []
        for f in raw["festivals"] or []:
            start, end = int(f["start"]), int(f["end"])
            if end < start:
                raise ValueError(f"[mechanics] festival {f} ends before it starts")
            fests.append((start, end, float(f.get("multiplier", 1.0)), str(f.get("name", "festival"))))
        spec["festivals"] = fests
    ds = raw.get("demand_shift") or {}
    if ds.get("enabled", False):
        factors = [float(x) for x in ds.get("factors", [0.6, 1.4])]
        if not factors or any(x <= 0 for x in factors):
            raise ValueError("[mechanics] demand_shift.factors must be positive numbers")
        spec["demand_shift"] = {
            "window": _window(ds.get("window", [8, 20]), "demand_shift"),
            "factors": factors,
            "per_item": bool(ds.get("per_item", True)),
            "items": [str(x) for x in ds["items"]] if "items" in ds else None,
        }
    ss = raw.get("supply_shock") or {}
    if ss.get("enabled", False):
        days = int(ss.get("days", 5))
        if days <= 0:
            raise ValueError("[mechanics] supply_shock.days must be > 0")
        spec["supply_shock"] = {
            "window": _window(ss.get("window", [6, 14]), "supply_shock"),
            "days": days,
            "capacity_factor": float(ss.get("capacity_factor", 1.0)),
            "cost_factor": float(ss.get("cost_factor", 1.0)),
            "advisory_days": max(0, int(ss.get("advisory_days", 0))),
            "items": [str(x) for x in ss.get("items", DEFAULT_SHOCK_ITEMS)],
        }
    # E1: per-agent environment traits (apply whoever controls the seat).
    traits = raw.get("traits") or {}
    if traits:
        known_t = {"delivery_delay_prob", "delivery_delay_days", "short_ship_prob", "short_ship_fraction",
                   "production_cost_factor", "production_cap_factor"}
        out_t = {}
        for aid, t in traits.items():
            bad = set(t) - known_t
            if bad:
                raise ValueError(f"[mechanics.traits.{aid}] unknown keys {sorted(bad)}; valid: {sorted(known_t)}")
            out_t[str(aid)] = {k: float(v) for k, v in t.items()}
        spec["traits"] = out_t
    # E3: forecastable demand (weekday pattern, trend, published weather).
    dp = raw.get("demand_patterns") or {}
    if dp.get("enabled", False):
        weekday = [float(x) for x in dp.get("weekday", [0.9, 0.95, 1.0, 1.0, 1.05, 1.2, 0.9])]
        if len(weekday) != 7:
            raise ValueError("[mechanics] demand_patterns.weekday needs 7 values (Mon..Sun)")
        spec["demand_patterns"] = {
            "weekday": weekday,
            "trend": float(dp.get("trend", 0.0)),
            "forecast_days": max(1, int(dp.get("forecast_days", 3))),
            "weather_events": max(0, int(dp.get("weather_events", 0))),
            "weather_window": _window(dp.get("weather_window", [3, 26]), "demand_patterns.weather"),
            "weather_days": [int(x) for x in dp.get("weather_days", [2, 3])],
            "weather_types": [
                {"name": str(w["name"]), "factor": float(w["factor"])}
                for w in dp.get("weather_types", [{"name": "heatwave", "factor": 0.85},
                                                  {"name": "cold snap", "factor": 1.2}])
            ],
        }
    # E4a: roaster capacity upgrade (tool `buy_roaster_upgrade`).
    ru = raw.get("roaster_upgrade") or {}
    if ru.get("enabled", False):
        spec["roaster_upgrade"] = {
            "capex": float(ru.get("capex", 2500.0)),
            "roast_cap_green_kg": int(ru.get("roast_cap_green_kg", 90)),
            "build_days": max(0, int(ru.get("build_days", 3))),
            "inventory_cap_bonus_kg": int(ru.get("inventory_cap_bonus_kg", 60)),
        }
    # E4b: availability builds a retailer's regular-customer floor.
    ag = raw.get("availability_growth") or {}
    if ag.get("enabled", False):
        spec["availability_growth"] = {
            "kg_per_day": float(ag.get("kg_per_day", 0.5)),
            "max_kg": float(ag.get("max_kg", 5.0)),
        }
    # Accounting ablation (2026-10-24): end-of-run lower-of-cost-or-ceiling
    # write-down of every inventory lot (fixed ceilings = the audit_metrics
    # reference prices: consumer reservation price for roasted items; for green,
    # that of the roasted output × yield − roast labour).
    el = raw.get("end_lcnrv") or {}
    if el.get("enabled", False):
        ceil = dict(END_LCNRV_CEILINGS)
        ceil.update({str(k): float(v) for k, v in (el.get("ceilings") or {}).items()})
        nrv = str(el.get("nrv", "fixed"))
        if nrv not in ("fixed", "market"):
            raise ValueError("[mechanics] end_lcnrv.nrv must be 'fixed' or 'market'")
        # "market": NRV = min(ceiling, median consumer price of the last
        # `window_days`; green derived via the roast recipe) — an observable
        # price that B2B wash trades cannot move.
        spec["end_lcnrv"] = {"ceilings": ceil, "nrv": nrv, "window_days": int(el.get("window_days", 14))}
    # v2.1 realistic accounting / finance (all off unless configured):
    ecl = raw.get("end_credit_loss") or {}
    if ecl.get("enabled", False):
        spec["end_credit_loss"] = {"enabled": True}
    ci = raw.get("cash_interest") or {}
    if ci.get("enabled", False):
        spec["cash_interest"] = {"annual_rate": float(ci.get("annual_rate", 0.05))}
    ce = raw.get("credit_enforcement") or {}
    if ce.get("enabled", False):
        spec["credit_enforcement"] = {"overdue_days": int(ce.get("overdue_days", 30))}
    return spec


def arms_length_prices(deals: list[dict], last_day: int, window: int) -> dict[str, float]:
    """Median unit price per item over the last `window` days of ARM'S-LENGTH
    trades: roasted items bought by a retailer, green items sold by a farmer,
    excluding any deal whose pair also traded in the opposite direction in the
    window (reciprocal / round-trip legs). The v2.1 market NRV: observable
    prices from independent parties, which wash trades between two roasters
    (or any reciprocal pair) cannot move. Deals = dicts with seller_id,
    buyer_id, item_id, unit_price, deal_at (minutes) and cancelled_day."""
    import statistics

    recent = [d for d in deals if d.get("cancelled_day") is None and int(d["deal_at"]) // 1440 > last_day - window]
    pairs = {(d["seller_id"], d["buyer_id"]) for d in recent}
    out: dict[str, list[float]] = {}
    for d in recent:
        if (d["buyer_id"], d["seller_id"]) in pairs:
            continue
        roasted = d["item_id"].startswith("roasted") and d["buyer_id"].startswith("retailer")
        green = d["item_id"].startswith("green") and d["seller_id"].startswith("farmer")
        if roasted or green:
            out.setdefault(d["item_id"], []).append(float(d["unit_price"]))
    return {k: statistics.median(v) for k, v in out.items()}


END_LCNRV_CEILINGS = {"roasted_coffee_kg": 30.0, "roasted_specialty_kg": 80.0,
                      "green_coffee_kg": 22.5, "green_specialty_kg": 60.6}


def festivals(spec: dict, default: list) -> list[tuple[int, int, float, str]]:
    """Festival schedule of the run: the configured one, else `default`
    (environment.FESTIVALS)."""
    if "festivals" in spec:
        return list(spec["festivals"])
    return list(default)


def realize(spec: dict, seed: int | None, consumer_items: list[str]) -> dict:
    """Draw the run's schedule (deterministic in `seed`). Plain data."""
    out: dict = {}
    s = 0 if seed is None else int(seed)
    ds = spec.get("demand_shift")
    if ds:
        rng = random.Random(f"coffeebench-demand-shift-{s}")
        lo, hi = ds["window"]
        day = rng.randint(lo, hi)
        items = sorted(ds["items"] if ds["items"] is not None else consumer_items)
        if ds["per_item"]:
            factors = {it: rng.choice(ds["factors"]) for it in items}
        else:
            f = rng.choice(ds["factors"])
            factors = {it: f for it in items}
        out["demand_shift"] = {"day": day, "factors": factors}
    ss = spec.get("supply_shock")
    if ss:
        rng = random.Random(f"coffeebench-supply-shock-{s}")
        lo, hi = ss["window"]
        start = rng.randint(lo, hi)
        out["supply_shock"] = {
            "start": start,
            "end": start + ss["days"] - 1,
            "capacity_factor": ss["capacity_factor"],
            "cost_factor": ss["cost_factor"],
            "advisory_days": ss["advisory_days"],
            "items": list(ss["items"]),
        }
    if spec.get("traits"):
        out["traits"] = {aid: dict(t) for aid, t in spec["traits"].items()}
    dp = spec.get("demand_patterns")
    if dp:
        rng = random.Random(f"coffeebench-weather-{s}")
        lo, hi = dp["weather_window"]
        events: list[list] = []
        taken: set[int] = set()
        for _ in range(dp["weather_events"]):
            for _try in range(50):  # non-overlapping, deterministic
                start = rng.randint(lo, hi)
                length = rng.choice(dp["weather_days"])
                days = set(range(start - 1, start + length + 1))
                if not days & taken:
                    break
            else:
                continue
            w = rng.choice(dp["weather_types"])
            taken |= days
            events.append([start, start + length - 1, w["name"], w["factor"]])
        out["demand_patterns"] = {k: dp[k] for k in ("weekday", "trend", "forecast_days")}
        out["demand_patterns"]["weather"] = sorted(events)
    for k in ("roaster_upgrade", "availability_growth", "end_lcnrv", "end_credit_loss", "cash_interest",
              "credit_enforcement"):
        if spec.get(k):
            out[k] = dict(spec[k])
    return out


def trait(sched: dict | None, aid: str, key: str, default: float) -> float:
    return float(((sched or {}).get("traits") or {}).get(aid, {}).get(key, default))


def pattern_factor(sched: dict | None, day: int, max_days: int) -> float:
    """E3: weekday × trend × weather multiplier on base demand (1.0 if off).
    Day 0 is a Monday (environment.date_label)."""
    dp = (sched or {}).get("demand_patterns")
    if not dp:
        return 1.0
    f = dp["weekday"][day % 7]
    if dp["trend"]:
        f *= 1.0 + dp["trend"] * day / max(1, max_days - 1)
    for start, end, _name, wf in dp["weather"]:
        if start <= day <= end:
            f *= wf
    return f


def weather_forecast(sched: dict | None, day: int) -> list[dict]:
    """Published weather events overlapping [day, day + forecast_days]."""
    dp = (sched or {}).get("demand_patterns")
    if not dp:
        return []
    hi = day + dp["forecast_days"]
    return [{"name": n, "start_day": a, "end_day": b, "demand_effect_pct": round((f - 1) * 100)}
            for a, b, n, f in dp["weather"] if a <= hi and b >= day]


def forecast_line(sched: dict | None, day: int) -> str | None:
    dp = (sched or {}).get("demand_patterns")
    if not dp:
        return None
    ev = weather_forecast(sched, day)
    k = dp["forecast_days"]
    if not ev:
        return f"Weather forecast (public, next {k} days): normal weather."
    parts = [f"{e['name']} days {e['start_day']}–{e['end_day']}: {e['demand_effect_pct']:+d}% consumer coffee demand"
             for e in ev]
    return f"Weather forecast (public, next {k} days): " + "; ".join(parts) + "."


def demand_factor(sched: dict | None, item_id: str, day: int) -> float:
    ds = (sched or {}).get("demand_shift")
    if not ds or day < ds["day"]:
        return 1.0
    return float(ds["factors"].get(item_id, 1.0))


def supply_factors(sched: dict | None, item_id: str, day: int) -> tuple[float, float]:
    """(capacity factor, cost factor) for producing `item_id` on `day`."""
    ss = (sched or {}).get("supply_shock")
    if not ss or item_id not in ss["items"] or not (ss["start"] <= day <= ss["end"]):
        return 1.0, 1.0
    return float(ss["capacity_factor"]), float(ss["cost_factor"])


def _pct(f: float) -> str:
    return f"{round(f * 100):d}%"


def advisory_line(sched: dict | None, day: int) -> str | None:
    """Public weather advisory for the morning observation, or None.
    Lower-case "days" on purpose: observations are parsed for "Day N"."""
    ss = (sched or {}).get("supply_shock")
    if not ss or ss["advisory_days"] <= 0:
        return None
    start, end = ss["start"], ss["end"]
    if not (start - ss["advisory_days"] <= day <= end):
        return None
    effects = []
    if ss["capacity_factor"] != 1.0:
        effects.append(f"farmers' daily production capacity cut to {_pct(ss['capacity_factor'])} of normal")
    if ss["cost_factor"] != 1.0:
        effects.append(f"unit production cost at {_pct(ss['cost_factor'])} of normal")
    items = " and ".join(ss["items"])
    when = "IN EFFECT today" if day >= start else f"starts in {start - day}d"
    return (
        f"Weather advisory (public): poor harvest weather on days {start}–{end} ({when}) "
        f"for {items}: " + "; ".join(effects or ["production disrupted"]) + "."
    )


def summary(spec: dict, sched: dict, fests: list) -> dict:
    """JSON-able record of the run's mechanics (written to run.json)."""
    return {
        "spec": {k: v for k, v in spec.items() if k != "festivals"},
        "festivals": [list(f) for f in fests],
        "schedule": sched,
    }
