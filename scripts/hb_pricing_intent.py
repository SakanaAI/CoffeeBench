"""Pricing intent of the focal roaster: share of agent-step thoughts that reason about
raising / lowering prices (and with a demand/margin rationale), and listing-price moves.

    python3 scripts/hb_pricing_intent.py results/hb_f1
"""

import json
import glob
import re
import collections
import statistics
import sys

UP = re.compile(
    r"\b(rais(e|ing)|increas(e|ing)|bump(ing)?|push(ing)?|lift(ing)?|nudg(e|ing))\b[^.]{0,40}\b(price|ask|asking)|\b(price|ask)s?\b[^.]{0,25}\b(higher|up)\b|\btest(ing)? (a )?higher|\bunderpric",
    re.I,
)
DOWN = re.compile(
    r"\b(lower(ing)?|cut(ting)?|reduc(e|ing)|drop(ping)?|discount(ing)?)\b[^.]{0,40}\b(price|ask|asking)|\b(price|ask)s?\b[^.]{0,25}\b(lower|down)\b",
    re.I,
)
WHY = re.compile(
    r"sold out|sell(s|ing)? out|sold (quickly|fast|immediately)|snapped up|strong demand|demand (is|was) (strong|high)|both bid|multiple (offers|bids)|competing (offers|bids)|accepted (immediately|instantly)|reservation|willing to pay|margin|elastic|room to",
    re.I,
)
rows = []
for ev in sorted(
    glob.glob(sys.argv[1] + "/*/*/hb_roaster_none_E_llmrival/seed_0/run.events.jsonl")
):
    ro, mo = ev.split("/")[-5].replace("Persona", ""), ev.split("/")[-4]
    if mo in ("passive", "heuristicv2"):
        continue
    th = 0
    up = down = upwhy = 0
    asks = collections.defaultdict(list)
    ex = None
    for line in open(ev):
        if '"roaster_A"' not in line or '"agent_step"' not in line:
            continue
        e = json.loads(line)
        if e.get("agent_id") != "roaster_A":
            continue
        t = e.get("thought") or ""
        if t:
            th += 1
        u = bool(UP.search(t))
        d = bool(DOWN.search(t))
        up += u
        down += d
        if u and WHY.search(t):
            upwhy += 1
            if ex is None or len(t) < len(ex[1]):
                ex = (e["day"], t)
        if e.get("action") == "post_listing":
            ai = e.get("action_input") or {}
            if "asking_price" in ai:
                asks[ai.get("item_id")].append((e["day"], float(ai["asking_price"])))
    # listing-price moves
    moves = collections.Counter()
    for it, L in asks.items():
        for (d0, p0), (d1, p1) in zip(L, L[1:]):
            moves["up" if p1 > p0 else "down" if p1 < p0 else "same"] += 1
    n = sum(moves.values()) or 1
    first = {it: L[0][1] for it, L in asks.items()}
    last = {
        it: statistics.mean(p for d, p in L if d >= 60)
        if any(d >= 60 for d, p in L)
        else None
        for it, L in asks.items()
    }
    rows.append(
        (
            ro,
            mo,
            th,
            up,
            down,
            upwhy,
            moves["up"] / n,
            moves["down"] / n,
            {
                k[8:12]: (round(first[k], 1), round(last[k], 1) if last[k] else None)
                for k in first
            },
            ex,
        )
    )
for r in sorted(rows, key=lambda r: (r[0], -r[5] / max(r[2], 1))):
    ro, mo, th, up, down, upwhy, fu, fd, fl, ex = r
    print(
        f"{ro:11s} {mo:10s} thoughts={th:5d} raise%={100 * up / th:4.1f} lower%={100 * down / th:4.1f} raise_with_reason%={100 * upwhy / th:4.1f} | listing moves up={fu:.2f} down={fd:.2f} | first->late ask {fl}"
    )
print()
for r in rows:
    if r[1] in ("Sonnet55", "Astra", "Gemini38F", "Qwen38F", "Fable5") and r[9]:
        print(f"[{r[0]}/{r[1]} d{r[9][0]}] {r[9][1][:600]}\n")
