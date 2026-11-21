"""Negotiation metrics of the focal roaster: bids below the ask when buying (and their
acceptance), bids below the ask on its own listings (and how often it accepts), and
messages that quote prices.

    python3 scripts/hb_negotiation.py results/hb_f1
"""

import json
import glob
import sys
import re

for rj in sorted(
    glob.glob(sys.argv[1] + "/*/*/hb_roaster_none_E_llmrival/seed_0/run.json")
):
    ro, mo = rj.split("/")[-5].replace("Persona", ""), rj.split("/")[-4]
    if mo in ("passive",):
        continue
    r = json.load(open(rj))
    L = {x["id"]: x for x in r["marketplace"]["listings"]}
    deals_by_offer = {
        d["offer_id"] for d in r["marketplace"]["deals"] if d.get("offer_id")
    }
    offs = [o for o in r["marketplace"]["offers"] if o.get("buyer_id") == "roaster_A"]
    below = acc_below = atask = 0
    for o in offs:
        lst = L.get(o.get("listing_id"))
        if not lst:
            continue
        ask = lst.get("asking_price")
        px = o.get("offered_price") or o.get("unit_price")
        if ask is None or px is None:
            continue
        if px < ask - 1e-9:
            below += 1
            acc_below += o.get("status") == "accepted"
        else:
            atask += 1
    # incoming offers on focal listings: accepted below ask?
    inc = [
        o
        for o in r["marketplace"]["offers"]
        if L.get(o.get("listing_id"), {}).get("seller_id") == "roaster_A"
    ]
    inc_below = sum(
        1
        for o in inc
        if (o.get("offered_price") or 0) < L[o["listing_id"]]["asking_price"] - 1e-9
    )
    inc_below_acc = sum(
        1
        for o in inc
        if (o.get("offered_price") or 0) < L[o["listing_id"]]["asking_price"] - 1e-9
        and o.get("status") == "accepted"
    )
    msgs = [m for m in r["marketplace"]["messages"] if m["sender_id"] == "roaster_A"]
    priced = sum(
        bool(re.search(r"\$?\d+(\.\d+)?\s*(/kg|per kg)|\$\d", m["body"])) for m in msgs
    )
    print(
        f"{ro:11s} {mo:11s} buy_offers={below + atask:4d} below_ask={below / (below + atask or 1):.2f} (accepted {acc_below / (below or 1):.2f}) | bids_on_my_listings={len(inc):4d} below_ask={inc_below / (len(inc) or 1):.2f} I_accepted_below={inc_below_acc / (inc_below or 1):.2f} | msgs={len(msgs)} with_price={priced / (len(msgs) or 1):.2f}"
    )
