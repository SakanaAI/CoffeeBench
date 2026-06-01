"""Dataclasses for the CoffeeBench operations testbed."""

from dataclasses import dataclass, field


@dataclass
class Item:
    """A traded tangible good. Distinct items are NOT interchangeable here —
    each item has its own id and name, and prices are discovered via deals.

    `retail_reservation_price`, when set, marks the item as
    consumer-facing — retailers can sell it via the env's auto-run
    consumer-demand model at or below that per-unit price. Items
    without a reservation price never sell to consumers (raw goods,
    intermediates).

    `consumer_demand_base`, when set, overrides the global DEMAND_BASE
    (D_0) for this item — used to express different consumer market
    sizes per consumer-facing item (e.g. premium/blend has lower
    baseline demand than commodity roasted). None = use global default.

    `consumer_demand_floor`, when set, gives EACH retailer that prices
    this item within reservation a flat per-day inelastic demand of
    this many kg before the elastic market pool is applied. Models
    the "regulars" segment — habitual customers who buy from their
    chosen shop regardless of competitor pricing. Stabilises sales
    against day-to-day demand-noise wipeouts and prevents the
    market-wide collapse that occurs when all shops price slightly
    above the reservation. None = no floor (purely elastic market).
    """

    id: str
    name: str
    description: str = ""
    retail_reservation_price: float | None = None
    consumer_demand_base: float | None = None
    consumer_demand_floor: float | None = None
    # Days from accept until the buyer receives the goods. Tangibles
    # default to 1 (overnight shipping).
    delivery_lag_days: int = 1
    # Production fields. If `produced_by_role` is set, agents in that role
    # can call `produce_item(item_id, quantity)` to mint new units at
    # `production_cost_per_unit` from their cash, capped at
    # `daily_production_cap` kg per agent per day. Items without a
    # producer-role cannot be produced directly via `produce_item` (e.g.
    # roasted coffee, which is the output of the roaster's `roast` action,
    # not a primary production target).
    produced_by_role: str | None = None
    production_cost_per_unit: float = 0.0
    daily_production_cap: int = 0
    # Days from `produce_item(...)` call until the new units land in the
    # producer's inventory. Cash is debited at call time (commitment); the
    # asset materializes after `production_lag_days`. 0 = same-day.
    production_lag_days: int = 0


@dataclass
class Listing:
    id: str
    seller_id: str
    item_id: str
    qty: int  # units offered
    asking_price: float  # PER UNIT, in USD
    payment_terms_days: int  # net-N from delivery
    # Sim-time at posting, in minutes since sim start (e.g. 44040
    # = Day 30, 14:00). Day = `posted_at // 1440`. Render in agent
    # views via `coffeebench.event_loop.format_min()`.
    posted_at: int
    status: str = "open"  # "open" | "closed" | "cancelled" | "expired"


@dataclass
class Offer:
    id: str
    listing_id: str
    buyer_id: str
    seller_id: str  # mirrored from listing for convenience
    offered_price: float  # PER UNIT
    qty: int  # ≤ listing.qty
    payment_terms_days: int  # buyer's proposed payment terms
    posted_at: int  # minutes since sim start
    message: str | None = None
    status: str = "pending"  # "pending" | "accepted" | "rejected" | "withdrawn"


@dataclass
class Deal:
    """Ground-truth record of one accepted transaction. Created by the env at
    accept_offer time. Both sides' inventory and ledgers are updated based on
    this single record, and the linked Invoice (`invoice_id`) settles cash.
    """

    id: str
    listing_id: str
    offer_id: str
    seller_id: str
    buyer_id: str
    item_id: str
    qty: int
    unit_price: float
    total_price: float  # qty * unit_price
    payment_terms_days: int
    # Sim-time at acceptance (`accepted_at`) and at delivery, in
    # minutes since sim start. Day-level access: `deal_at // 1440`.
    deal_at: int
    delivery_at: int
    invoice_id: str
    notes: str | None = None
    # Cumulative qty returned by the buyer via return_shipment. Cannot exceed
    # `qty`. Used by the env to enforce the partial-return cap and by the
    # audit to compute total_returned_value / return_rate.
    returned_qty: int = 0
    # Seller-side cogs snapshot pinned at accept_offer time (qty ×
    # seller's then-current WAVG cost). Used at delivery to book
    # inventory_out, at delivery loss to book the writedown, and on
    # buyer returns to compute the proportional cogs_reversal — all
    # insulated from any post-accept WAVG drift on the seller's books.
    # Populated by `Environment._on_deal_accepted` synchronously inside
    # `Marketplace.accept_offer`, so every deal exposed to downstream
    # code has it set; the dataclass default is the pre-hook sentinel.
    _reserved_seller_cogs: float = 0.0


@dataclass
class Message:
    """A private free-form message from `sender_id` to `recipient_id`.
    Visible only to those two agents.

    `title` is REQUIRED — a short subject line (≤80 chars hard cap, ≤60
    recommended) that lets listing views and the audit pipeline scan
    intent without reading every body. Bodies stay free-form."""

    id: str
    sender_id: str
    recipient_id: str
    title: str
    body: str
    sent_at: int  # minutes since sim start
    # Reference fields so a message can be tied to a specific listing/offer/deal
    # (helpful for both agents and post-hoc audit).
    ref_listing_id: str | None = None
    ref_offer_id: str | None = None
    ref_deal_id: str | None = None


@dataclass
class AgentEndowment:
    """Initial state for one agent at run start."""

    agent_id: str
    display_name: str
    role: str  # "farmer" | "roaster" | "retailer"
    persona: str  # short narrative description
    initial_cash: float
    initial_inventory: dict[str, int] = field(default_factory=dict)  # item_id -> qty
    initial_cost_basis: dict[str, float] = field(
        default_factory=dict
    )  # item_id -> per-unit cost basis
    # retailer-only: per-item starting consumer-facing price ($/kg).
    # Empty for non-retailers. Pre-set so the supply chain runs from
    # day 0 instead of stalling until the retailer calls
    # set_retail_price; the retailer is free to adjust at any time.
    initial_retail_prices: dict[str, float] = field(default_factory=dict)


# --- Bookkeeping primitives ---------------------------------------------
# Invoices live on both sides of every credit transaction (AR for the
# issuer, AP for the payer). JournalEntry is the env's ground-truth shadow
# ledger from which the run-end audit (and the agent's score) is computed.


@dataclass
class Invoice:
    """A bill issued at delivery. Lives on both sides:
      - payer side books it as accounts_payable (liability)
      - issuer side books it as accounts_receivable (asset)

    Payment terms are measured in days from issue_date. The amount covers
    goods already delivered. Returns/credits are tracked via
    `credited_amount`; `net_outstanding` reflects what the payer still owes.
    """

    id: str
    issuer: str  # agent_id of the entity that's owed
    payer: str  # agent_id of the entity that owes
    amount: float
    issue_date: int
    due_date: int
    reference: str | None = None  # linked coffeebench/order id
    paid: bool = False
    paid_date: int | None = None
    credited_amount: float = 0.0  # cumulative return credits
    last_return_date: int | None = None
    returned: bool = False  # True once credited_amount covers amount
    returned_date: int | None = None
    # Late-payment interest accrued while past due. Env runs a daily sweep
    # that adds `LATE_FEE_PER_DAY * amount` for every day overdue. The buyer
    # eventually pays this on top of the principal; the seller realizes it
    # as interest revenue when collected.
    late_fees_accrued: float = 0.0
    # Direct write-off marker. Set to True when the payer goes bankrupt
    # and the issuer (seller) recognizes the receivable as uncollectible.
    # Once flagged, the invoice is excluded from both AR and AP balances
    # via `net_outstanding`. The seller's truth ledger logs a one-shot
    # `bad_debt_expense` entry (-> `other_operating_expense` bucket) on
    # the same day. We keep the invoice in the books as an audit trail
    # — `bad_debt` + `bad_debt_date` mark when and why it was written
    # off rather than deleting the record.
    bad_debt: bool = False
    bad_debt_date: int | None = None

    @property
    def net_outstanding(self) -> float:
        if self.paid or self.returned or self.bad_debt:
            return 0.0
        return max(0.0, self.amount + self.late_fees_accrued - self.credited_amount)


@dataclass
class JournalEntry:
    id: str
    day: int
    trader_id: str  # agent_id this entry belongs to
    # Coarse categories used by the env's truth ledger:
    # "sale_revenue", "sale_reversal", "purchase_cogs", "cash_in", "cash_out",
    # "inventory_in", "inventory_out", "inventory_consumed", "cogs_reversal",
    # "operating_expense", "writedown", "spoilage_expense", "bad_debt_expense",
    # "interest_expense", "interest_revenue", "other"
    # `cogs_reversal` nets against `inventory_out` in audit cogs — emitted
    # on the seller side when goods come back via return_shipment so that
    # cogs reflects only kg actually sold (not kg later returned).
    entry_type: str
    amount: float  # always non-negative; direction encoded by entry_type
    counterparty: str | None = None
    item: str | None = None
    quantity: int | None = None
    memo: str | None = None
    reference: str | None = None  # linked coffeebench/invoice id
