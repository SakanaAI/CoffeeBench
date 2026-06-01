"""Run-config loader (TOML) for CoffeeBench experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover — Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class RunConfig:
    name: str
    description: str = ""
    max_days: int = 90
    seeds: list[int] = field(default_factory=lambda: [0])
    default_model: str = "claude-sonnet-4-6"
    # Designated focal/main agent. When set, the run terminates as soon
    # as this agent goes bankrupt — once the subject of measurement is
    # dead, continuing collects no signal. None = no early-stop.
    main_agent: str | None = None
    # agent_id -> model_id
    models: dict[str, str] = field(default_factory=dict)
    # env-constant overrides (only keys that are set will override defaults).
    economy: dict[str, Any] = field(default_factory=dict)
    # Per-agent score-framing overrides for the LLM SYSTEM_PROMPT. Keyed
    # by agent_id; value is a dict of {metric, target_usd?}. Recognised
    # metrics: "net_income" (default), "revenue", "revenue_pressure".
    # Truth-ledger leaderboard score is unaffected by this — it tells
    # the agent what to optimise for, not what is reported.
    kpi: dict[str, dict[str, Any]] = field(default_factory=dict)
    # path the config was loaded from (for snapshot copying)
    source_path: str | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        exp = data.get("experiment", {}) or {}
        run = data.get("run", {}) or {}
        models_block = dict(data.get("models", {}) or {})
        default_model = models_block.pop("default", "claude-sonnet-4-6")
        # All remaining keys in [models] are agent_id -> model_id overrides.
        models = {str(k): str(v) for k, v in models_block.items()}

        # Keep raw values; apply_economy_overrides type-casts per-key.
        economy = {str(k): v for k, v in (data.get("economy") or {}).items()}

        # [kpi.<agent_id>] tables → {agent_id: {metric, target_usd?}}
        # Also accept [kpi] as a flat agent_id -> "metric" string (shorthand).
        kpi_block = data.get("kpi") or {}
        kpi: dict[str, dict[str, Any]] = {}
        for k, v in kpi_block.items():
            aid = str(k)
            if isinstance(v, str):
                kpi[aid] = {"metric": v}
            elif isinstance(v, dict):
                kpi[aid] = {str(kk): vv for kk, vv in v.items()}
            else:
                print(f"[config] WARN: ignoring kpi[{aid}] (must be str or table)")

        name = exp.get("name") or path.stem
        main_agent_raw = run.get("main_agent")
        return cls(
            name=str(name),
            description=str(exp.get("description") or ""),
            max_days=int(run.get("max_days", 90)),
            seeds=[int(s) for s in (run.get("seeds") or [0])],
            default_model=str(default_model),
            models=models,
            economy=economy,
            kpi=kpi,
            main_agent=str(main_agent_raw) if main_agent_raw else None,
            source_path=str(path),
        )

    def output_dir_for_seed(self, seed: int, root: str = "trajectories") -> Path:
        return Path(root) / self.name / f"seed_{seed}"

    def apply_economy_overrides(self) -> None:
        """Mutate env-module-level constants in-place. Call before build_run.

        Recognised keys (case-sensitive in the TOML):
          - demand_base, demand_hi, demand_sigma
          - inventory_spoilage_per_day
          - late_fee_per_day
          - opex                     -> sets ALL roles' DAILY_OPEX_BY_ROLE values
          - farmer_delivery_extra_days
          - return_window_days
          - roast_cost_per_kg, roast_yield, roast_lag_days — overrides
            the COMMODITY recipe (green_coffee_kg → roasted_coffee_kg)
            in ROAST_RECIPES. Specialty-recipe overrides aren't wired
            into TOML yet; edit `ROAST_RECIPES` directly for those.
          - roast_daily_cap_green_kg
          - delivery_delay_prob, delivery_loss_prob
          - p_res (sets coffee items' retail_reservation_price — applied at item-build time)
        """
        if not self.economy:
            return
        from coffeebench import environment as env_mod

        mapping = {
            "demand_base": ("DEMAND_BASE", float),
            "demand_hi": ("DEMAND_HI", float),
            "demand_sigma": ("DEMAND_SIGMA", float),
            "inventory_spoilage_per_day": ("INVENTORY_SPOILAGE_PER_DAY", float),
            "late_fee_per_day": ("LATE_FEE_PER_DAY", float),
            "farmer_delivery_extra_days": ("FARMER_DELIVERY_EXTRA_DAYS", int),
            "return_window_days": ("RETURN_WINDOW_DAYS", int),
            "roast_daily_cap_green_kg": ("ROAST_DAILY_CAP_GREEN_KG", int),
            "delivery_delay_prob": ("DELIVERY_DELAY_PROB", float),
            "delivery_loss_prob": ("DELIVERY_LOSS_PROB", float),
        }
        # Commodity-recipe override keys → field in ROAST_RECIPES["green_coffee_kg"].
        recipe_mapping = {
            "roast_cost_per_kg": ("labor_cost_per_kg", float),
            "roast_yield": ("yield", float),
            "roast_lag_days": ("lag_days", int),
        }
        for key, value in self.economy.items():
            if key == "opex":
                # Flat opex across all roles.
                for r in env_mod.DAILY_OPEX_BY_ROLE:
                    env_mod.DAILY_OPEX_BY_ROLE[r] = float(value)
                continue
            if key in recipe_mapping:
                field, caster = recipe_mapping[key]
                env_mod.ROAST_RECIPES["green_coffee_kg"][field] = caster(value)
                continue
            if key not in mapping:
                # `p_res` is handled by main._seed_world which reads
                # `economy.p_res` separately; other unknowns are warnings.
                if key != "p_res":
                    print(f"[config] WARN: unknown economy key '{key}' (ignored)")
                continue
            attr, caster = mapping[key]
            setattr(env_mod, attr, caster(value))
