"""
Prototype Natural Gas Storage Contract Pricer
---------------------------------------------
- Uses monthly end-of-month prices from "Nat_Gas (1) for task 2.csv" (Oct-2020 .. Sep-2024)
- Builds a DAILY price curve via Holt-Winters (ETS) with 12-month additive seasonality
  and extrapolates 12 months beyond last observation.
- Simulates inventory day-by-day with rate & capacity constraints:
    * Injection legs: (start_date, volume)
    * Withdrawal legs: (start_date, volume)
    * Each leg executes over consecutive days from its start, limited by per-day rate and inventory/capacity
- Cash flows:
    * Buy on injection days at price(date)
    * Sell on withdrawal days at price(date)
    * Per-MMBtu injection/withdrawal fees
    * Per-leg fixed transport fees (applied once when the leg first executes)
    * Storage fees:
        - Fixed per calendar month where inventory > 0 on any day
        - Optional variable fee per MMBtu per day held
- Zero interest rates, no holidays/weekends adjustments.

USAGE (CLI examples at bottom) or import `price_storage_contract(...)`.

Dependencies: pandas, numpy, statsmodels (holt-winters only)
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ------------------------- Price Curve Utilities -------------------------

def _load_monthly_prices(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    # Expected columns: "Dates" (mm/dd/yy), "Prices" (float)
    df["Date"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    df = df.drop(columns=["Dates"]).rename(columns={"Prices": "Price"}).sort_values("Date")
    # Align to month-end
    idx = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    s = pd.Series(df["Price"].astype(float).to_numpy(), index=idx).asfreq("M")
    return s

def build_daily_price_curve(csv_path: Path, extra_months: int = 12) -> pd.Series:
    """Fit ETS to monthly series and return a DAILY time series (history + forecast)."""
    monthly = _load_monthly_prices(csv_path)
    model = ExponentialSmoothing(
        monthly, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated"
    )
    fit = model.fit(optimized=True)
    monthly_curve = pd.concat([monthly, fit.forecast(extra_months)])
    # Interpolate daily
    daily_idx = pd.date_range(monthly_curve.index[0], monthly_curve.index[-1], freq="D")
    daily = pd.Series(index=daily_idx, dtype=float)
    daily.loc[monthly_curve.index] = monthly_curve.values
    daily = daily.interpolate(method="time")
    daily.name = "Price"
    return daily

# ------------------------- Contract Structures -------------------------

@dataclass
class Leg:
    start_date: pd.Timestamp
    volume: float               # in MMBtu
    _remaining: float = field(init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)  # for transport fee

    def __post_init__(self):
        self.start_date = pd.Timestamp(self.start_date).normalize()
        self._remaining = float(self.volume)

@dataclass
class StorageFees:
    storage_fixed_fee_per_month: float = 0.0               # applied for each calendar month with inventory > 0
    storage_variable_fee_per_mmbtu_per_day: float = 0.0    # optional, applied to daily inventory

@dataclass
class ActionFees:
    inject_fee_per_mmbtu: float = 0.0
    withdraw_fee_per_mmbtu: float = 0.0
    transport_fee_per_injection_leg: float = 0.0           # charged once per injection leg on first execution day
    transport_fee_per_withdrawal_leg: float = 0.0          # charged once per withdrawal leg on first execution day

@dataclass
class Constraints:
    max_storage: float          # capacity in MMBtu
    inj_rate_per_day: float     # max MMBtu injected per day across all open injection legs
    wdr_rate_per_day: float     # max MMBtu withdrawn per day across all open withdrawal legs

@dataclass
class PriceEnv:
    daily_prices: pd.Series     # daily price curve (pd.Series indexed by pd.Timestamp)

    def price(self, date: pd.Timestamp) -> float:
        date = pd.Timestamp(date).normalize()
        if date not in self.daily_prices.index:
            raise ValueError(f"Price not available for {date.date()}. Range: "
                             f"{self.daily_prices.index[0].date()}..{self.daily_prices.index[-1].date()}")
        return float(self.daily_prices.loc[date])

# ------------------------- Core Pricer -------------------------

def price_storage_contract(
    prices: PriceEnv,
    injection_legs: List[Tuple[str, float]],
    withdrawal_legs: List[Tuple[str, float]],
    constraints: Constraints,
    storage_fees: StorageFees = StorageFees(),
    action_fees: ActionFees = ActionFees(),
    simulation_end_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Simulate daily inventory & cash flows given legs and constraints, return valuation breakdown.

    Args:
        prices: PriceEnv with daily price curve.
        injection_legs: list of (start_date, volume) for injections (volume in MMBtu).
        withdrawal_legs: list of (start_date, volume) for withdrawals (volume in MMBtu).
        constraints: capacity & daily rate constraints.
        storage_fees: fixed monthly and optional variable per-MMBtu-per-day fee.
        action_fees: per-MMBtu and per-leg transport fees.
        simulation_end_date: optional override for end date (YYYY-MM-DD). If None, will run until:
            max(last price date, latest start + days needed by total volume/rate).

    Returns (dict):
        {
            "gross_sales": ..., "gross_purchases": ..., "inject_fees": ..., "withdraw_fees": ...,
            "transport_fees": ..., "storage_fixed_fees": ..., "storage_variable_fees": ...,
            "net_value": ..., "executed_injection": ..., "executed_withdrawal": ...,
            "unfilled_injection": ..., "unfilled_withdrawal": ...
        }
    """
    # Prepare legs
    inj_legs = [Leg(d, v) for d, v in injection_legs]
    wdr_legs = [Leg(d, v) for d, v in withdrawal_legs]

    if not inj_legs and not wdr_legs:
        return {"net_value": 0.0}

    start_date = min([l.start_date for l in inj_legs + wdr_legs])
    # Estimate required days if end not given
    def days_needed(total, rate):
        return int(np.ceil(total / max(rate, 1e-12)))

    total_inj = sum(l.volume for l in inj_legs)
    total_wdr = sum(l.volume for l in wdr_legs)
    est_end = max([l.start_date for l in inj_legs + wdr_legs]) + pd.Timedelta(days=max(days_needed(total_inj, constraints.inj_rate_per_day),
                                                                                      days_needed(total_wdr, constraints.wdr_rate_per_day)))
    last_price_day = prices.daily_prices.index[-1]
    end_date = pd.Timestamp(simulation_end_date).normalize() if simulation_end_date else max(est_end, last_price_day)

    # State & accounting
    dates = pd.date_range(start_date, end_date, freq="D")
    inventory = 0.0
    gross_purchases = 0.0
    gross_sales = 0.0
    inject_fees = 0.0
    withdraw_fees = 0.0
    transport_fees = 0.0
    storage_fixed_fee_months: set = set()
    storage_variable_fees = 0.0

    # For variable storage fee, track daily inventory
    for day in dates:
        # Determine inventory>0 -> month counted for fixed fee
        if inventory > 0:
            storage_fixed_fee_months.add((day.year, day.month))
            if storage_fees.storage_variable_fee_per_mmbtu_per_day:
                storage_variable_fees += storage_fees.storage_variable_fee_per_mmbtu_per_day * inventory

        # Apply injection flows
        inj_cap = constraints.inj_rate_per_day
        for leg in inj_legs:
            if inj_cap <= 0:
                break
            if leg._remaining <= 0 or day < leg.start_date:
                continue
            # Volume possible today bounded by leg remaining, per-day cap, and capacity headroom
            headroom = constraints.max_storage - inventory
            if headroom <= 0:
                break
            vol = min(leg._remaining, inj_cap, headroom)
            if vol > 0:
                price = prices.price(day)
                gross_purchases += price * vol
                inject_fees += action_fees.inject_fee_per_mmbtu * vol
                if not leg._started and action_fees.transport_fee_per_injection_leg:
                    transport_fees += action_fees.transport_fee_per_injection_leg
                    leg._started = True
                inventory += vol
                leg._remaining -= vol
                inj_cap -= vol

        # Apply withdrawal flows
        wdr_cap = constraints.wdr_rate_per_day
        for leg in wdr_legs:
            if wdr_cap <= 0:
                break
            if leg._remaining <= 0 or day < leg.start_date:
                continue
            if inventory <= 0:
                break
            vol = min(leg._remaining, wdr_cap, inventory)
            if vol > 0:
                price = prices.price(day)
                gross_sales += price * vol
                withdraw_fees += action_fees.withdraw_fee_per_mmbtu * vol
                if not leg._started and action_fees.transport_fee_per_withdrawal_leg:
                    transport_fees += action_fees.transport_fee_per_withdrawal_leg
                    leg._started = True
                inventory -= vol
                leg._remaining -= vol
                wdr_cap -= vol

    # After loop: charge fixed storage fees per active months
    storage_fixed_fees = storage_fees.storage_fixed_fee_per_month * len(storage_fixed_fee_months)

    executed_injection = sum(l.volume - l._remaining for l in inj_legs)
    executed_withdrawal = sum(l.volume - l._remaining for l in wdr_legs)
    unfilled_injection = sum(max(l._remaining, 0.0) for l in inj_legs)
    unfilled_withdrawal = sum(max(l._remaining, 0.0) for l in wdr_legs)

    net_value = (
        gross_sales
        - gross_purchases
        - inject_fees
        - withdraw_fees
        - transport_fees
        - storage_fixed_fees
        - storage_variable_fees
    )

    return {
        "gross_sales": float(gross_sales),
        "gross_purchases": float(gross_purchases),
        "inject_fees": float(inject_fees),
        "withdraw_fees": float(withdraw_fees),
        "transport_fees": float(transport_fees),
        "storage_fixed_fees": float(storage_fixed_fees),
        "storage_variable_fees": float(storage_variable_fees),
        "net_value": float(net_value),
        "executed_injection": float(executed_injection),
        "executed_withdrawal": float(executed_withdrawal),
        "unfilled_injection": float(unfilled_injection),
        "unfilled_withdrawal": float(unfilled_withdrawal),
    }

# ------------------------- Convenience Builder & CLI -------------------------

def _find_csv() -> Path:
    candidates = [
        Path("Nat_Gas (1) for task 2.csv"),
        Path("./Nat_Gas (1) for task 2.csv"),
        Path("/mnt/data/Nat_Gas (1) for task 2.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find 'Nat_Gas (1) for task 2.csv'.")

def build_env() -> PriceEnv:
    csv_path = _find_csv()
    daily_curve = build_daily_price_curve(csv_path, extra_months=12)
    return PriceEnv(daily_curve)

if __name__ == "__main__":
    # Example scenario: buy in summer, sell in winter
    env = build_env()
    # Legs: (start_date, volume in MMBtu)
    injection_legs = [
        ("2024-06-15", 1_000_000.0),
    ]
    withdrawal_legs = [
        ("2024-12-15", 1_000_000.0),
    ]

    constraints = Constraints(
        max_storage=1_200_000.0,
        inj_rate_per_day=400_000.0,
        wdr_rate_per_day=400_000.0,
    )

    storage_fees = StorageFees(
        storage_fixed_fee_per_month=100_000.0,            # e.g., $100k per active month
        storage_variable_fee_per_mmbtu_per_day=0.0,       # optional, set if needed
    )

    action_fees = ActionFees(
        inject_fee_per_mmbtu=0.01,                        # $0.01 per MMBtu injection
        withdraw_fee_per_mmbtu=0.01,                      # $0.01 per MMBtu withdrawal
        transport_fee_per_injection_leg=50_000.0,         # $50k per injection leg
        transport_fee_per_withdrawal_leg=50_000.0,        # $50k per withdrawal leg
    )

    result = price_storage_contract(
        env,
        injection_legs,
        withdrawal_legs,
        constraints,
        storage_fees,
        action_fees,
        simulation_end_date=None,  # let it auto-extend within available price curve
    )

    # Print a tidy breakdown
    print("---- Storage Contract Valuation ----")
    for k, v in result.items():
        print(f"{k:24s}: {v:,.2f}")
