"""
Natural Gas Price Estimator (Monthly → Daily with 12M Extrapolation)

- Uses the uploaded monthly end-of-month prices (Oct-2020 .. Sep-2024) from nat_gas.csv
- Fits Holt-Winters (ETS) with additive trend + 12-month seasonality
- Forecasts 12 additional months
- Builds a DAILY curve via time interpolation
- Exposes: estimate_price(date_str) -> float
- CLI: pass a date as YYYY-MM-DD to get an estimate

Dependencies: pandas, numpy, statsmodels
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --------------------- Locate CSV ---------------------
def _find_csv() -> Path:
    candidates = [
        Path("nat_gas.csv"),
        Path("./nat_gas.csv"),
        Path("/mnt/data/nat_gas.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find 'nat_gas.csv'. Place it next to this script or provide /mnt/data/nat_gas.csv."
    )

CSV_PATH = _find_csv()

# --------------------- Build curves ---------------------
def _load_monthly(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    # Provided columns: Dates (mm/dd/yy), Prices
    df["Date"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    df = df.drop(columns=["Dates"]).rename(columns={"Prices": "Price"}).sort_values("Date")
    # Force month-end index to be explicit month-end timestamps
    idx = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    s = pd.Series(df["Price"].to_numpy(float), index=idx).asfreq("M")
    return s

def _fit_and_extend(monthly_raw: pd.Series, extra_months: int = 12) -> Tuple[pd.Series, pd.Series]:
    model = ExponentialSmoothing(
        monthly_raw,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    future = fit.forecast(extra_months)
    monthly_curve = pd.concat([monthly_raw, future])
    # Build a daily curve via time interpolation between monthly knots
    daily_idx = pd.date_range(monthly_curve.index[0], monthly_curve.index[-1], freq="D")
    daily_curve = pd.Series(index=daily_idx, dtype=float)
    daily_curve.loc[monthly_curve.index] = monthly_curve.values
    daily_curve = daily_curve.interpolate(method="time")
    return monthly_curve, daily_curve

# Build global curves once
_MONTHLY_RAW = _load_monthly(CSV_PATH)
_MONTHLY_CURVE, _DAILY_CURVE = _fit_and_extend(_MONTHLY_RAW, extra_months=12)

# --------------------- Public API ---------------------
def estimate_price(date_str: str) -> float:
    """
    Return estimated natural gas price for any date within
    [{start} .. {end}] based on monthly data + 12M extrapolation.
    """
    ts = pd.Timestamp(date_str).normalize()
    if ts < _DAILY_CURVE.index[0] or ts > _DAILY_CURVE.index[-1]:
        raise ValueError(
            f"Date out of range. Allowed: {_DAILY_CURVE.index[0].date()} "
            f"to {_DAILY_CURVE.index[-1].date()}"
        )
    return float(_DAILY_CURVE.loc[ts])

# --------------------- CLI ---------------------
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        query_date = sys.argv[1]
    else:
        query_date = input("Enter date (YYYY-MM-DD): ").strip()
    try:
        value = estimate_price(query_date)
        print(f"{query_date}: {value:.6f}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
