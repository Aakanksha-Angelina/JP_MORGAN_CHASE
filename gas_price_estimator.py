
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def load_and_build(csv_path: str):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
    df = df.drop(columns=["Dates"]).rename(columns={"Prices": "Price"}).sort_values("Date")
    idx = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    monthly_raw = pd.Series(df["Price"].values, index=idx).asfreq("M")
    model = ExponentialSmoothing(monthly_raw, trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated")
    fit = model.fit(optimized=True)
    monthly_curve = pd.concat([monthly_raw, fit.forecast(12)])
    daily_index = pd.date_range(monthly_curve.index[0], monthly_curve.index[-1], freq="D")
    daily_curve = pd.Series(index=daily_index, dtype=float)
    daily_curve.loc[monthly_curve.index] = monthly_curve.values
    daily_curve = daily_curve.interpolate(method="time")
    return monthly_raw, fit, monthly_curve, daily_curve

class GasPriceEstimator:
    def __init__(self, csv_path: str):
        self.monthly_raw, self.fit, self.monthly_curve, self.daily_curve = load_and_build(csv_path)
    def estimate_price(self, date_str: str) -> float:
        ts = pd.Timestamp(date_str).normalize()
        if ts < self.daily_curve.index[0] or ts > self.daily_curve.index[-1]:
            raise ValueError(f"Date out of range. Allowed: {self.daily_curve.index[0].date()} to {self.daily_curve.index[-1].date()}")
        return float(self.daily_curve.loc[ts])
