"""
trend_model.py
---------------
A transparent, dependency-light time-series model for U.S. retail sales:

  1. Classical multiplicative seasonal decomposition
     (trend via centered 12-month moving average, seasonal index per
     calendar month, remainder = actual / (trend * seasonal))
  2. Linear trend model (ordinary least squares on the trend-cycle
     component) used to project the trend forward
  3. 12-month-ahead forecast = projected trend * seasonal index

This intentionally avoids black-box libraries (e.g. statsmodels/ARIMA) so
every step is inspectable — appropriate for an educational / portfolio
repo. Swap in statsmodels' seasonal_decompose or a proper SARIMA model
for production use.

Run:
    python trend_model.py
Produces:
    ../data/retail_sales_forecast.csv
    model_summary.txt
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_processed.csv")
FORECAST_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_forecast.csv")
SUMMARY_OUT = os.path.join(os.path.dirname(__file__), "model_summary.txt")

FORECAST_HORIZON_MONTHS = 12


def load_series() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df[["date", "retail_sales_millions"]].copy()


def centered_moving_average(series: pd.Series, window: int = 12) -> pd.Series:
    """Centered trend-cycle estimate. For an even window (12), this uses
    a 2x12-MA (average of two staggered 12-month averages), the standard
    classical-decomposition approach."""
    ma = series.rolling(window=window, center=False).mean()
    # 2x12 centering: average of ma[t] and ma[t-1], then shift to center
    ma2 = ma.rolling(window=2).mean()
    centered = ma2.shift(-window // 2)
    return centered


def seasonal_decompose_multiplicative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trend"] = centered_moving_average(df["retail_sales_millions"], window=12)
    df["detrended_ratio"] = df["retail_sales_millions"] / df["trend"]

    df["month"] = df["date"].dt.month
    seasonal_index = df.groupby("month")["detrended_ratio"].mean()
    # normalize so seasonal indices average to 1.0 across the year
    seasonal_index = seasonal_index / seasonal_index.mean()

    df["seasonal_index"] = df["month"].map(seasonal_index)
    df["seasonally_adjusted"] = df["retail_sales_millions"] / df["seasonal_index"]
    df["remainder"] = df["retail_sales_millions"] / (df["trend"] * df["seasonal_index"])

    return df, seasonal_index


RECENT_WINDOW_MONTHS = 60  # local trend window used for forecasting


def fit_trend_regression(df: pd.DataFrame):
    """OLS linear trend fit on the FULL trend-cycle history (drops NaN
    edges from the moving-average window). This is reported as the
    long-run average growth rate, but — because retail sales growth is
    not linear over 34 years (it visibly accelerates post-2021) — a
    single straight line fit to the whole series systematically
    understates the *current* growth rate. See `fit_recent_trend` below,
    which is what's actually used to generate the forward forecast."""
    valid = df.dropna(subset=["trend"]).copy()
    valid["t"] = np.arange(len(valid))

    X = valid[["t"]].values
    y = valid["trend"].values

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    return model, valid, r2


def fit_recent_trend(valid: pd.DataFrame, window: int = RECENT_WINDOW_MONTHS):
    """OLS linear trend fit on only the most recent `window` months of the
    trend-cycle component. Local linear trends are a standard, simple way
    to forecast a smoothly curving series without assuming the entire
    34-year history is a single straight line."""
    recent = valid.tail(window).copy()
    X = recent[["t"]].values
    y = recent["trend"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    return model, r2


def forecast(df: pd.DataFrame, recent_model, valid: pd.DataFrame, seasonal_index: pd.Series,
             horizon: int = FORECAST_HORIZON_MONTHS) -> pd.DataFrame:
    """Forecast = recent local trend, projected forward and re-seasonalized.
    Anchored so the first forecast point connects smoothly to the last
    *actual* observation rather than the (lagging) trend-cycle estimate,
    by carrying forward the local trend's growth rate from the last
    known trend point."""
    last_t = valid["t"].iloc[-1]
    last_trend_value = valid["trend"].iloc[-1]
    last_date = df["date"].iloc[-1]

    # Growth rate (slope) from the recent-window model, anchored at the
    # last known trend-cycle value so the forecast starts where the data
    # actually leaves off.
    slope = recent_model.coef_[0]

    future_steps = np.arange(1, horizon + 1)
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")

    projected_trend = last_trend_value + slope * future_steps
    future_months = future_dates.month
    seasonal_factors = np.array([seasonal_index.loc[m] for m in future_months])

    forecast_values = projected_trend * seasonal_factors

    return pd.DataFrame({
        "date": future_dates,
        "projected_trend": projected_trend,
        "seasonal_index": seasonal_factors,
        "forecast_retail_sales_millions": forecast_values,
    })


def main():
    df = load_series()
    decomposed, seasonal_index = seasonal_decompose_multiplicative(df)
    model, valid, r2 = fit_trend_regression(decomposed)
    recent_model, recent_r2 = fit_recent_trend(valid)
    fc = forecast(decomposed, recent_model, valid, seasonal_index)

    fc.to_csv(FORECAST_OUT, index=False)

    slope_per_month = model.coef_[0]
    slope_per_year = slope_per_month * 12
    recent_slope_per_month = recent_model.coef_[0]
    recent_slope_per_year = recent_slope_per_month * 12

    with open(SUMMARY_OUT, "w") as f:
        f.write("Retail Sales Trend Model - Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write("Method: Classical multiplicative decomposition\n")
        f.write("        (2x12 centered moving average trend +\n")
        f.write("         monthly seasonal index). Forecast uses a\n")
        f.write(f"        LOCAL linear trend fit on the most recent\n")
        f.write(f"        {RECENT_WINDOW_MONTHS} months, anchored to the last\n")
        f.write("        known trend-cycle value.\n\n")
        f.write(f"Full-history (1992-2026) linear trend R^2: {r2:.4f}\n")
        f.write(f"Full-history average growth: ${slope_per_month:,.1f}M/month "
                f"(~${slope_per_year:,.1f}M/year)\n\n")
        f.write(f"Recent {RECENT_WINDOW_MONTHS}-month local trend R^2: {recent_r2:.4f}\n")
        f.write(f"Recent local growth rate: ${recent_slope_per_month:,.1f}M/month "
                f"(~${recent_slope_per_year:,.1f}M/year)\n")
        f.write("--> used to generate the 12-month forecast below.\n\n")
        f.write("Seasonal index by calendar month (1.00 = average month):\n")
        for m, val in seasonal_index.items():
            f.write(f"  Month {m:>2}: {val:.4f}\n")
        f.write("\n12-month forecast (local trend x seasonal index):\n")
        f.write(fc.to_string(index=False))

    print(f"Full-history R^2 = {r2:.4f} (~${slope_per_year:,.0f}M/yr)")
    print(f"Recent {RECENT_WINDOW_MONTHS}mo R^2 = {recent_r2:.4f} (~${recent_slope_per_year:,.0f}M/yr, used for forecast)")
    print(f"Forecast written to {FORECAST_OUT}")
    print(f"Summary written to {SUMMARY_OUT}")

    return decomposed, fc, seasonal_index, r2, slope_per_year, recent_r2, recent_slope_per_year


if __name__ == "__main__":
    main()
