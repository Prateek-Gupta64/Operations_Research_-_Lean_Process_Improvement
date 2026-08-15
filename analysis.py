"""
analysis.py
------------
Generates all charts used in the final report from the processed data
and the trend model's outputs.

Run:
    python analysis.py
Produces PNGs in ./charts/
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from trend_model import (  # noqa: E402
    load_series, seasonal_decompose_multiplicative, fit_trend_regression,
    fit_recent_trend, forecast
)

BASE = os.path.dirname(__file__)
CHARTS_DIR = os.path.join(BASE, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

PROCESSED_PATH = os.path.join(BASE, "data", "retail_sales_processed.csv")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "font.size": 11,
})

BLUE = "#1f77b4"
RED = "#d62728"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
GREY = "#7f7f7f"

billions_fmt = mticker.FuncFormatter(lambda x, _: f"${x/1000:,.0f}B")


def savefig(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def chart_raw_series(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["retail_sales_millions"], color=BLUE, linewidth=1.3)
    ax.set_title("U.S. Advance Retail Sales, 1992–2026")
    ax.set_ylabel("Retail Sales")
    ax.yaxis.set_major_formatter(billions_fmt)
    ax.set_xlabel("Year")
    savefig(fig, "01_raw_series.png")


def chart_yoy_growth(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["yoy_pct_change"], color=RED, linewidth=1.1)
    ax.axhline(0, color="black", linewidth=0.8)
    # group into contiguous recession blocks (block id increments each
    # time is_recession flips from False->True or True->False)
    block_id = (df["is_recession"] != df["is_recession"].shift()).cumsum()
    for _, grp in df[df["is_recession"]].groupby(block_id[df["is_recession"]]):
        ax.axvspan(grp["date"].min(), grp["date"].max(), color=GREY, alpha=0.3)
    ax.set_title("Year-over-Year Retail Sales Growth (%)")
    ax.set_ylabel("YoY % change")
    ax.set_xlabel("Year")
    savefig(fig, "02_yoy_growth.png")


def chart_rolling_avg(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["retail_sales_millions"], color=GREY, alpha=0.5, linewidth=1, label="Monthly (actual)")
    ax.plot(df["date"], df["rolling_12m_avg"], color=BLUE, linewidth=2, label="12-month rolling avg")
    ax.yaxis.set_major_formatter(billions_fmt)
    ax.set_title("Retail Sales: Actual vs. 12-Month Rolling Average")
    ax.set_xlabel("Year")
    ax.legend()
    savefig(fig, "03_rolling_average.png")


def chart_seasonal_index(seasonal_index):
    fig, ax = plt.subplots(figsize=(8, 5))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    vals = [seasonal_index.loc[m] for m in range(1, 13)]
    colors = [GREEN if v >= 1 else ORANGE for v in vals]
    ax.bar(months, vals, color=colors)
    ax.axhline(1.0, color="black", linewidth=0.8)
    span = max(0.01, max(abs(v - 1) for v in vals) * 1.4)
    ax.set_ylim(1 - span, 1 + span)
    ax.set_title("Residual Seasonal Pattern by Month (1.00 = typical month)\n"
                 "Small by design — source series is already seasonally adjusted",
                 fontsize=11)
    ax.set_ylabel("Seasonal index")
    savefig(fig, "04_seasonal_index.png")


def chart_decomposition(decomposed):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(decomposed["date"], decomposed["retail_sales_millions"], color=BLUE)
    axes[0].set_title("Observed")
    axes[0].yaxis.set_major_formatter(billions_fmt)

    axes[1].plot(decomposed["date"], decomposed["trend"], color=RED)
    axes[1].set_title("Trend-cycle (2x12 centered moving average)")
    axes[1].yaxis.set_major_formatter(billions_fmt)

    axes[2].plot(decomposed["date"], decomposed["remainder"], color=GREY, linewidth=0.9)
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].set_title("Remainder (noise, after removing trend & seasonality)")
    axes[2].set_xlabel("Year")

    fig.suptitle("Classical Multiplicative Decomposition", y=1.02, fontsize=13)
    savefig(fig, "05_decomposition.png")


def chart_forecast(decomposed, fc):
    fig, ax = plt.subplots(figsize=(10, 5))
    recent = decomposed[decomposed["date"] >= "2018-01-01"]
    ax.plot(recent["date"], recent["retail_sales_millions"], color=BLUE, label="Actual")
    ax.plot(fc["date"], fc["forecast_retail_sales_millions"], color=RED,
            linestyle="--", marker="o", markersize=3, label="12-month forecast")
    ax.yaxis.set_major_formatter(billions_fmt)
    ax.set_title("Retail Sales: Actual (since 2018) + 12-Month Forecast")
    ax.set_xlabel("Year")
    ax.legend()
    savefig(fig, "06_forecast.png")


def chart_covid_zoom(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    window = df[(df["date"] >= "2019-06-01") & (df["date"] <= "2021-06-01")]
    ax.plot(window["date"], window["retail_sales_millions"], color=BLUE, marker="o", markersize=3)
    ax.yaxis.set_major_formatter(billions_fmt)
    ax.set_title("Detail: COVID-19 Retail Sales Shock and Recovery (2019–2021)")
    ax.set_xlabel("Date")
    savefig(fig, "07_covid_zoom.png")


def chart_mom_volatility(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [GREEN if v >= 0 else RED for v in df["mom_pct_change"].fillna(0)]
    ax.bar(df["date"], df["mom_pct_change"], color=colors, width=20)
    ax.set_title("Month-over-Month % Change (Volatility View)")
    ax.set_ylabel("MoM % change")
    ax.set_xlabel("Year")
    savefig(fig, "08_mom_volatility.png")


def main():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])

    raw = load_series()
    decomposed, seasonal_index = seasonal_decompose_multiplicative(raw)
    model, valid, r2 = fit_trend_regression(decomposed)
    recent_model, recent_r2 = fit_recent_trend(valid)
    fc = forecast(decomposed, recent_model, valid, seasonal_index)

    chart_raw_series(df)
    chart_yoy_growth(df)
    chart_rolling_avg(df)
    chart_seasonal_index(seasonal_index)
    chart_decomposition(decomposed)
    chart_forecast(decomposed, fc)
    chart_covid_zoom(df)
    chart_mom_volatility(df)

    print("All charts generated.")


if __name__ == "__main__":
    main()
