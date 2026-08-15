"""
process_data.py
----------------
Loads the raw U.S. Advance Retail Sales series (RSAFS, source: U.S. Census
Bureau via FRED) and engineers analysis-ready features:

  - month-over-month (MoM) % change
  - year-over-year (YoY) % change
  - 12-month rolling average (trend-cycle smoothing)
  - 12-month rolling std (volatility)
  - recession flag for well-known NBER recession windows that fall in range

Run:
    python process_data.py
Produces:
    retail_sales_processed.csv
"""

import pandas as pd
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "retail_sales_raw.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "retail_sales_processed.csv")

# NBER-dated US recessions that overlap our 1992-2026 sample window
RECESSION_WINDOWS = [
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["mom_pct_change"] = df["retail_sales_millions"].pct_change() * 100
    df["yoy_pct_change"] = df["retail_sales_millions"].pct_change(periods=12) * 100

    df["rolling_12m_avg"] = df["retail_sales_millions"].rolling(window=12).mean()
    df["rolling_12m_std"] = df["retail_sales_millions"].rolling(window=12).std()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df["is_recession"] = False
    for start, end in RECESSION_WINDOWS:
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "is_recession"] = True

    return df


def main():
    raw = load_raw()
    processed = engineer_features(raw)
    processed.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(processed)} rows to {OUT_PATH}")
    print(processed.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
