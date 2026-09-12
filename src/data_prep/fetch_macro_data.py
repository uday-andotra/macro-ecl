"""Pull FRED series used by this project.

Series
------
GDPC1  Real Gross Domestic Product, quarterly
UNRATE Civilian Unemployment Rate, monthly

Source: Federal Reserve Bank of St. Louis FRED (https://fred.stlouisfed.org).
Resampled to month-start with forward-fill so GDP can join a monthly as-of date.
"""
from __future__ import annotations

import datetime
import os


def fetch_fred_data(output_path: str = "data/raw/historical_macro_data.csv") -> str:
    try:
        import pandas_datareader.data as web
    except ImportError as e:
        raise ImportError("pandas-datareader is required to fetch FRED data") from e

    start_date = datetime.datetime(1987, 1, 1)
    end_date = datetime.datetime.today()
    print(
        "FRED extract: GDPC1 (real GDP), UNRATE (unemployment) "
        f"{start_date.date()} → {end_date.date()}"
    )
    df = web.DataReader(["GDPC1", "UNRATE"], "fred", start_date, end_date)
    df = df.resample("MS").ffill().dropna()
    df = df.rename(columns={"GDPC1": "real_gdp", "UNRATE": "unemployment_rate"})
    df.index.name = "date"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path)
    print(f"Wrote {output_path} rows={len(df)}")
    return output_path


if __name__ == "__main__":
    print(fetch_fred_data())
