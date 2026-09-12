"""Offline sample tape + FRED-like macros so the repo runs without network."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _macro_frame(n_months: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    # Local-level GDP with mild trend + cycle; unemployment mean-reverting.
    gdp_growth = 0.004 + 0.003 * np.sin(np.arange(n_months) / 18) + rng.normal(0, 0.004, n_months)
    gdp = 10000 * np.cumprod(1 + gdp_growth)
    unemp = np.zeros(n_months)
    unemp[0] = 5.0
    for t in range(1, n_months):
        unemp[t] = 0.85 * unemp[t - 1] + 0.15 * 5.2 - 8.0 * gdp_growth[t] + rng.normal(0, 0.15)
        unemp[t] = float(np.clip(unemp[t], 3.0, 12.0))
    return pd.DataFrame({"date": dates, "real_gdp": gdp, "unemployment_rate": unemp})


def _loan_frame(macro: pd.DataFrame, n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    asof = macro["date"].iloc[::3].to_numpy()  # quarterly snapshot dates
    snap = rng.choice(asof, size=n)
    ead = rng.uniform(50_000, 500_000, n)
    collateral = ead * rng.uniform(0.4, 1.3, n)
    assets = rng.uniform(80_000, 900_000, n)
    debt = assets * rng.uniform(0.15, 0.95, n)
    ca = rng.uniform(10_000, 200_000, n)
    cl = ca * rng.uniform(0.4, 1.8, n)
    maturity = rng.uniform(1.0, 5.0, n)
    dpd = rng.choice([0, 0, 0, 0, 15, 30, 60, 90, 120], size=n, p=[0.62, 0.1, 0.08, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01])

    m = macro.set_index("date")
    # nearest prior month
    snap_ts = pd.to_datetime(snap)
    idx = m.index.get_indexer(snap_ts, method="ffill")
    gdp_g = m["real_gdp"].pct_change().fillna(0.0).to_numpy()[idx]
    unemp = m["unemployment_rate"].to_numpy()[idx]
    leverage = debt / np.maximum(assets, 1.0)
    liq = ca / np.maximum(cl, 1.0)

    # Latent log-odds plus large idiosyncratic noise so the label is not a feature clone.
    logits = (
        -2.2
        + 1.6 * (leverage - 0.5)
        - 0.8 * (liq - 1.0)
        - 25.0 * gdp_g
        + 0.18 * (unemp - 5.0)
        + rng.normal(0, 0.85, n)
    )
    p = 1 / (1 + np.exp(-logits))
    default_flag = rng.binomial(1, p)
    # charged-off names for a status-based path
    status = np.where(default_flag == 1, "CHGOFF", "PIF")
    status = np.where((default_flag == 0) & (dpd >= 90), "DELQ", status)

    orig_pd = np.clip(p * rng.uniform(0.4, 1.1, n), 0.002, 0.4)
    severity = np.clip(0.35 + 0.25 * (ead / collateral - 0.8) - 4.0 * gdp_g + rng.normal(0, 0.08, n), 0.05, 0.95)

    return pd.DataFrame({
        "loan_id": np.arange(n),
        "asof_date": snap_ts,
        "grossapproval": ead,
        "exposure_at_default": ead,
        "collateral_value": collateral,
        "total_assets": assets,
        "total_debt": debt,
        "current_assets": ca,
        "current_liabilities": cl,
        "maturity_years": maturity,
        "days_past_due": dpd,
        "loanstatus": status,
        "origination_pd": orig_pd,
        "default_flag": default_flag,
        "loss_severity": np.where(default_flag == 1, severity, 0.0),
    })


def write_sample_data(root: str | Path = ".") -> dict:
    root = Path(root)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    macro = _macro_frame()
    loans = _loan_frame(macro)
    macro_path = raw / "historical_macro_data.csv"
    loan_path = raw / "sba_raw.csv"
    sample_macro = raw / "sample_historical_macro_data.csv"
    sample_loans = raw / "sample_sba_raw.csv"
    macro.to_csv(macro_path, index=False)
    loans.to_csv(loan_path, index=False)
    macro.to_csv(sample_macro, index=False)
    loans.to_csv(sample_loans, index=False)
    return {"macro": str(macro_path), "loans": str(loan_path), "n_loans": len(loans)}


if __name__ == "__main__":
    print(write_sample_data())
