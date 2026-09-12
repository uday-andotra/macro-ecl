from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    return df[~df.index.duplicated(keep="last")]


def _is_quarterly(index: pd.DatetimeIndex) -> bool:
    if len(index) < 3:
        return False
    gaps = pd.Series(index).diff().dt.days.dropna()
    med = float(gaps.median())
    return 70 <= med <= 120


def load_macro(input_path: str) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    date_col = "date" if "date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).rename(columns={date_col: "date"}).sort_values("date")
    raw = raw.set_index("date")
    if "real_gdp" not in raw.columns or "unemployment_rate" not in raw.columns:
        raise ValueError("Macro file must contain real_gdp and unemployment_rate")
    raw["unemp"] = pd.to_numeric(raw["unemployment_rate"], errors="coerce")
    raw["real_gdp"] = pd.to_numeric(raw["real_gdp"], errors="coerce")
    raw = _ensure_datetime_index(raw)
    try:
        raw["realgdp_growth"] = raw["real_gdp"].pct_change(fill_method=None)
    except TypeError:
        raw["realgdp_growth"] = raw["real_gdp"].pct_change()
    print(
        f"Macro loaded rows={len(raw)} quarterly={_is_quarterly(raw.index)} "
        f"growth_nonnull={int(raw['realgdp_growth'].notna().sum())} "
        f"{raw.index.min().date()}→{raw.index.max().date()}"
    )
    return raw


def quarterly_for_lasso(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["realgdp_growth", "unemp"] if c in df.columns]
    q = _ensure_datetime_index(df[cols])
    return q.replace([np.inf, -np.inf], np.nan).dropna()


def _fallback(base_cols: List[str], reason: str):
    print(f"LASSO skipped ({reason}); using lag_1Q")
    names = [f"{c}_lag_1Q" for c in base_cols]
    summary = pd.DataFrame({
        "candidate": names,
        "coef": [0.0] * len(names),
        "abs_weight": [0.0] * len(names),
        "selected": [True] * len(names),
    })
    return names, summary, float("nan")


def run_lasso_optimization(df: pd.DataFrame, base_cols: List[str], max_lags: int = 4):
    panel = quarterly_for_lasso(df)
    print(f"LASSO quarterly panel rows={len(panel)}")
    if len(panel) < max_lags + 8:
        return _fallback(base_cols, f"panel too short: {len(panel)}")

    X = pd.DataFrame({
        f"{col}_lag_{lag}Q": panel[col].shift(lag)
        for col in base_cols if col in panel.columns
        for lag in range(1, max_lags + 1)
    }, index=panel.index)
    if X.empty:
        return _fallback(base_cols, "no lag columns")

    y = panel["unemp"].diff().shift(-1)
    aligned = X.join(y.rename("y")).dropna()
    if len(aligned) < 12:
        return _fallback(base_cols, f"aligned rows={len(aligned)}")
    X, y = aligned.drop(columns=["y"]), aligned["y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=min(5, len(X)), random_state=42, max_iter=20000, alphas=np.logspace(-4, 1, 30)).fit(X_scaled, y)
    coef = pd.Series(lasso.coef_, index=X.columns)

    optimal = []
    for base in base_cols:
        block = coef.filter(like=f"{base}_lag_")
        nz = block[block != 0]
        optimal.append(nz.abs().idxmax() if not nz.empty else f"{base}_lag_1Q")

    summary = pd.DataFrame({
        "candidate": coef.index,
        "coef": coef.values,
        "abs_weight": np.abs(coef.values),
        "selected": coef.index.isin(optimal),
    }).sort_values("abs_weight", ascending=False)
    return optimal, summary, float(lasso.alpha_)


def stationarize_macro_data(input_path: str, config_path: str = "config/config.yaml"):
    df = load_macro(input_path)
    optimal_lags, summary_df, alpha = run_lasso_optimization(df, ["realgdp_growth", "unemp"])

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.setdefault("model_features", {})["macro_lags"] = optimal_lags
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    lagged = df.copy()
    for name in optimal_lags:
        parts = name.rsplit("_lag_", 1)
        base, lag = parts[0], int(parts[1].replace("Q", ""))
        lagged[name] = df[base].shift(lag) if base in df.columns else np.nan
        lagged[name] = lagged[name].ffill().bfill()
    return lagged, optimal_lags, summary_df, alpha
