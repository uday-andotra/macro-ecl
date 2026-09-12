from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_prep.transform_macro_data import load_macro
from src.macro_engine.var_model import clean_var_panel, fit_var_model


def _selected_lag_names(config: dict) -> list:
    return list(config["model_features"]["macro_lags"])


def _base_from_lag(name: str) -> str:
    return name.rsplit("_lag_", 1)[0]


def _quarterly_panel(macro: pd.DataFrame) -> pd.DataFrame:
    if "realgdp_growth" in macro.columns and "unemp" in macro.columns:
        panel = macro[["realgdp_growth", "unemp"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(panel) >= 12:
            return clean_var_panel(panel, ["realgdp_growth", "unemp"])
    levels = macro[["real_gdp", "unemp"]].replace([np.inf, -np.inf], np.nan).dropna()
    try:
        levels["realgdp_growth"] = levels["real_gdp"].pct_change(fill_method=None)
    except TypeError:
        levels["realgdp_growth"] = levels["real_gdp"].pct_change()
    return clean_var_panel(levels, ["realgdp_growth", "unemp"])


def _baseline_from_history(panel: pd.DataFrame) -> dict:
    return {
        "realgdp_growth": float(panel["realgdp_growth"].iloc[-1]),
        "unemp": float(panel["unemp"].iloc[-1]),
    }


def generate_scenarios(config_path: str = "config/config.yaml", root: str = ".") -> pd.DataFrame:
    root_p = Path(root)
    with open(root_p / config_path) as f:
        config = yaml.safe_load(f)

    macro = load_macro(str(root_p / config["paths"]["raw_macro_data"]))
    panel = _quarterly_panel(macro)
    used = "history"
    try:
        results = fit_var_model(panel, ["realgdp_growth", "unemp"])
        k = int(results.k_ar)
        tail = clean_var_panel(panel, ["realgdp_growth", "unemp"]).iloc[-k:]
        forecast = results.forecast(tail.to_numpy(), steps=1)[0]
        if not np.isfinite(forecast).all():
            raise ValueError("non-finite VAR forecast")
        baseline = {"realgdp_growth": float(forecast[0]), "unemp": float(forecast[1])}
        used = f"VAR(p={k})"
    except Exception as e:
        print(f"VAR fit skipped ({e}); using last quarterly observation as baseline")
        baseline = _baseline_from_history(panel)

    rows = []
    for name, shocks in config["stress_scenarios"].items():
        gdp = baseline["realgdp_growth"] + float(shocks.get("gdp_shock", 0.0))
        unemp = baseline["unemp"] + float(shocks.get("unemp_shock", 0.0))
        row = {"scenario_name": name, "realgdp_growth": gdp, "unemp": unemp}
        for lag_name in _selected_lag_names(config):
            base = _base_from_lag(lag_name)
            row[lag_name] = gdp if "gdp" in base else unemp
        rows.append(row)

    out = root_p / config["paths"]["macro_forecasts"]
    os.makedirs(out.parent, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Scenarios ({used}) written to {out}\n{df.to_string(index=False)}")
    return df


if __name__ == "__main__":
    generate_scenarios()
