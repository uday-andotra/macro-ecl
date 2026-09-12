from __future__ import annotations

import numpy as np
import pandas as pd
import yaml


def estimate_overlay(tape: pd.DataFrame, config_path: str = "config/config.yaml") -> dict:
    """OLS: logit(quarterly charge-off rate) ~ unemp + GDP growth on 2000+ originations.

    Writes coefficients into config ecl.pd_overlay. This is a portfolio-rate elasticity,
    not a loan-level PIT beta.
    """
    t = tape.copy()
    t["asof_date"] = pd.to_datetime(t.get("asof_date"), errors="coerce")
    t = t.dropna(subset=["asof_date"])
    t = t.loc[t["asof_date"] >= "2000-01-01"]
    if t.empty or "default_flag" not in t.columns:
        print("Overlay OLS skipped: no 2000+ rows")
        return {}

    t["q"] = t["asof_date"].dt.to_period("Q")
    g = t.groupby("q").agg(
        dr=("default_flag", "mean"),
        n=("default_flag", "size"),
        unemp=("unemp_lag_1Q", "mean") if "unemp_lag_1Q" in t.columns else ("default_flag", "size"),
        gdp=("realgdp_growth_lag_1Q", "mean") if "realgdp_growth_lag_1Q" in t.columns else ("default_flag", "size"),
    )
    if "unemp_lag_1Q" not in t.columns:
        print("Overlay OLS skipped: no unemp lag on tape")
        return {}

    g = g.loc[g["n"] >= 200].replace([np.inf, -np.inf], np.nan).dropna()
    p = g["dr"].clip(1e-4, 1 - 1e-4)
    y = np.log(p / (1 - p))
    X = np.column_stack([np.ones(len(g)), g["gdp"].to_numpy(), g["unemp"].to_numpy()])
    if len(g) < 12:
        print(f"Overlay OLS skipped: {len(g)} quarters")
        return {}
    try:
        beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    except np.linalg.LinAlgError as e:
        print(f"Overlay OLS failed: {e}")
        return {}

    out = {
        "intercept": float(beta[0]),
        "gdp_elasticity": float(np.clip(beta[1], -20.0, 5.0)),
        "unemp_elasticity": float(np.clip(beta[2], -0.5, 0.5)),
        "n_quarters": int(len(g)),
    }
    print(
        f"Overlay OLS quarters={out['n_quarters']} "
        f"gdp_e={out['gdp_elasticity']:.3f} unemp_e={out['unemp_elasticity']:.3f}"
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.setdefault("ecl", {}).setdefault("pd_overlay", {})
    config["ecl"]["pd_overlay"]["gdp_elasticity"] = out["gdp_elasticity"]
    config["ecl"]["pd_overlay"]["unemp_elasticity"] = out["unemp_elasticity"]
    config["ecl"]["pd_overlay"]["source"] = "ols_quarterly_chargeoff_rate_2000on"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return out
