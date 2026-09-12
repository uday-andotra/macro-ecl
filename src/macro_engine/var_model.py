from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def clean_var_panel(macro_df: pd.DataFrame, variables: list) -> pd.DataFrame:
    panel = macro_df[list(variables)].replace([np.inf, -np.inf], np.nan).dropna()
    panel = panel.loc[np.isfinite(panel.to_numpy()).all(axis=1)]
    if "realgdp_growth" in panel.columns:
        panel["realgdp_growth"] = panel["realgdp_growth"].clip(-0.2, 0.2)
    if "unemp" in panel.columns:
        panel["unemp"] = panel["unemp"].clip(0.0, 30.0)
    return panel


def fit_var_model(macro_df: pd.DataFrame, variables: list, maxlags: int = 4):
    panel = clean_var_panel(macro_df, variables)
    if len(panel) < maxlags + 15:
        raise ValueError(f"Not enough clean rows to fit VAR on {variables}: {len(panel)}")
    if isinstance(panel.index, pd.DatetimeIndex):
        try:
            panel = panel.asfreq("QS").ffill().dropna()
        except Exception:
            try:
                panel = panel.asfreq("Q").ffill().dropna()
            except Exception:
                pass
    model = VAR(panel)
    try:
        return model.fit(maxlags=min(int(maxlags), 4), ic="aic")
    except Exception:
        return model.fit(1)
