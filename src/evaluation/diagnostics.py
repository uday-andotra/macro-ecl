from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def tape_profile(df: pd.DataFrame) -> dict:
    out = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "default_rate": float(df["default_flag"].mean()) if "default_flag" in df.columns else None,
        "ead_sum": float(df["exposure_at_default"].sum()) if "exposure_at_default" in df.columns else None,
        "ead_p50": float(df["exposure_at_default"].median()) if "exposure_at_default" in df.columns else None,
        "maturity_mean": float(df["maturity_years"].mean()) if "maturity_years" in df.columns else None,
        "asof_min": str(pd.to_datetime(df["asof_date"], errors="coerce").min()) if "asof_date" in df.columns else None,
        "asof_max": str(pd.to_datetime(df["asof_date"], errors="coerce").max()) if "asof_date" in df.columns else None,
        "source": str(df["source"].dropna().mode().iloc[0]) if "source" in df.columns and df["source"].notna().any() else None,
    }
    if "leverage" in df.columns:
        out["leverage_mean"] = float(df["leverage"].mean())
    if "sba_guaranteed_portion" in df.columns:
        out["guarantee_mean"] = float(df["sba_guaranteed_portion"].mean())
    return out


def print_tape_profile(df: pd.DataFrame, title: str = "TAPE PROFILE") -> dict:
    p = tape_profile(df)
    print(f"\n=== {title} ===")
    for k, v in p.items():
        if isinstance(v, float):
            print(f"  {k:20s} {v:,.6g}" if abs(v) < 1e6 else f"  {k:20s} {v:,.2f}")
        else:
            print(f"  {k:20s} {v}")
    if "default_flag" in df.columns and "exposure_at_default" in df.columns:
        g = df.groupby(df["default_flag"].astype(int))["exposure_at_default"].agg(["count", "sum"])
        print("  by default_flag:")
        print(g.to_string())
    return p


def print_scenario_book(results: dict) -> None:
    print("\n=== SCENARIO BOOK ===")
    print(f"{'scenario':<12} {'ECL':>16} {'cov':>8} {'PD':>8} {'LGD':>8} {'S1':>7} {'S2':>7} {'S3':>7}")
    ead = None
    for name, block in results.items():
        if name in {"Probability_Weighted_ECL", "weight_sum"} or not isinstance(block, dict):
            continue
        df = block.get("Facility_Data")
        if ead is None and df is not None:
            ead = float(df["exposure_at_default"].sum())
        cov = block["Total_ECL"] / ead if ead else float("nan")
        s1 = block.get("Stage1_share", 0)
        s2 = block.get("Stage2_share", 0)
        s3 = block.get("Stage3_share", 0)
        print(
            f"{name:<12} {block['Total_ECL']:>16,.0f} {cov:>7.2%} "
            f"{block['Average_PD']:>7.2%} {block['Average_PIT_LGD']:>7.2%} "
            f"{s1:>6.1%} {s2:>6.1%} {s3:>6.1%}"
        )
    print(f"PWECL {results.get('Probability_Weighted_ECL', 0):,.2f}  weight_sum={results.get('weight_sum')}")


def dump_run_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, pd.Timestamp):
            return str(o)
        return str(o)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_conv)
    print(f"Wrote {path}")
