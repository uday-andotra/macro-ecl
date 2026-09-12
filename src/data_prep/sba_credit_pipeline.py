from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from src.data_prep.clean_loan_tape import clean_sba_data
from src.data_prep.make_sample_data import write_sample_data
from src.data_prep.transform_macro_data import stationarize_macro_data

LOAN_CANDIDATES = [
    "data/raw/SBAnational.csv",
    "data/raw/sbanational.csv",
    "data/raw/sba_raw.csv",
    "data/raw/sample_sba_raw.csv",
]
MACRO_CANDIDATES = [
    "data/raw/historical_macro_data.csv",
    "data/raw/sample_historical_macro_data.csv",
]


def resolve_raw_paths(config: dict, root: Path) -> tuple[Path, Path, str]:
    configured_loan = root / config["paths"]["raw_loan_data"]
    configured_macro = root / config["paths"]["raw_macro_data"]
    loan = configured_loan if configured_loan.exists() else None
    if loan is None:
        for rel in LOAN_CANDIDATES:
            p = root / rel
            if p.exists():
                loan = p
                break
    macro = configured_macro if configured_macro.exists() else None
    if macro is None:
        for rel in MACRO_CANDIDATES:
            p = root / rel
            if p.exists():
                macro = p
                break
    mode = "configured"
    if loan is None or macro is None:
        write_sample_data(root)
        loan = root / "data/raw/sba_raw.csv"
        macro = root / "data/raw/historical_macro_data.csv"
        mode = "sample_fallback"
    print(f"RAW FILES used_mode={mode}")
    print(f"  loans : {loan}")
    print(f"  macros: {macro}  (expected FRED GDPC1 + UNRATE extract)")
    return loan, macro, mode


def attach_macro_asof(loans: pd.DataFrame, macro_lagged: pd.DataFrame, lag_cols: list) -> pd.DataFrame:
    have = [c for c in lag_cols if c in macro_lagged.columns]
    clean = macro_lagged.dropna(subset=have) if have else macro_lagged.dropna()
    if clean.empty:
        last = macro_lagged.iloc[-1].copy()
        for c in lag_cols:
            last[c] = 0.0
        print("macro lag frame empty after dropna; filled lags with 0")
    else:
        last = clean.iloc[-1]
    if "asof_date" not in loans.columns:
        for c in lag_cols:
            loans[c] = last[c]
        return loans

    loans = loans.copy()
    loans["asof_date"] = pd.to_datetime(loans["asof_date"], errors="coerce")
    missing = loans["asof_date"].isna()
    if missing.all():
        for c in lag_cols:
            loans[c] = last[c]
        print(f"asof_date missing on all {len(loans)} rows; used last FRED observation")
        return loans

    m = macro_lagged.reset_index().rename(columns={"date": "macro_date"})
    m["macro_date"] = pd.to_datetime(m["macro_date"], errors="coerce")
    m = m.dropna(subset=["macro_date"]).sort_values("macro_date")

    dated = loans.loc[~missing].sort_values("asof_date")
    merged = pd.merge_asof(
        dated,
        m[["macro_date"] + lag_cols],
        left_on="asof_date",
        right_on="macro_date",
        direction="backward",
    )
    if missing.any():
        orphan = loans.loc[missing].copy()
        for c in lag_cols:
            orphan[c] = last[c]
        merged = pd.concat([merged, orphan], ignore_index=True)
        print(f"asof_date missing on {int(missing.sum()):,} rows; filled with last FRED observation")
    return merged


def run_pipeline(config_path: str = "config/config.yaml", root: str = ".") -> pd.DataFrame:
    root_p = Path(root)
    with open(root_p / config_path) as f:
        config = yaml.safe_load(f)

    loan_path, macro_path, mode = resolve_raw_paths(config, root_p)
    os.makedirs(root_p / "data/processed", exist_ok=True)

    loans = clean_sba_data(str(loan_path))
    lagged, optimal_lags, summary, alpha = stationarize_macro_data(
        str(macro_path), config_path=str(root_p / config_path)
    )
    tape = attach_macro_asof(loans, lagged, optimal_lags)
    for c in optimal_lags:
        tape[c] = tape[c].astype(float)
        tape[c] = tape[c].fillna(tape[c].median())
    tape.attrs["raw_loan_file"] = str(loan_path)
    tape.attrs["raw_macro_file"] = str(macro_path)
    tape.attrs["raw_mode"] = mode
    out = root_p / config["paths"]["processed_tape"]
    tape.to_csv(out, index=False)
    print(f"Processed tape: {out} rows={len(tape)} lags={optimal_lags} lasso_alpha={alpha:.4f}")
    from src.evaluation.diagnostics import print_tape_profile
    print("LASSO lag table (top 8):")
    print(summary.head(8).to_string(index=False))
    print_tape_profile(tape, "PROCESSED TAPE")
    return tape


if __name__ == "__main__":
    run_pipeline()
