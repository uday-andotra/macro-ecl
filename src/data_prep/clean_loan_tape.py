"""Normalize a loan tape into the engine schema.

Supported raw layouts
---------------------
1. Kaggle / Li–Mickel–Taylor "Should This Loan Be Approved or Denied?"
   file typically named SBAnational.csv
   https://www.kaggle.com/datasets (search: SBAnational)
   Outcome: MIS_Status in {P I F, CHGOFF}
   Size:    GrAppv / DisbursementGross (currency strings)
   Term:    months
   Dates:   ApprovalDate, DisbursementDate, ChgOffDate
   There are NO current_assets / total_debt columns on this file.

2. Already-cleaned engine tape (sample or prior run) with
   exposure_at_default, default_flag, financial ratios.

Proxies used only when Kaggle columns are present and GAAP items are absent
are named and logged — they are not FOIA financials.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KAGGLE_HINTS = {
    "mis_status",
    "grappv",
    "sba_appv",
    "chgoffpringr",
    "disbursementgross",
    "loannr_chkdgt",
}


def _num(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    if getattr(s, "dtype", None) == object or getattr(s.dtype, "kind", None) in "OSU":
        cleaned = s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(s, errors="coerce")


def _parse_date(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    out = pd.to_datetime(raw, format="%d-%b-%y", errors="coerce")
    missing = out.isna() & raw.notna() & ~raw.isin(["", "nan", "None"])
    if missing.any():
        out.loc[missing] = pd.to_datetime(raw.loc[missing], errors="coerce")
    y = out.dt.year
    out = out.mask(y > 2014, out - pd.DateOffset(years=100))
    y = out.dt.year
    out = out.mask(y < 1970, out + pd.DateOffset(years=100))
    y = out.dt.year
    out = out.mask((y < 1970) | (y > 2014), pd.NaT)
    return out


def detect_layout(columns: list[str]) -> str:
    cols = {c.lower().replace(" ", "_") for c in columns}
    if cols & KAGGLE_HINTS:
        return "kaggle_sba_national"
    if "exposure_at_default" in cols or "default_flag" in cols:
        return "engine_tape"
    return "unknown"


def _from_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Map SBAnational.csv onto the engine schema."""
    out = pd.DataFrame(index=df.index)
    out["source"] = "kaggle_sba_national"
    out["loan_id"] = df["loannr_chkdgt"] if "loannr_chkdgt" in df.columns else np.arange(len(df))

    ead = None
    for cand in ("disbursementgross", "grappv", "grossapproval"):
        if cand in df.columns:
            ead = _num(df[cand])
            break
    if ead is None:
        raise ValueError("Kaggle tape missing GrAppv / DisbursementGross")
    out["exposure_at_default"] = ead.fillna(0).clip(lower=0)
    out["gr_appv"] = _num(df["grappv"]) if "grappv" in df.columns else out["exposure_at_default"]
    out["sba_appv"] = _num(df["sba_appv"]) if "sba_appv" in df.columns else np.nan
    out["sba_guaranteed_portion"] = (out["sba_appv"] / out["gr_appv"].replace(0, np.nan)).clip(0, 1)
    out["sba_unguaranteed_share"] = (1.0 - out["sba_guaranteed_portion"]).fillna(0.25)

    # Term is months on the Kaggle file
    if "term" in df.columns:
        out["maturity_years"] = (_num(df["term"]) / 12.0).clip(0.25, 30)
    else:
        out["maturity_years"] = 3.0

    # Li et al. teaching proxy: term >= 240 months ≈ real-estate collateral
    re_backed = out["maturity_years"] >= 20.0
    out["real_estate_backed"] = re_backed.astype(int)
    out["collateral_value"] = np.where(re_backed, out["exposure_at_default"], out["exposure_at_default"] * 0.5)
    out["ltv"] = (out["exposure_at_default"] / out["collateral_value"].replace(0, np.nan)).clip(0, 5).fillna(1.0)

    # No GAAP leverage/liquidity on this file. Named proxies:
    # leverage  := unguaranteed share (more bank skin → higher observed default in the paper)
    # liquidity := inverse size proxy from employee count (not a current ratio)
    out["leverage"] = out["sba_unguaranteed_share"].clip(0, 5)
    if "noemp" in df.columns:
        emp = _num(df["noemp"]).fillna(1).clip(lower=0)
        out["no_emp"] = emp
        out["liquidity_ratio"] = np.log1p(emp) / 4.0
    else:
        out["no_emp"] = np.nan
        out["liquidity_ratio"] = 1.0

    status = df["mis_status"].astype(str).str.upper().str.replace(" ", "") if "mis_status" in df.columns else ""
    chg_amt = _num(df["chgoffpringr"]) if "chgoffpringr" in df.columns else 0
    out["default_flag"] = (
        status.str.contains("CHGOFF") | (chg_amt.fillna(0) > 0)
    ).astype(int)
    out["loanstatus"] = df["mis_status"] if "mis_status" in df.columns else np.where(out["default_flag"] == 1, "CHGOFF", "PIF")
    out["loss_severity"] = np.where(
        out["default_flag"] == 1,
        (chg_amt / out["exposure_at_default"].replace(0, np.nan)).clip(0, 1).fillna(0.45),
        0.0,
    )

    parsed = {}
    for cand in ("disbursementdate", "approvaldate", "chgoffdate"):
        if cand in df.columns:
            parsed[cand] = _parse_date(df[cand])
    orig = parsed.get("disbursementdate")
    if orig is None:
        orig = parsed.get("approvaldate")
    elif "approvaldate" in parsed:
        orig = orig.fillna(parsed["approvaldate"])
    out["origination_date"] = orig if orig is not None else pd.NaT
    out["asof_date"] = out["origination_date"]
    out["chgoff_date"] = parsed.get("chgoffdate", pd.NaT)
    if orig is not None and "chgoffdate" in parsed:
        delta = (parsed["chgoffdate"] - orig).dt.days / 365.25
        out["years_to_default"] = np.where(out["default_flag"] == 1, delta.clip(0.08, 30), np.nan)
    else:
        out["years_to_default"] = np.nan
    out["days_past_due"] = 0
    out["origination_pd"] = np.nan

    if "newexist" in df.columns:
        out["new_business"] = (_num(df["newexist"]) == 2).astype(int)
    if "lowdoc" in df.columns:
        out["lowdoc"] = df["lowdoc"].astype(str).str.upper().eq("Y").astype(int)
    if "revlinecr" in df.columns:
        out["revolving"] = df["revlinecr"].astype(str).str.upper().eq("Y").astype(int)
    if "urbanrural" in df.columns:
        out["urban_rural"] = _num(df["urbanrural"])
    if "state" in df.columns:
        out["state"] = df["state"]
    if "naics" in df.columns:
        out["naics2"] = df["naics"].astype(str).str[:2]

    out.attrs["feature_notes"] = (
        "Kaggle SBAnational.csv has no borrower financial statements. "
        "leverage = 1 - SBA_Appv/GrAppv; liquidity_ratio = log1p(NoEmp)/4; "
        "collateral = EAD if Term>=240m else 0.5*EAD; "
        "default_flag from MIS_Status CHGOFF or ChgOffPrinGr>0; "
        "loss_severity = ChgOffPrinGr/EAD on defaults."
    )
    return out


def _from_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source"] = out.get("source", "engine_tape")
    if "exposure_at_default" not in out.columns:
        if "grossapproval" in out.columns:
            out["exposure_at_default"] = _num(out["grossapproval"])
        elif "grappv" in out.columns:
            out["exposure_at_default"] = _num(out["grappv"])
        else:
            raise ValueError("Tape needs exposure_at_default / GrAppv / grossapproval")
    out["exposure_at_default"] = _num(out["exposure_at_default"]).fillna(0).clip(lower=0)

    if "collateral_value" not in out.columns:
        out["collateral_value"] = out["exposure_at_default"] * 0.8
    out["collateral_value"] = _num(out["collateral_value"]).replace(0, np.nan).fillna(out["exposure_at_default"] * 0.8)
    out["ltv"] = (out["exposure_at_default"] / out["collateral_value"].replace(0, np.nan)).clip(0, 5).fillna(1.0)

    for col in ("current_assets", "current_liabilities", "total_debt", "total_assets"):
        if col in out.columns:
            out[col] = _num(out[col])
    if "liquidity_ratio" not in out.columns and {"current_assets", "current_liabilities"} <= set(out.columns):
        out["liquidity_ratio"] = out["current_assets"] / out["current_liabilities"].replace(0, np.nan)
    if "leverage" not in out.columns and {"total_debt", "total_assets"} <= set(out.columns):
        out["leverage"] = out["total_debt"] / out["total_assets"].replace(0, np.nan)
    if "liquidity_ratio" in out.columns:
        out["liquidity_ratio"] = _num(out["liquidity_ratio"]).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 20)
    if "leverage" in out.columns:
        out["leverage"] = _num(out["leverage"]).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0, 5)

    if "maturity_years" not in out.columns:
        out["maturity_years"] = (_num(out["term"]) / 12.0) if "term" in out.columns else 3.0
    out["maturity_years"] = _num(out["maturity_years"]).fillna(3.0).clip(0.25, 30)
    if "days_past_due" not in out.columns:
        out["days_past_due"] = 0
    out["days_past_due"] = _num(out["days_past_due"]).fillna(0).clip(lower=0)
    if "origination_pd" not in out.columns:
        out["origination_pd"] = np.nan

    if "default_flag" not in out.columns:
        blob = ""
        if "mis_status" in out.columns:
            blob = out["mis_status"].astype(str)
        elif "loanstatus" in out.columns:
            blob = out["loanstatus"].astype(str)
        out["default_flag"] = blob.str.upper().str.replace(" ", "").str.contains("CHGOFF|CHARGE|DEFAULT").astype(int)
    if "loss_severity" not in out.columns:
        out["loss_severity"] = np.where(out["default_flag"] == 1, 0.45, 0.0)
    if "asof_date" in out.columns:
        out["asof_date"] = _parse_date(out["asof_date"])
    return out


def clean_sba_data(input_path: str) -> pd.DataFrame:
    raw = pd.read_csv(input_path, low_memory=False)
    layout = detect_layout(list(raw.columns))
    raw.columns = raw.columns.str.lower().str.replace(" ", "_")
    print(f"Loan tape layout={layout} file={input_path} rows={len(raw)}")

    if layout == "kaggle_sba_national":
        # Keep resolved PIF / CHGOFF rows only — active loans have no outcome.
        if "mis_status" in raw.columns:
            keep = raw["mis_status"].astype(str).str.upper().str.replace(" ", "").isin({"PIF", "CHGOFF"})
            raw = raw.loc[keep].copy()
        df = _from_kaggle(raw)
        print(df.attrs.get("feature_notes", ""))
        return df
    if layout == "unknown":
        print("Unknown layout; attempting engine mapping.")
    return _from_engine(raw)
