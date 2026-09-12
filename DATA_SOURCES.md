# Raw data this repo expects

## 1. Loan tape — Kaggle SBA National

Paper: Li, Mickel, Taylor (2018), *“Should This Loan be Approved or Denied?”*
File: `SBAnational.csv` (~899k rows, 27 columns, SBA 7(a) guarantees 1987–2014)

Place it at one of:

- `data/raw/SBAnational.csv`
- `data/raw/sba_raw.csv` (config default)

Columns the cleaner reads:

| Kaggle column | Engine field |
|---|---|
| `MIS_Status` (`P I F` / `CHGOFF`) | `default_flag` |
| `ChgOffPrinGr` | `loss_severity` = charge-off / EAD on defaults |
| `GrAppv` / `DisbursementGross` | `exposure_at_default` |
| `SBA_Appv` / `GrAppv` | `sba_guaranteed_portion`, `leverage` := 1 − that share |
| `Term` (months) | `maturity_years` |
| `DisbursementDate` or `ApprovalDate` | `asof_date` (joined to FRED) |
| `NoEmp` | `liquidity_ratio` proxy `log1p(NoEmp)/4` |
| `Term >= 240` | real-estate collateral proxy |

This file has **no** `current_assets`, `total_debt`, or days-past-due. Those ratios in the engine are named proxies when the Kaggle layout is detected. Rows that are not resolved `PIF`/`CHGOFF` are dropped.

Kaggle listing (one of several mirrors): search “SBAnational” or “Should This Loan be Approved or Denied”.

## 2. Macros — FRED

| FRED id | Meaning | Engine name |
|---|---|---|
| `GDPC1` | Real GDP, quarterly | `real_gdp` → `realgdp_growth` |
| `UNRATE` | Unemployment rate, monthly | `unemployment_rate` → `unemp` |

```bash
PYTHONPATH=. python -m src.data_prep.fetch_macro_data
```

writes `data/raw/historical_macro_data.csv`.

Start year is 1987 so the extract covers the SBA national window.

## 3. If those files are missing

`run_pipeline` writes a **sample** tape and a synthetic macro path and prints `used_mode=sample_fallback`. That path is only for unit tests / demo. It is not the Kaggle or FRED extract.
