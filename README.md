# macro-ecl

IFRS 9-style ECL demo. Not a production or validated model.

## Run

```bash
cd macro-ecl
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 scripts/run_all.py
PYTHONPATH=. pytest -q
```

Put Kaggle `SBAnational.csv` at `data/raw/sba_raw.csv` and FRED GDPC1+UNRATE at `data/raw/historical_macro_data.csv`.  
If those files are missing, a small sample tape is used.

```bash
python3 -m mlflow ui --backend-store-uri ./mlruns --port 5000
```

Confirm the live engine with:

```text
engine_rev=2026-09-12-overlay-sicr
```

## Outputs

- `reports/ecl_macro_SR117_Final.pdf` — short run report
- `reports/ecl_macro_SR117_Final.md`
- `reports/last_run.json`
- `reports/figures/*.png`
- `models/*.pkl`

## Method (this revision)

- Labels: Kaggle PIF vs CHGOFF (or documented sample DGP)
- PD: 12m HGB; scenario movement via config logit overlay
- LGD: hurdle × severity, downturn floor
- Origination PD: score on the loan’s own tape macros
- Stage 2: PIT / origination ≥ 1.5 or +3pp
- Stage 3: PIT PD ≥ 0.40; ECL = EAD × LGD
- Stage 1 ECL: year 1 only; Stage 2: constant-hazard lifetime ≤ 5y
- VAR on quarterly GDP growth + unemployment

See `DATA_SOURCES.md`.
