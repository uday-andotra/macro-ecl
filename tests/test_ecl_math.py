import pandas as pd

from src.ecl_models.ecl_execution_engine import ECLExecutionEngine


class _Dummy(ECLExecutionEngine):
    def __init__(self):
        self.ecl_cfg = {
            "discount_rate": 0.0,
            "max_horizon_years": 2,
            "hazard_growth": 0.0,
            "hazard_floor": 1e-6,
            "hazard_cap": 0.99,
            "downturn_lgd_floor": 0.0,
            "sicr_pd_ratio": 2.0,
            "stage2_dpd": 30,
            "stage3_dpd": 90,
        }


def test_stage1_uses_year1_only():
    eng = _Dummy()
    df = pd.DataFrame({
        "PIT_PD_12M": [0.10],
        "maturity_years": [2.0],
        "Downturn_LGD": [0.50],
        "exposure_at_default": [100.0],
        "ifrs9_stage": [1],
    })
    out = eng._term_ecl(df)
    assert abs(out["ECL_12M"].iloc[0] - 5.0) < 1e-6
    assert out["ECL"].iloc[0] == out["ECL_12M"].iloc[0]
    assert out["ECL_lifetime"].iloc[0] > out["ECL_12M"].iloc[0]


def test_stage2_uses_lifetime():
    eng = _Dummy()
    df = pd.DataFrame({
        "PIT_PD_12M": [0.10],
        "maturity_years": [2.0],
        "Downturn_LGD": [0.50],
        "exposure_at_default": [100.0],
        "ifrs9_stage": [2],
    })
    out = eng._term_ecl(df)
    assert out["ECL"].iloc[0] == out["ECL_lifetime"].iloc[0]
