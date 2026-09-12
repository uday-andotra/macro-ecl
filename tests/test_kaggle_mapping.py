from pathlib import Path

import pandas as pd

from src.data_prep.clean_loan_tape import clean_sba_data, detect_layout


def test_detect_kaggle():
    assert detect_layout(["LoanNr_ChkDgt", "MIS_Status", "GrAppv"]) == "kaggle_sba_national"


def test_kaggle_maps_mis_status(tmp_path: Path):
    p = tmp_path / "SBAnational.csv"
    pd.DataFrame({
        "LoanNr_ChkDgt": [1, 2],
        "MIS_Status": ["P I F", "CHGOFF"],
        "GrAppv": ["$100,000.00", "$50,000.00"],
        "SBA_Appv": ["$75,000.00", "$25,000.00"],
        "DisbursementGross": ["$100,000.00", "$50,000.00"],
        "ChgOffPrinGr": ["$0.00", "$20,000.00"],
        "Term": [84, 240],
        "NoEmp": [5, 20],
        "DisbursementDate": ["10-Jun-08", "15-Jan-09"],
        "NewExist": [1, 2],
        "LowDoc": ["N", "Y"],
        "RevLineCr": ["N", "N"],
    }).to_csv(p, index=False)
    df = clean_sba_data(str(p))
    assert list(df["default_flag"]) == [0, 1]
    assert abs(df.loc[1, "loss_severity"] - 0.4) < 1e-6
    assert df.loc[1, "real_estate_backed"] == 1
    assert df.loc[0, "leverage"] == 0.25
    assert pd.api.types.is_datetime64_any_dtype(df["asof_date"])
