from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.ecl_models.train_lgd_model import train_lgd
from src.ecl_models.train_pd_model import train_pd
from src.evaluation.metrics import classification_report_dict, fit_challenger


def run_training(config_path: str = "config/config.yaml", root: str = ".") -> dict:
    root_p = Path(root)
    with open(root_p / config_path) as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(root_p / config["paths"]["processed_tape"], low_memory=False)
    pd_features = config["model_features"]["pd_financials"] + config["model_features"]["macro_lags"]
    lgd_features = config["model_features"]["lgd_financials"] + [
        c for c in config["model_features"]["macro_lags"] if "gdp" in c.lower()
    ]
    if not lgd_features:
        lgd_features = config["model_features"]["lgd_financials"][:]

    missing = [c for c in pd_features + lgd_features if c not in df.columns]
    if missing:
        raise ValueError(f"Processed tape missing {missing}. Run the data pipeline first.")

    y = df["default_flag"].astype(int)
    if y.nunique() < 2:
        raise ValueError("default_flag has a single class")

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.3, random_state=42, stratify=y
    )
    X_pd = df[pd_features]
    X_lgd = df[lgd_features]

    os.makedirs(root_p / "models", exist_ok=True)
    pd_path = root_p / config["paths"]["pd_model"]
    lgd_path = root_p / config["paths"]["lgd_model"]

    pd_models = train_pd(
        X_pd.loc[idx_train],
        y.loc[idx_train],
        str(pd_path),
        feature_names=pd_features,
        monotonic_map=config.get("monotonic_constraints", {}),
    )
    lgd_model = train_lgd(
        X_lgd.loc[idx_train],
        y.loc[idx_train],
        df.loc[idx_train, "loss_severity"],
        str(lgd_path),
    )

    p_hold = pd_models["horizon_1y"].predict_proba(X_pd.loc[idx_test])[:, 1]
    p_ch, _ = fit_challenger(X_pd.loc[idx_train], y.loc[idx_train], X_pd.loc[idx_test])
    metrics = {
        "model": classification_report_dict(y.loc[idx_test], p_hold),
        "challenger_logit": classification_report_dict(y.loc[idx_test], p_ch),
    }
    print("\n=== HOLDOUT PD ===")
    print("HGB   ", metrics["model"])
    print("Logit ", metrics["challenger_logit"])
    hold = pd.DataFrame({"y": y.loc[idx_test].to_numpy(), "p": p_hold})
    hold["decile"] = pd.qcut(hold["p"], 10, labels=False, duplicates="drop")
    dec = hold.groupby("decile").agg(n=("y", "size"), default_rate=("y", "mean"), mean_pred=("p", "mean"))
    print("Decile calibration (1=low PD, 10=high):")
    print(dec.to_string())
    print("\nFeature means by default_flag (train):")
    print(df.loc[idx_train, pd_features + ["default_flag"]].groupby("default_flag").mean().to_string())
    label_note = "tape default_flag (Kaggle CHGOFF/PIF if layout detected; else as cleaned)"
    metrics["label_note"] = label_note

    run_id = "local_no_mlflow"
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "IFRS9_ECL_Modeling"))
        with mlflow.start_run(run_name="ecl_train") as run:
            run_id = run.info.run_id
            mlflow.log_params({
                "pd_features": ",".join(pd_features),
                "lgd_features": ",".join(lgd_features),
                "label": label_note,
            })
            for prefix in ("model", "challenger_logit"):
                block = metrics.get(prefix) or {}
                if not isinstance(block, dict):
                    continue
                for k, v in block.items():
                    try:
                        mlflow.log_metric(f"{prefix}_{k}", float(v))
                    except (TypeError, ValueError):
                        pass
            import warnings
            ex_pd = X_pd.loc[idx_test].head(5).astype("float64")
            ex_lgd = X_lgd.loc[idx_test].head(5).astype("float64")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Hint: Inferred schema")
                mlflow.sklearn.log_model(pd_models["horizon_1y"], name="pd_model_horizon_1y", input_example=ex_pd)
                mlflow.sklearn.log_model(lgd_model["classifier"], name="lgd_classifier", input_example=ex_lgd)
                mlflow.sklearn.log_model(lgd_model["regressor"], name="lgd_regressor", input_example=ex_lgd)
            try:
                import shap

                sample = X_pd.loc[idx_train].sample(n=min(300, len(idx_train)), random_state=42)
                explainer = shap.Explainer(pd_models["horizon_1y"], sample)
                with tempfile.TemporaryDirectory() as td:
                    path = os.path.join(td, "shap_explainer.pkl")
                    joblib.dump(explainer, path)
                    mlflow.log_artifact(path)
            except Exception as e:
                print(f"SHAP artifact skipped: {e}")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

    metrics["run_id"] = run_id
    joblib.dump(metrics, root_p / "models" / "holdout_metrics.pkl")
    return metrics


if __name__ == "__main__":
    run_training()
