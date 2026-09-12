from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def classification_report_dict(y_true, p_hat) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_hat).astype(float), 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "default_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
    }
    if y.min() != y.max():
        out["auc"] = float(roc_auc_score(y, p))
    else:
        out["auc"] = float("nan")
    # calibration slope from logit(p) ~ y via linear probability on p
    var = np.var(p)
    out["calibration_slope"] = float(np.cov(p, y)[0, 1] / var) if var > 0 else float("nan")
    return out


def fit_challenger(X_train, y_train, X_test):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1], pipe
