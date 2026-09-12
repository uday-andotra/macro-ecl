from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression, Ridge


def train_lgd(X_train, y_loss_indicator, y_severity, output_path):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_loss_indicator)

    mask = y_loss_indicator.astype(int) == 1
    reg = Ridge(alpha=1.0)
    if mask.sum() >= 5:
        reg.fit(X_train.loc[mask], y_severity.loc[mask])
    else:
        reg.fit(X_train, y_severity)

    bundle = {"classifier": clf, "regressor": reg}
    joblib.dump(bundle, output_path)
    return bundle
