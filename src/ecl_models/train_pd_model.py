from __future__ import annotations

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from src.ecl_models.monotonic import constraints_for


def train_pd(X_train, y_train, output_path, feature_names, monotonic_map):
    cst = constraints_for(feature_names, monotonic_map)
    model = HistGradientBoostingClassifier(
        monotonic_cst=cst,
        max_iter=200,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=40,
        random_state=42,
    )
    model.fit(X_train, y_train)
    joblib.dump({"horizon_1y": model, "features": list(feature_names)}, output_path)
    return {"horizon_1y": model}
