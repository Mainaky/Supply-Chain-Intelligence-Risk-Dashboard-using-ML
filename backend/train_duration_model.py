"""Train an order-level shipping-duration regressor from the packaged dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "cleaned_data.csv"
MODEL_PATH = ROOT / "models" / "ship_days_regressor.pkl"
METRICS_PATH = ROOT / "models" / "ship_days_regressor_metrics.json"
TARGET = "Days for shipping (real)"
GROUP = "Order Id"
CATEGORICAL = ["Shipping Mode", "Order Region", "Category Name"]
NUMERIC = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate",
]
FEATURES = CATEGORICAL + NUMERIC


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET, GROUP])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing fields: {sorted(missing)}")

    data = data.dropna(subset=FEATURES + [TARGET, GROUP]).copy()
    X = data[FEATURES]
    y = data[TARGET].astype(float)
    groups = data[GROUP]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    preprocessing = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=5,
                    max_features=0.85,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    train_orders = int(groups.iloc[train_indices].nunique())
    test_orders = int(groups.iloc[test_indices].nunique())
    if set(groups.iloc[train_indices]).intersection(groups.iloc[test_indices]):
        raise RuntimeError("Grouped holdout unexpectedly contains overlapping order IDs")

    model = {
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET,
    }
    joblib.dump(model, MODEL_PATH, compress=3)
    metrics = {
        "target": TARGET,
        "model": "RandomForestRegressor",
        "split": "80/20 grouped by Order Id; random_state=42",
        "rows_used": int(len(data)),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_orders": train_orders,
        "test_orders": test_orders,
        "mae_days": mae,
        "rmse_days": rmse,
        "r2": r2,
        "feature_columns": FEATURES,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved regressor: {MODEL_PATH}")


if __name__ == "__main__":
    main()
