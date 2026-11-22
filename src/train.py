import json
import os
from pathlib import Path

# import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pickle

# Import your local modules
from src import config, datautils, preprocess

TOP_5_FEATURES = ["feature_2", "feature_13", "feature_9", "feature_11", "feature_18"]


def build_pipeline(features: list[str] = TOP_5_FEATURES) -> Pipeline:
    """Builds the leak-free sparse polynomial regression pipeline."""
    feature_selector = ColumnTransformer(
        transformers=[("select_top_5", "passthrough", features)],
        remainder="drop",
    )

    return Pipeline(
        [
            ("column_filter", feature_selector),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            (
                "selector",
                SelectFromModel(
                    estimator=LassoCV(cv=5, max_iter=20000, random_state=42),
                    threshold=0.05,
                ),
            ),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 50))),
        ]
    )


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict]:
    """Trains the pipeline, evaluates CV vs. validation metrics, and saves artifacts."""
    pipe = build_pipeline(TOP_5_FEATURES)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Cross-validation
    cv_results = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
    )
    cv_r2 = cv_results["test_r2"]
    cv_rmse = -cv_results["test_rmse"]

    # 2. Fit on full training set & evaluate on validation set
    pipe.fit(X_train, y_train)
    y_val_pred = pipe.predict(X_val)
    val_r2 = float(r2_score(y_val, y_val_pred))
    val_rmse = float(root_mean_squared_error(y_val, y_val_pred))

    # 3. Extract selected polynomial terms and weights
    all_names = pipe.named_steps["poly"].get_feature_names_out(TOP_5_FEATURES)
    selected_mask = pipe.named_steps["selector"].get_support()
    selected_names = all_names[selected_mask]
    coefs = pipe.named_steps["model"].coef_

    selected_features = {
        name: round(float(weight), 6)
        for name, weight in sorted(
            zip(selected_names, coefs), key=lambda x: abs(x[1]), reverse=True
        )
    }

    # 4. Build metrics summary divided into 'train' and 'val'
    metrics = {
        "train": {
            "r2_mean": round(float(cv_r2.mean()), 5),
            "r2_std": round(float(cv_r2.std()), 5),
            "rmse_mean": round(float(cv_rmse.mean()), 5),
            "rmse_std": round(float(cv_rmse.std()), 5),
        },
        "val": {
            "r2": round(val_r2, 5),
            "rmse": round(val_rmse, 5),
        },
        "gap": {
            "r2": round(float(cv_r2.mean() - val_r2), 5),
            "rmse": round(float(cv_rmse.mean() - val_rmse), 5),
        },
        "selected_terms_count": f"{len(selected_names)} / {len(all_names)}",
        "selected_coefficients": selected_features,
    }

    # Optional: Print a clean side-by-side table in the console
    summary_table = pd.DataFrame(
        {
            "Metric": ["R2", "RMSE"],
            "Train": [
                f"{metrics['train']['r2_mean']:.5f} (+/- {metrics['train']['r2_std']:.5f})",
                f"{metrics['train']['rmse_mean']:.5f} (+/- {metrics['train']['rmse_std']:.5f})",
            ],
            "Val": [
                f"{metrics['val']['r2']:.5f}",
                f"{metrics['val']['rmse']:.5f}",
            ],
            "Gap": [
                f"{metrics['gap']['r2']:+.5f}",
                f"{metrics['gap']['rmse']:+.5f}",
            ],
        }
    )
    print("\n", summary_table.to_string(index=False), "\n")

    # 5. Save model and metrics
    out_path = Path(config.MODEL_ROOT_PATH)
    out_path.mkdir(parents=True, exist_ok=True)

    # joblib.dump(pipe, out_path / "sparse_poly_model.joblib")
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    print(f"Saved model and metrics to {out_path}/")

    return pipe, metrics


def main() -> None:
    """Loads data, validates dimensions, trains the model, and generates test predictions."""
    # 1. Load datasets
    full_path_train = os.path.join(config.DATASET_ROOT_PATH, config.DATASET_FILENAME)
    full_path_test = os.path.join(config.DATASET_ROOT_PATH, config.TEST_FILENAME)

    df_train = pd.read_csv(full_path_train)
    df_test = pd.read_csv(full_path_test)

    # 2. Extract features/target and verify full dataset dimensions
    X, y = preprocess.get_feature_target(df_train, target_column=config.TARGET_COLUMN)
    X_test = df_test
    datautils.check_dim_datasets(X, X_test, (800, 200, 20))

    # 3. Split train/val sets and verify split dimensions
    X_train, X_val, y_train, y_val = preprocess.get_train_val_sets(X, y)
    datautils.check_dim_datasets(X_train, X_val, (640, 160, 20))

    # 4. Train, evaluate, and save artifacts
    model, _ = train_and_evaluate(X_train, y_train, X_val, y_val)

    # 5. Optional: Generate test set predictions immediately
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    out_path = Path(config.MODEL_ROOT_PATH)
    out_path.mkdir(parents=True, exist_ok=True)

    y_actual_arr = np.ravel(y_train)
    y_pred_arr = np.ravel(y_train_pred)

    train_results = pd.DataFrame(
        {
            "y_actual": y_actual_arr,
            "y_pred": y_pred_arr,
            "residual": y_actual_arr - y_pred_arr,
        }
    )

    y_actual_arr = np.ravel(y_val)
    y_pred_arr = np.ravel(y_val_pred)

    val_results = pd.DataFrame(
        {
            "y_actual": y_actual_arr,
            "y_pred": y_pred_arr,
            "residual": y_actual_arr - y_pred_arr,
        }
    )
    train_results.to_csv(out_path / config.TRAIN_PREDICTIONS_FILENAME, index=False)
    val_results.to_csv(out_path / config.VAL_PREDICTIONS_FILENAME, index=False)

    print(
        f"Saved {config.TRAIN_PREDICTIONS_FILENAME} and {config.VAL_PREDICTIONS_FILENAME} to {out_path}/"
    )


if __name__ == "__main__":
    main()
