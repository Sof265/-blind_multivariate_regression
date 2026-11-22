import numpy as np
import pandas as pd
import pytest
from src.predict import predict_from_df
from src.train import TOP_5_FEATURES, train_and_evaluate


@pytest.fixture
def synthetic_regression_data():
    """Creates a clean synthetic dataset with a known polynomial relationship."""
    rng = np.random.default_rng(42)
    n_train, n_val = 120, 30

    # Include your top 5 features plus a few extra columns to test ColumnTransformer filtering
    all_cols = TOP_5_FEATURES + ["extra_col_1", "extra_col_2"]
    X_train = pd.DataFrame(
        rng.normal(0, 1, size=(n_train, len(all_cols))), columns=all_cols
    )
    X_val = pd.DataFrame(
        rng.normal(0, 1, size=(n_val, len(all_cols))), columns=all_cols
    )

    # Create a clear signal so LassoCV selects features and R2 is high
    y_train = (
        2.5 * X_train["feature_2"]
        - 1.8 * (X_train["feature_13"] ** 2)
        + 1.2 * X_train["feature_9"] * X_train["feature_11"]
        + rng.normal(0, 0.1, size=n_train)
    ).to_numpy()

    y_val = (
        2.5 * X_val["feature_2"]
        - 1.8 * (X_val["feature_13"] ** 2)
        + 1.2 * X_val["feature_9"] * X_val["feature_11"]
        + rng.normal(0, 0.1, size=n_val)
    ).to_numpy()

    return X_train, y_train, X_val, y_val


def test_train_and_predict_pipeline(tmp_path, synthetic_regression_data):
    """Verifies end-to-end training, artifact creation, R2 quality, and DataFrame inference."""
    X_train, y_train, X_val, y_val = synthetic_regression_data

    # 1. Run training and save all artifacts to pytest's temporary directory
    _, metrics = train_and_evaluate(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=str(tmp_path),
    )

    # 2. Verify all three expected artifacts were created
    saved_model_path = tmp_path / "model.pkl"
    assert saved_model_path.exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "val_predictions.csv").exists()

    # 3. Verify regression performance threshold (R2 > 0.80 on clean synthetic signal)
    assert metrics["val_r2"] > 0.80
    assert metrics["val_rmse"] < 0.50

    # 4. Verify inference works on a new DataFrame and returns a Series of floats
    preds = predict_from_df(X_val, model_path=str(saved_model_path))
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(X_val)
    assert not preds.isna().any()


def test_missing_required_columns_raises_error(tmp_path, synthetic_regression_data):
    """Verifies ColumnTransformer raises a ValueError if required features are missing."""
    X_train, y_train, X_val, y_val = synthetic_regression_data
    train_and_evaluate(X_train, y_train, X_val, y_val, output_dir=str(tmp_path))

    saved_model_path = tmp_path / "model.pkl"

    # Drop one of the required TOP_5_FEATURES
    broken_df = X_val.drop(columns=["feature_2"])

    with pytest.raises(ValueError):
        predict_from_df(broken_df, model_path=str(saved_model_path))
