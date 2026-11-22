from src import config
import pandas as pd
import numpy as np


def check_dim_datasets(
    app_train: pd.DataFrame, app_test: pd.DataFrame, expected_dims: tuple[int, int, int]
):
    """
    Receives train and test dataframes
    and checks if they have the correct shape.

    Arguments:
        train_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        Message indicating if the datasets have the correct shape and type.
    """
    train_rows, test_rows, cols = expected_dims

    if config.TARGET_COLUMN in app_train.columns:
        print(f"Target column '{config.TARGET_COLUMN}' found in training dataset")
        df_train_feat = app_train.drop(columns=[config.TARGET_COLUMN])
    else:
        df_train_feat = app_train

    if df_train_feat.shape == (train_rows, cols):
        print("Success: Train shape is correct!")
    else:
        raise ValueError("Train dataset shape is incorrect, please review your code")

    if app_test.shape == (test_rows, cols):
        print("Success: Test shape is correct!")
    else:
        raise ValueError("Test dataset shape is incorrect, please review your code")


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Computes min, percentiles, max, skewness, kurtosis, and variation coefficient for numeric columns."""
    num_df = df.select_dtypes(include="number")

    return pd.DataFrame(
        {
            "Minimum": num_df.min(),
            "Percentil 10": num_df.quantile(0.10).round(2),
            "Percentil 25": num_df.quantile(0.25).round(2),
            "Percentil 50": num_df.quantile(0.50).round(2),
            "Percentil 75": num_df.quantile(0.75).round(2),
            "Percentil 90": num_df.quantile(0.90).round(2),
            "Maximum": num_df.max(),
            "Skewness": num_df.skew().round(2),
            "Kurtosis": num_df.kurtosis().round(2),
            "Variation Coefficient": (num_df.std() / num_df.mean().replace(0, np.nan))
            .abs()
            .round(2),
        }
    )
