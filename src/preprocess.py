from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


def get_feature_target(
    app_train: pd.DataFrame,
    target_column: str = "target",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Arguments:
        app_train : pd.DataFrame
            Training datasets

    Returns:
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
    """

    X_train = app_train.drop(columns=target_column)
    y_train = app_train[target_column]

    return X_train, y_train


def get_train_val_sets(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split training dataset in two new sets used for train and validation.

    Arguments:
        X_train : pd.DataFrame
            Original training features
        y_train: pd.Series
            Original training labels/target

    Returns:
        X_train : pd.DataFrame
            Training features
        X_val : pd.DataFrame
            Validation features
        y_train : pd.Series
            Training target
        y_val : pd.Series
            Validation target
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42
    )

    return X_train, X_val, y_train, y_val


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, add_f18_quad=True):
        self.add_f18_quad = add_f18_quad

    def fit(self, X, y=None):
        self.f18_mean_ = X["feature_18"].mean()
        return self

    def transform(self, X):
        X_out = X.copy()
        f13_clipped = np.maximum(X_out["feature_13"], 0)
        X_out["f13_sqrt"] = np.sqrt(f13_clipped)
        X_out["f13_log"] = np.log1p(f13_clipped)

        if self.add_f18_quad:
            X_out["f18_quad"] = (X_out["feature_18"] - self.f18_mean_) ** 2
        return X_out


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # std_scaler = StandardScaler()
    # train = std_scaler.fit_transform(working_train_df)
    # val = std_scaler.transform(working_val_df)
    # test = std_scaler.transform(working_test_df)

    return working_train_df, working_val_df, working_test_df
