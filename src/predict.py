import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from src import config


def load_model(model_path: Path | str = config.MODEL_PATH):
    """Loads the trained pipeline from a pickle file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python -m src.train' first."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def predict_from_df(
    df: pd.DataFrame, model_path: Path | str = config.MODEL_PATH
) -> pd.Series:
    """Generates predictions for a DataFrame as a Series, raising an error on nulls."""
    cols_to_check = getattr(config, "TOP_5_FEATURES", df.columns)
    null_counts = df[cols_to_check].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()

    if cols_with_nulls:
        raise ValueError(
            f"Invalid data: input contains null values in {cols_with_nulls}. "
            "Please provide complete data without nulls."
        )

    model = load_model(model_path)
    preds = model.predict(df)

    return pd.Series(np.ravel(preds), name="target_pred")


def main() -> None:
    """Loads test data from config, predicts, and saves results to CSV."""
    full_path_test = os.path.join(config.DATASET_ROOT_PATH, config.TEST_FILENAME)
    df_test = pd.read_csv(full_path_test)

    try:
        predictions = predict_from_df(df_test, model_path=config.MODEL_PATH)
    except ValueError as e:
        print(e)
        return

    output_path = Path(config.TEST_PRED_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Generated {len(predictions)} predictions and saved to {output_path}")


if __name__ == "__main__":
    main()
