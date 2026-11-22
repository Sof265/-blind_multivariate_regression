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
    """Generates predictions for a DataFrame using the saved pipeline."""
    model = load_model(model_path)
    # The pipeline's ColumnTransformer automatically selects your TOP_5_FEATURES
    preds = model.predict(df)
    return pd.Series(np.ravel(preds), name="target_pred")


def main() -> None:
    """Loads test data from config, predicts, and saves results to CSV."""
    full_path_test = os.path.join(config.DATASET_ROOT_PATH, config.TEST_FILENAME)
    df_test = pd.read_csv(full_path_test)

    predictions = predict_from_df(df_test, model_path=config.MODEL_PATH)

    output_path = Path(config.TEST_PRED_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Generated {len(predictions)} predictions and saved to {output_path}")


if __name__ == "__main__":
    main()
