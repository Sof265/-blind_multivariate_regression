from pathlib import Path

# 1. Resolve project root reliably regardless of working directory or symlinks
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Keep paths as Path objects
DATASET_ROOT_PATH = PROJECT_ROOT / "data"
MODEL_ROOT_PATH = PROJECT_ROOT / "models"

DATASET_ROOT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_ROOT_PATH.mkdir(parents=True, exist_ok=True)

# 3. Full file paths ready to use everywhere
DATASET_FILENAME = "training_data.csv"
TEST_FILENAME = "blind_test_data.csv"

TRAIN_DATA_PATH = DATASET_ROOT_PATH / DATASET_FILENAME
TEST_DATA_PATH = DATASET_ROOT_PATH / TEST_FILENAME
MODEL_PATH = MODEL_ROOT_PATH / "regressor.pkl"

TRAIN_PREDICTIONS_FILENAME = "train_predictions.csv"
VAL_PREDICTIONS_FILENAME = "val_predictions.csv"
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"

TRAIN_PRED_PATH = MODEL_ROOT_PATH / TRAIN_PREDICTIONS_FILENAME
VAL_PRED_PATH = MODEL_ROOT_PATH / VAL_PREDICTIONS_FILENAME
TEST_PRED_PATH = MODEL_ROOT_PATH / TEST_PREDICTIONS_FILENAME

# 4. Shared modeling constants (prevents mismatch between train.py, predict.py, and notebooks)
TARGET_COLUMN = "target"
TOP_FEATURES = ["feature_2", "feature_9", "feature_11", "feature_13", "feature_18"]
RANDOM_STATE = 42
