# Sparse Polynomial Regression Model

A leak-free `scikit-learn` regression pipeline designed for small tabular datasets (800 training samples, 20 anonymized features). It isolates the top 5 predictive features and applies degree-3 polynomial expansion paired with L1/L2 regularization to capture non-linear interactions without overfitting.

## Project Structure

```text
├── data/               # Raw datasets (train and test CSVs)
├── model/              # Saved model artifact (regressor.pkl), metrics, and predictions
├── notebooks/          # Exploratory analysis and model experimentation
│   ├── EDA.py          # Exploratory data analysis (EDA) with feature selection diagnostics
│   ├── Model.py        # Candidate model benchmarking and model evaluation
├── src/                # Source code for data preparation, training, and inference
│   ├── config.py       # File paths and dataset constants
│   ├── datautils.py    # Dataset shape validations and visualization helpers
│   ├── preprocess.py   # Feature/target splitting and train/validation splits
│   ├── train.py        # Pipeline training, CV evaluation, and artifact export
│   └── predict.py      # Batch and DataFrame inference using regressor.pkl
├── tests/              # Unit tests for data validation, training, and inference
├── README.md
└── requirements.txt    # Python dependencies
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Model Architecture & Overview

Rather than fitting all 20 features—which introduces noise in high-degree expansions—the model isolates the **top 5 predictive features** (`feature_2`, `feature_9`, `feature_11`, `feature_13`, and `feature_18`) identified during exploratory mutual information and permutation importance diagnostics.

- **Pipeline Steps:**
  1. **`column_filter` (`ColumnTransformer`):** Selects the top 5 features directly from raw input DataFrames and drops remaining columns.
  2. **`scaler` (`StandardScaler`):** Standardizes features to zero mean and unit variance inside each fold.
  3. **`poly` (`PolynomialFeatures`):** Generates non-linear interactions and polynomial terms up to `degree=3` (55 candidate terms).
  4. **`selector` (`SelectFromModel` + `LassoCV`):** Applies L1 regularization (`threshold=0.05`) to prune noisy polynomial terms and retain a sparse, high-signal subset.
  5. **`model` (`RidgeCV`):** Refits the selected sparse polynomial features using L2 regularization across 50 log-spaced \(\alpha\) values (\(10^{-3}\) to \(10^{3}\)).
- **Input:** `pandas.DataFrame` containing `feature_2`, `feature_9`, `feature_11`, `feature_13`, and `feature_18`.
- **Output:** Continuous target predictions (`pandas.Series` of floats).

## Performance & Evaluation

Evaluated using 5-fold shuffled cross-validation (`random_state=42`) on the 640-row training split and tested on a held-out 160-row validation split:

| Metric   |   Train (5-Fold CV)   | Validation | Gap (Train - Val) |
| :------- | :-------------------: | :--------: | :---------------: |
| **R²**   | 0.90603 (+/- 0.02385) |  0.89690   |     +0.00913      |
| **RMSE** | 1.56078 (+/- 0.16090) |  1.52140   |     +0.03938      |

## Usage

### Training from Scratch

Runs dimension checks, 5-fold cross-validation, and validation evaluation, saving `regressor.pkl`, `metrics.json`, `train_predictions.csv`, and `val_predictions.csv` to `model/`:

```bash
python -m src.train
```

### Batch Inference (CLI)

Loads the trained `regressor.pkl` artifact and saves test predictions to `target_pred.csv`:

```bash
python -m src.predict
```

### Inference in Python (Quickstart)

Use `predict_from_df()` to run model inference on a pandas `DataFrame`. Before predicting, the function automatically checks required feature columns for null values (`NaN`/`None`)—returning a `pd.Series` of predictions if the data is complete, or raising a `ValueError` with details on any missing values.

```python
import pandas as pd
from src import config
from src.predict import predict_from_df

sample_df = pd.DataFrame([{
    "feature_2": 0.5,
    "feature_9": -1.2,
    "feature_11": 0.8,
    "feature_13": 1.4,
    "feature_18": -0.3,
}])

try:
    predictions = predict_from_df(sample_df, model_path=config.MODEL_PATH)
    print(f"Prediction: {predictions.iloc[0]:.4f}")
except ValueError as e:
    print(e)
```

### Running Unit Tests

```bash
python -m pytest -v
```

## Limitations & What's Next

- **No Missing-Value Imputation:** No imputation transformations were applied at this stage, as appropriate imputation strategies depend heavily on the domain context of each variable. Currently, inference halts and raises a `ValueError` when nulls are present.
- **Small Sample Size & Extrapolation:** Trained on 800 samples; degree-3 polynomial terms can diverge sharply on out-of-distribution inputs outside the training feature range.
- **Anonymized Features:** Feature definitions were anonymized, so domain-specific feature ratios (e.g., $x_a / x_b$) were intentionally excluded to avoid spurious correlations.
- **Domain Error Tolerance:** Production readiness depends on the target application's acceptable error margin relative to the $\approx 1.52$ validation RMSE.
- **Future Improvements:** Once the domain context of each feature is better understood, implement context-aware imputation (e.g., mean, median, or constant value) within the preprocessing pipeline. Additionally, benchmark `ElasticNetCV` as a single-stage alternative to `LassoCV` + `RidgeCV`, modularize evaluation plots, and add staging validation before deployment.
