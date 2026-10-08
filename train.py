"""Train, compare and save the insurance charge model.

Run from the project folder:
    python train.py

Steps:
1. Load data/insurance.csv and remove exact duplicate rows.
2. Split into 80% train / 20% test (fixed random_state, so results repeat).
3. Compare a few models with 5-fold cross-validation on the TRAINING set only.
4. Pick the model with the lowest cross-validated MAE.
5. Evaluate every model once on the held-out test set (reported, not used for selection).
6. Compute permutation importance for the selected model on the test set.
7. Save the full pipeline (preprocessing + model) and a JSON summary.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "insurance.csv"
MODEL_PATH = ROOT / "models" / "insurance_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, int]:
    """Load the CSV, check its columns, and drop exact duplicate rows."""
    df = pd.read_csv(path)

    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_cols)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing values; clean them before training.")

    n_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_duplicates


def build_pipeline(model) -> Pipeline:
    """One-hot encode the categorical columns; numeric columns pass through.

    Scaling is not needed: tree models ignore feature scale, and plain linear
    regression gives the same predictions with or without scaling.
    The encoder is fitted inside the pipeline, so it only ever learns from
    the training data it is fitted on (no leakage from the test set).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def candidate_models() -> dict:
    """A small, sensible set: a linear baseline and two tree ensembles.

    Hyperparameters are reasonable fixed values, not tuned.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


def regression_metrics(y_true, y_pred) -> dict:
    # Rounded so repeated runs write an identical file (parallel sums can
    # differ in the 12th decimal place).
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main() -> None:
    df, n_duplicates = load_data()
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Rows: {len(df)} (removed {n_duplicates} duplicate) | "
          f"train: {len(X_train)} | test: {len(X_test)}\n")

    # --- 1. Compare models with cross-validation on the training set only ---
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    results = {}
    for name, model in candidate_models().items():
        scores = cross_validate(build_pipeline(model), X_train, y_train, cv=cv, scoring=scoring)
        results[name] = {
            "cv": {
                "mae": round(float(-scores["test_mae"].mean()), 2),
                "rmse": round(float(-scores["test_rmse"].mean()), 2),
                "r2": round(float(scores["test_r2"].mean()), 4),
            }
        }

    selected = min(results, key=lambda name: results[name]["cv"]["mae"])

    # --- 2. Fit every model on the full training set, evaluate once on test ---
    fitted = {}
    for name, model in candidate_models().items():
        pipeline = build_pipeline(model).fit(X_train, y_train)
        fitted[name] = pipeline
        results[name]["test"] = regression_metrics(y_test, pipeline.predict(X_test))

    print(f"{'Model':<20}{'CV MAE':>10}{'CV RMSE':>10}{'CV R2':>8}"
          f"{'Test MAE':>11}{'Test RMSE':>11}{'Test R2':>9}")
    for name, r in results.items():
        mark = "  <- selected" if name == selected else ""
        print(f"{name:<20}{r['cv']['mae']:>10,.0f}{r['cv']['rmse']:>10,.0f}{r['cv']['r2']:>8.3f}"
              f"{r['test']['mae']:>11,.0f}{r['test']['rmse']:>11,.0f}{r['test']['r2']:>9.3f}{mark}")

    final_pipeline = fitted[selected]

    # --- 3. Permutation importance (how much test MAE rises when a feature is shuffled) ---
    perm = permutation_importance(
        final_pipeline, X_test, y_test,
        scoring="neg_mean_absolute_error", n_repeats=10, random_state=RANDOM_STATE,
    )
    importance = sorted(
        (
            {"feature": f, "mae_increase": round(float(m), 2), "std": round(float(s), 2)}
            for f, m, s in zip(FEATURES, perm.importances_mean, perm.importances_std)
        ),
        key=lambda item: item["mae_increase"],
        reverse=True,
    )
    print(f"\nPermutation importance ({selected}, increase in test MAE in USD):")
    for item in importance:
        print(f"  {item['feature']:<10}{item['mae_increase']:>10,.0f}")

    # --- 4. Save the pipeline and a summary used by the app and README ---
    info = {
        "selected_model": selected,
        "selection_rule": "lowest mean 5-fold cross-validated MAE on the training set",
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "dataset": {
            "file": "data/insurance.csv",
            "rows_after_cleaning": int(len(df)),
            "duplicates_removed": n_duplicates,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "features": FEATURES,
        "target": TARGET,
        "categories": {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES},
        # Inputs outside the training data's range are refused: the model has
        # never seen them, so its output there would be guesswork.
        "input_ranges": {
            "age": [int(X_train["age"].min()), int(X_train["age"].max())],
            "bmi": [math.floor(X_train["bmi"].min() * 10) / 10,
                    math.ceil(X_train["bmi"].max() * 10) / 10],
            "children": [int(X_train["children"].min()), int(X_train["children"].max())],
        },
        "reference_averages_usd": {
            "Dataset average": round(float(y.mean()), 2),
            "Non-smoker average": round(float(df.loc[df["smoker"] == "no", TARGET].mean()), 2),
            "Smoker average": round(float(df.loc[df["smoker"] == "yes", TARGET].mean()), 2),
        },
        "results": results,
        "permutation_importance": importance,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved model ({selected}) to {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved summary to {INFO_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
