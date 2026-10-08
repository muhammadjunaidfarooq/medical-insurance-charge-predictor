"""Load the trained pipeline and make predictions.

Shared by the web app (main.py) and the command-line tool (predict.py).
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "insurance_model.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Train the model first: python train.py")


@lru_cache(maxsize=1)
def load_model_info() -> dict:
    _require(INFO_PATH)
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model():
    _require(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def validate_inputs(age, sex, bmi, children, smoker, region) -> list[str]:
    """Return a list of problems (empty list = inputs are valid)."""
    info = load_model_info()
    ranges, categories = info["input_ranges"], info["categories"]
    errors = []
    for name, value in (("age", age), ("bmi", bmi), ("children", children)):
        low, high = ranges[name]
        if not low <= value <= high:
            errors.append(f"{name} must be between {low} and {high} (range of the training data)")
    for name, value in (("sex", sex), ("smoker", smoker), ("region", region)):
        if value not in categories[name]:
            errors.append(f"{name} must be one of: {', '.join(categories[name])}")
    return errors


def predict_charge(age, sex, bmi, children, smoker, region) -> float:
    """Predict the yearly insurance charge in USD for one person."""
    errors = validate_inputs(age, sex, bmi, children, smoker, region)
    if errors:
        raise ValueError("; ".join(errors))

    row = pd.DataFrame(
        [{"age": age, "sex": sex, "bmi": bmi, "children": children,
          "smoker": smoker, "region": region}],
        columns=FEATURES,
    )
    return float(load_model().predict(row)[0])
