"""FastAPI web app: serves the form page and a JSON prediction endpoint.

Run locally:
    uvicorn main:app --reload
"""

from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predictor import load_model, load_model_info, predict_charge

STATIC_DIR = Path(__file__).resolve().parent / "static"

INFO = load_model_info()
load_model()  # load once at startup so the first request is fast and errors show early
RANGES = INFO["input_ranges"]

app = FastAPI(title="Insurance Charge Predictor")


class PredictRequest(BaseModel):
    # Limits come from the training data, so the model is never asked to extrapolate.
    age: int = Field(ge=RANGES["age"][0], le=RANGES["age"][1])
    sex: Literal["female", "male"]
    bmi: float = Field(ge=RANGES["bmi"][0], le=RANGES["bmi"][1])
    children: int = Field(ge=RANGES["children"][0], le=RANGES["children"][1])
    smoker: Literal["no", "yes"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/model-info")
def model_info():
    """Facts the page needs: input limits, reference averages and model details."""
    selected = INFO["selected_model"]
    return {
        "model": selected,
        "training_rows": INFO["dataset"]["rows_after_cleaning"],
        "input_ranges": RANGES,
        "reference_averages_usd": INFO["reference_averages_usd"],
        "test_metrics": INFO["results"][selected]["test"],
        "permutation_importance": INFO["permutation_importance"],
    }


@app.post("/predict")
def predict(data: PredictRequest):
    charge = predict_charge(**data.model_dump())
    return {"predicted_charge": round(charge, 2), "currency": "USD"}
