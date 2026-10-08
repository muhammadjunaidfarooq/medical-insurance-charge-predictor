from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

VALID = {"age": 45, "sex": "male", "bmi": 30.0, "children": 2, "smoker": "yes", "region": "southeast"}


def test_home_page_loads():
    response = client.get("/")
    assert response.status_code == 200
    assert "Insurance Charge Predictor" in response.text


def test_model_info_endpoint():
    data = client.get("/model-info").json()
    assert {"model", "input_ranges", "reference_averages_usd"} <= data.keys()


def test_predict_valid_input():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert body["currency"] == "USD"
    assert body["predicted_charge"] > 0


def test_predict_rejects_age_outside_training_range():
    response = client.post("/predict", json={**VALID, "age": 100})
    assert response.status_code == 422


def test_predict_rejects_unknown_region():
    response = client.post("/predict", json={**VALID, "region": "mars"})
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    payload = {k: v for k, v in VALID.items() if k != "bmi"}
    assert client.post("/predict", json=payload).status_code == 422
