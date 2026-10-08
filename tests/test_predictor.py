import pytest

from predictor import load_model_info, predict_charge, validate_inputs


def test_prediction_is_positive_for_young_healthy_profile():
    # The original linear model returned a negative charge for this profile.
    charge = predict_charge(18, "female", 20.0, 0, "no", "southeast")
    assert charge > 0


def test_smoker_predicted_higher_than_same_non_smoker():
    base = dict(age=45, sex="male", bmi=30.0, children=2, region="southeast")
    assert predict_charge(smoker="yes", **base) > predict_charge(smoker="no", **base)


def test_values_outside_training_range_are_rejected():
    max_age = load_model_info()["input_ranges"]["age"][1]
    with pytest.raises(ValueError):
        predict_charge(max_age + 1, "male", 25.0, 0, "no", "northeast")


def test_unknown_category_is_reported():
    errors = validate_inputs(30, "male", 25.0, 0, "no", "mars")
    assert any("region" in e for e in errors)
