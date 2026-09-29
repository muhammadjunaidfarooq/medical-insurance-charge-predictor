import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Insurance Charge Predictor")

# No CORS middleware needed: the form below is served by this same app, so the
# browser's fetch("/predict") call is same-origin and never triggers a CORS check.

# Loaded once at startup instead of on every request.
model = joblib.load("insurance_model.pkl")
with open("feature_order.json") as f:
    feature_order = json.load(f)


FORM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Insurance Charge Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --navy: #0b2942;
      --navy-light: #123a5c;
      --teal: #0f5c66;
      --teal-hover: #0c4a52;
      --amber: #e8965a;
      --amber-dark: #d97f3d;
      --bg: #eef2f4;
      --panel-bg: #ffffff;
      --text: #1e2b33;
      --muted: #64748a;
      --border: #d7dee3;
      --warn-bg: #fdf1e7;
      --warn-border: #e8965a;
      --warn-text: #8a4a1c;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      padding: 32px 20px;
      background: var(--bg);
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .heading {
      text-align: center;
      margin-bottom: 28px;
    }

    h1 {
      font-family: "Poppins", "Inter", sans-serif;
      font-weight: 700;
      font-size: 1.9em;
      margin: 0 0 6px;
      color: var(--navy);
      letter-spacing: -0.01em;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 0.95em;
      max-width: 480px;
    }

    .layout {
      width: 100%;
      max-width: 900px;
      display: flex;
      gap: 24px;
      align-items: stretch;
    }

    @media (max-width: 720px) {
      .layout { flex-direction: column; }
    }

    .panel {
      background: var(--panel-bg);
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(11, 41, 66, 0.08), 0 2px 8px rgba(11, 41, 66, 0.04);
      padding: 30px 28px;
    }

    .form-panel { flex: 1.1; }
    .result-panel { flex: 1; display: flex; flex-direction: column; }

    .section-heading {
      font-family: "Poppins", "Inter", sans-serif;
      font-weight: 600;
      font-size: 0.82em;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--teal);
      margin: 0 0 14px;
      padding-bottom: 8px;
      border-bottom: 2px solid var(--border);
    }

    .section + .section { margin-top: 24px; }

    .field { margin-bottom: 16px; }

    label {
      display: block;
      margin-bottom: 6px;
      font-weight: 600;
      font-size: 0.88em;
      color: var(--navy);
    }

    .hint {
      margin-top: 4px;
      font-size: 0.76em;
      color: var(--muted);
    }

    input, select {
      width: 100%;
      padding: 10px 12px;
      font-size: 1em;
      font-family: inherit;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    input:focus, select:focus {
      outline: none;
      border-color: var(--teal);
      box-shadow: 0 0 0 3px rgba(15, 92, 102, 0.15);
    }

    button {
      margin-top: 10px;
      width: 100%;
      padding: 13px;
      font-family: "Poppins", "Inter", sans-serif;
      font-size: 1em;
      font-weight: 600;
      color: #fff;
      background: var(--teal);
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.15s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }

    button:hover:not(:disabled) { background: var(--teal-hover); }
    button:disabled { opacity: 0.75; cursor: not-allowed; }

    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.4);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      display: none;
    }

    button.loading .spinner { display: inline-block; }

    @keyframes spin { to { transform: rotate(360deg); } }

    /* --- Result panel --- */

    .result-panel h2 {
      font-family: "Poppins", "Inter", sans-serif;
      font-size: 1.05em;
      margin: 0 0 18px;
      color: var(--navy);
    }

    .result-body {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }

    .placeholder {
      text-align: center;
      color: var(--muted);
      font-size: 0.92em;
      padding: 20px 10px;
    }

    .estimate-label {
      font-size: 0.8em;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      text-align: center;
      margin-bottom: 6px;
    }

    .estimate-value {
      font-family: "Poppins", "Inter", sans-serif;
      font-size: 2.4em;
      font-weight: 700;
      color: var(--teal);
      text-align: center;
      margin-bottom: 22px;
    }

    .compare-title {
      font-size: 0.78em;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 10px;
    }

    .compare-row { margin-bottom: 12px; }

    .compare-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.82em;
      color: var(--text);
      margin-bottom: 4px;
    }

    .compare-track {
      background: #eef2f4;
      border-radius: 6px;
      height: 10px;
      overflow: hidden;
    }

    .compare-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.6s ease;
    }

    .compare-fill.you { background: var(--amber); }
    .compare-fill.ref { background: var(--teal); opacity: 0.55; }

    .compare-note {
      margin-top: 6px;
      font-size: 0.72em;
      color: var(--muted);
      font-style: italic;
    }

    .warning-box {
      background: var(--warn-bg);
      border: 1px solid var(--warn-border);
      color: var(--warn-text);
      border-radius: 10px;
      padding: 16px 18px;
      font-size: 0.9em;
      line-height: 1.4;
    }

    .warning-box strong { display: block; margin-bottom: 4px; }

    footer {
      margin-top: 28px;
      text-align: center;
      color: var(--muted);
      font-size: 0.78em;
    }
  </style>
</head>
<body>
  <div class="heading">
    <h1>Insurance Charge Predictor</h1>
    <p class="subtitle">A linear regression model estimates your annual medical insurance charge from your personal and health profile.</p>
  </div>

  <div class="layout">
    <div class="panel form-panel">
      <form id="predict-form">
        <div class="section">
          <div class="section-heading">Personal Details</div>

          <div class="field">
            <label for="age">Age</label>
            <input type="number" id="age" min="18" max="100" placeholder="18-100" required />
            <div class="hint">Valid range: 18-100</div>
          </div>

          <div class="field">
            <label for="sex">Sex</label>
            <select id="sex">
              <option value="0">Female</option>
              <option value="1">Male</option>
            </select>
          </div>

          <div class="field">
            <label for="bmi">BMI</label>
            <input type="number" id="bmi" min="10" max="60" step="0.1" placeholder="10-60" required />
            <div class="hint">Valid range: 10-60</div>
          </div>

          <div class="field">
            <label for="children">Children</label>
            <input type="number" id="children" min="0" max="10" placeholder="0-10" required />
            <div class="hint">Valid range: 0-10</div>
          </div>
        </div>

        <div class="section">
          <div class="section-heading">Health &amp; Location</div>

          <div class="field">
            <label for="smoker">Smoker</label>
            <select id="smoker">
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>

          <div class="field">
            <label for="region">Region</label>
            <select id="region">
              <option value="northeast">Northeast</option>
              <option value="northwest">Northwest</option>
              <option value="southeast">Southeast</option>
              <option value="southwest">Southwest</option>
            </select>
          </div>
        </div>

        <button type="submit" id="submit-btn">
          <span class="spinner"></span>
          <span class="btn-label">Predict</span>
        </button>
      </form>
    </div>

    <div class="panel result-panel">
      <h2>Your Estimate</h2>
      <div class="result-body" id="result-body">
        <p class="placeholder">Fill in your details to see your estimated cost.</p>
      </div>
    </div>
  </div>

  <footer>Built with scikit-learn Linear Regression | Trained on 1,337 records</footer>

  <script>
    const form = document.getElementById("predict-form");
    const resultBody = document.getElementById("result-body");
    const submitBtn = document.getElementById("submit-btn");
    const btnLabel = submitBtn.querySelector(".btn-label");

    // Static reference points, honestly labeled as dataset averages (not a live query).
    const REFERENCES = [
      { label: "Dataset average", value: 13279 },
      { label: "Non-smoker average", value: 8434 },
      { label: "Smoker average", value: 32050 },
    ];

    function formatCurrency(value) {
      return "$" + Number(value).toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      });
    }

    function buildComparisonHTML(charge) {
      const maxValue = Math.max(charge, ...REFERENCES.map(r => r.value));

      const yourRow = `
        <div class="compare-row">
          <div class="compare-label"><span>Your estimate</span><span>${formatCurrency(charge)}</span></div>
          <div class="compare-track"><div class="compare-fill you" style="width:${(charge / maxValue) * 100}%"></div></div>
        </div>`;

      const refRows = REFERENCES.map(r => `
        <div class="compare-row">
          <div class="compare-label"><span>${r.label}</span><span>${formatCurrency(r.value)}</span></div>
          <div class="compare-track"><div class="compare-fill ref" style="width:${(r.value / maxValue) * 100}%"></div></div>
        </div>`).join("");

      return `
        <div class="compare-title">How this compares (dataset averages)</div>
        ${yourRow}
        ${refRows}
        <div class="compare-note">Reference values are static averages from the training dataset, not a live query.</div>`;
    }

    // Counts a number up from 0 to target over ~800ms for a livelier reveal.
    function animateValue(el, target, duration = 800) {
      const start = performance.now();
      function tick(now) {
        const progress = Math.min((now - start) / duration, 1);
        const current = target * progress;
        el.textContent = formatCurrency(current);
        if (progress < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }

    function showSuccess(charge) {
      resultBody.innerHTML = `
        <div class="estimate-label">Predicted Annual Charge</div>
        <div class="estimate-value" id="estimate-value">$0.00</div>
        <div id="comparison"></div>`;
      animateValue(document.getElementById("estimate-value"), charge);
      document.getElementById("comparison").innerHTML = buildComparisonHTML(charge);
    }

    function showWarning(message) {
      resultBody.innerHTML = `
        <div class="warning-box">
          <strong>Please check your inputs</strong>
          ${message}
        </div>`;
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault(); // stop the browser from doing a normal page reload on submit

      const payload = {
        age: Number(document.getElementById("age").value),
        sex: Number(document.getElementById("sex").value),
        bmi: Number(document.getElementById("bmi").value),
        children: Number(document.getElementById("children").value),
        smoker: Number(document.getElementById("smoker").value),
        region: document.getElementById("region").value,
      };

      submitBtn.disabled = true;
      submitBtn.classList.add("loading");
      btnLabel.textContent = "Calculating...";

      try {
        // Relative URL: same origin as this page, so no CORS is involved.
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (response.status === 422) {
          showWarning("Please enter valid values (age 18-100, bmi 10-60, children 0-10).");
          return;
        }

        if (!response.ok) throw new Error(`Server returned ${response.status}`);

        const data = await response.json();
        showSuccess(data.predicted_charge);
      } catch (err) {
        showWarning(`Something went wrong: ${err.message}`);
      } finally {
        submitBtn.disabled = false;
        submitBtn.classList.remove("loading");
        btnLabel.textContent = "Predict";
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def form():
    return FORM_HTML


class PredictRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Literal[0, 1]
    bmi: float = Field(ge=10, le=60)
    children: int = Field(ge=0, le=10)
    smoker: Literal[0, 1]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


@app.post("/predict")
def predict(data: PredictRequest):
    # Northeast is the dropped baseline, so all region_* columns stay 0 for it.
    row = {
        "age": data.age,
        "sex": data.sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": data.smoker,
        "region_northwest": 1 if data.region == "northwest" else 0,
        "region_southeast": 1 if data.region == "southeast" else 0,
        "region_southwest": 1 if data.region == "southwest" else 0,
    }

    # Build the DataFrame with columns in the exact order the model was trained on.
    df = pd.DataFrame([row], columns=feature_order)

    prediction = model.predict(df)[0]

    return {"predicted_charge": round(float(prediction), 2)}
