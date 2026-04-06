"""
tests/test_app_smoke.py
End-to-end test: load model, run scaler, run simulation,
verify outputs are in expected ranges for each scenario.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import joblib
from scipy.integrate import odeint
from model.szr_predictor import SZRPredictor, FEATURE_COLUMNS

MODEL_PATH  = "outputs/best_model.pt"
SCALER_PATH = "outputs/scaler.pkl"

print("Loading model and scaler...")
model = SZRPredictor(input_dim=8, hidden_dims=[128, 256, 128], output_dim=3, dropout=0.2)
state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
scaler = joblib.load(SCALER_PATH)
print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters ✅")
print(f"  Features: {FEATURE_COLUMNS}")

TESTS = [
    {
        "name": "Pitt + Rabies (should be contained)",
        "inputs": {
            "beta": 0.05, "zeta": 0.25, "alpha": 0.01,
            "initial_population": 184226, "initial_infected": 185,
            "mobility_score": 0.42, "infrastructure_score": 0.44,
            "health_score": 0.46,
        },
        "expect_contained": True,
        "expect_peak_lt": 0.30,
    },
    {
        "name": "Pitt + 28 Days Later (should collapse)",
        "inputs": {
            "beta": 0.75, "zeta": 0.12, "alpha": 0.01,
            "initial_population": 184226, "initial_infected": 5,
            "mobility_score": 0.42, "infrastructure_score": 0.44,
            "health_score": 0.46,
        },
        "expect_contained": False,
        "expect_peak_gt": 0.20,
    },
    {
        "name": "Cumberland + Walking Dead (military — should survive)",
        "inputs": {
            "beta": 0.28, "zeta": 0.25, "alpha": 0.01,
            "initial_population": 337093, "initial_infected": 10,
            "mobility_score": 0.55, "infrastructure_score": 0.58,
            "health_score": 0.60,
        },
        "expect_contained": True,
        "expect_peak_lt": 0.50,
    },
]

import pandas as pd

PASS = FAIL = 0
print("\nRunning smoke tests...")
print("=" * 60)

for test in TESTS:
    name = test["name"]
    inp  = test["inputs"]

    # Scale inputs
    x_df = pd.DataFrame([inp], columns=FEATURE_COLUMNS)

    # Only select features that are in FEATURE_COLUMNS
    x_df = x_df[FEATURE_COLUMNS]
    x_sc = scaler.transform(x_df)
    x_t  = torch.FloatTensor(x_sc)

    with torch.no_grad():
        out = model(x_t).squeeze().numpy()

    peak_frac    = float(np.clip(out[0], 0, 1))
    time_to_peak = float(np.clip(out[1], 0, 365))
    contain_prob = float(torch.sigmoid(torch.tensor(out[2])).item())
    predicted_contained = contain_prob >= 0.5

    errors = []
    if "expect_contained" in test:
        if predicted_contained != test["expect_contained"]:
            errors.append(
                f"containment={predicted_contained} expected={test['expect_contained']}"
                f" (prob={contain_prob:.2%})"
            )
    if "expect_peak_lt" in test:
        if peak_frac >= test["expect_peak_lt"]:
            errors.append(f"peak={peak_frac:.1%} expected < {test['expect_peak_lt']:.0%}")
    if "expect_peak_gt" in test:
        if peak_frac <= test["expect_peak_gt"]:
            errors.append(f"peak={peak_frac:.1%} expected > {test['expect_peak_gt']:.0%}")

    status = "✅ PASS" if not errors else "❌ FAIL"
    if not errors:
        PASS += 1
    else:
        FAIL += 1

    print(f"\n{status} {name}")
    print(f"  Peak: {peak_frac:.1%}  |  Day: {time_to_peak:.0f}  |  Containment: {contain_prob:.1%}")
    for e in errors:
        print(f"  ⚠  {e}")

print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL:
    print("Model predictions are not matching expected qualitative outcomes.")
    print("Consider retraining with the updated scenario parameters.")
    sys.exit(1)
