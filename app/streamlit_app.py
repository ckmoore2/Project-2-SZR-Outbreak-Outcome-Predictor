"""
Streamlit app for the SZR Outbreak Outcome Predictor.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import torch
from scipy.integrate import odeint

# Allow importing from model/ and data/ when running from the project root
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_APP_DIR)
sys.path.insert(0, os.path.join(_ROOT, "model"))
sys.path.insert(0, os.path.join(_ROOT, "data"))

from szr_predictor import SZRPredictor          # noqa: E402
from nc_county_profiles import build_hsi_overrides  # noqa: E402

OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "best_model.pt")
SCALER_PATH = os.path.join(OUTPUTS_DIR, "scaler.pkl")

# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model_and_scaler():
    scaler = joblib.load(SCALER_PATH)
    model = SZRPredictor(input_dim=8, hidden_dims=[64, 128], output_dim=3)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model, scaler


_FEATURE_COLS = [
    "beta", "zeta", "alpha", "initial_population", "initial_infected",
    "mobility_score", "infrastructure_score", "health_score",
]
_DATA_PATH = os.path.join(_ROOT, "data", "szr_synthetic.csv")


@st.cache_resource
def load_shap_explainer():
    """Build a DeepExplainer backed by 100 randomly sampled background rows."""
    model, scaler = load_model_and_scaler()
    df = pd.read_csv(_DATA_PATH)
    bg_rows = df[_FEATURE_COLS].sample(n=100, random_state=42)
    bg_scaled = torch.FloatTensor(scaler.transform(bg_rows))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.DeepExplainer(model, bg_scaled)
    return explainer


# ---------------------------------------------------------------------------
# SZR ODE simulation
# ---------------------------------------------------------------------------


def szr_odes(y, t, beta, zeta, alpha, N):
    """SZR ODE system with frequency-dependent transmission (S*Z/N)."""
    S, Z, R = y
    dS_dt = -beta * S * Z / N
    dZ_dt = beta * S * Z / N - zeta * Z - alpha * Z
    dR_dt = zeta * Z + alpha * Z
    return [dS_dt, dZ_dt, dR_dt]


def run_simulation(beta, zeta, alpha, initial_population, initial_infected,
                   mobility_score, t_max=180):
    effective_beta = beta * (1.0 - mobility_score)
    N = initial_population
    Z0 = min(initial_infected, N - 1)
    S0 = N - Z0
    R0 = 0.0
    t = np.linspace(0, t_max, t_max + 1)
    sol = odeint(szr_odes, [S0, Z0, R0], t,
                 args=(effective_beta, zeta, alpha, N),
                 mxstep=5000)
    return t, sol[:, 0] / N, sol[:, 1] / N, sol[:, 2] / N


# ---------------------------------------------------------------------------
# County profiles — HSI scores derived from DASC 6010 team data
# ---------------------------------------------------------------------------

# Base profiles: epidemiological params and populations stay hardcoded.
# HSI scores (mobility, health, infrastructure) are overwritten below with
# values derived from the DASC 6010 census-tract CSVs in data/dasc6010/.
_BASE_PROFILES = {
    "Custom (use sliders)": None,
    "Mecklenburg (Charlotte)": {
        "beta": 0.45, "zeta": 0.08, "alpha": 0.005,
        "initial_population": 1_115_000, "initial_infected": 20,
    },
    "Wake (Raleigh)": {
        "beta": 0.38, "zeta": 0.10, "alpha": 0.005,
        "initial_population": 1_130_000, "initial_infected": 15,
    },
    "Dare (Outer Banks)": {
        "beta": 0.20, "zeta": 0.15, "alpha": 0.005,
        "initial_population": 38_000, "initial_infected": 3,
    },
    "Tyrrell (Rural)": {
        "beta": 0.15, "zeta": 0.18, "alpha": 0.005,
        "initial_population": 4_000, "initial_infected": 1,
    },
    "Robeson (Mixed)": {
        "beta": 0.30, "zeta": 0.12, "alpha": 0.005,
        "initial_population": 130_000, "initial_infected": 8,
    },
}

# Merge data-derived HSI scores into the base profiles
_hsi_overrides = build_hsi_overrides()
COUNTY_PROFILES = {}
for _name, _base in _BASE_PROFILES.items():
    if _base is None:
        COUNTY_PROFILES[_name] = None
    else:
        COUNTY_PROFILES[_name] = {**_base, **_hsi_overrides.get(_name, {})}

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("SZR Outbreak Outcome Predictor")
st.markdown(
    "Adjust the parameters in the sidebar and click **Predict Outbreak** "
    "to see neural-network predictions and the ground-truth ODE simulation."
)

st.sidebar.header("Outbreak Parameters")

# Seed session_state defaults on first render so sliders have a home value
_DEFAULTS = {
    "beta": 0.3, "zeta": 0.1, "alpha": 0.005,
    "initial_population": 100_000, "initial_infected": 5,
    "mobility_score": 0.5, "infrastructure_score": 0.5, "health_score": 0.5,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _apply_county_profile():
    profile = COUNTY_PROFILES[st.session_state["county_profile"]]
    if profile is not None:
        for k, v in profile.items():
            st.session_state[k] = v


st.sidebar.selectbox(
    "Quick-load a North Carolina county profile",
    options=list(COUNTY_PROFILES.keys()),
    key="county_profile",
    on_change=_apply_county_profile,
)
st.sidebar.caption(
    "HSI scores from DASC 6010 team data (nc_health.csv, nc_mobility.csv, "
    "nc_infrastructure.csv). "
    "Mobility inverted: 1 − score_M (escape capacity → spread risk)."
)

beta = st.sidebar.slider(
    "Transmission rate (β)", min_value=0.001, max_value=0.9,
    value=0.3, step=0.001, format="%.3f", key="beta",
    help="Base rate at which susceptibles become zombies per contact."
)
zeta = st.sidebar.slider(
    "Recovery/removal rate (ζ)", min_value=0.001, max_value=0.5,
    value=0.1, step=0.001, format="%.3f", key="zeta",
    help="Rate at which zombies are removed (quarantined/destroyed)."
)
alpha = st.sidebar.slider(
    "Natural death rate of zombies (α)", min_value=0.0001, max_value=0.01,
    value=0.005, step=0.0001, format="%.4f", key="alpha",
    help="Natural decay rate of the zombie population."
)
initial_population = st.sidebar.slider(
    "Initial population", min_value=5000, max_value=1_200_000,
    value=100_000, step=1000, key="initial_population",
    help="Total county population at outbreak start."
)
initial_infected = st.sidebar.slider(
    "Initial infected (Z₀)", min_value=1, max_value=50,
    value=5, step=1, key="initial_infected",
    help="Number of zombies at time 0."
)
mobility_score = st.sidebar.slider(
    "Mobility restriction score", min_value=0.0, max_value=1.0,
    value=0.5, step=0.01, key="mobility_score",
    help="Higher score → more movement restrictions → less transmission. "
         "effective β = β × (1 − mobility_score)."
)
infrastructure_score = st.sidebar.slider(
    "Infrastructure score", min_value=0.0, max_value=1.0,
    value=0.5, step=0.01, key="infrastructure_score",
    help="HSI: quality of roads, utilities, and logistics."
)
health_score = st.sidebar.slider(
    "Health score", min_value=0.0, max_value=1.0,
    value=0.5, step=0.01, key="health_score",
    help="HSI: baseline health-system capacity."
)

if st.button("Predict Outbreak"):
    # -----------------------------------------------------------------------
    # Load artifacts
    # -----------------------------------------------------------------------
    try:
        model, scaler = load_model_and_scaler()
    except FileNotFoundError as exc:
        st.error(
            f"Model or scaler not found ({exc}). "
            "Please run `python data/generate_data.py` then "
            "`python model/train.py` first."
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Neural-network prediction
    # -----------------------------------------------------------------------
    raw_input = pd.DataFrame([[
        beta, zeta, alpha, initial_population, initial_infected,
        mobility_score, infrastructure_score, health_score,
    ]], columns=["beta", "zeta", "alpha", "initial_population",
                 "initial_infected", "mobility_score",
                 "infrastructure_score", "health_score"])

    scaled_input = scaler.transform(raw_input)
    with torch.no_grad():
        out = model(torch.FloatTensor(scaled_input)).numpy()[0]

    pred_peak_fraction = float(np.clip(out[0], 0.0, 1.0))
    pred_time_to_peak = float(np.clip(out[1], 0.0, 180.0))
    containment_prob = float(torch.sigmoid(torch.tensor(out[2])).item())

    # -----------------------------------------------------------------------
    # Display predictions
    # -----------------------------------------------------------------------
    st.subheader("Neural Network Predictions")

    col1, col2 = st.columns(2)
    col1.metric("Peak Zombie Fraction", f"{pred_peak_fraction:.3f}",
                help="Fraction of population that becomes zombies at peak.")
    col2.metric("Time to Peak (days)", f"{pred_time_to_peak:.1f}",
                help="Day at which zombie population reaches its maximum.")

    st.write("**Containment Probability**")
    if containment_prob > 0.60:
        bar_color = "green"
        outcome_label = "🟢 Likely Contained"
    elif containment_prob >= 0.30:
        bar_color = "orange"
        outcome_label = "🟡 Uncertain"
    else:
        bar_color = "red"
        outcome_label = "🔴 Unlikely Contained"

    st.progress(containment_prob, text=f"{outcome_label} ({containment_prob:.1%})")

    # -----------------------------------------------------------------------
    # ODE simulation
    # -----------------------------------------------------------------------
    st.subheader("ODE Simulation (Ground Truth)")

    t, S_frac, Z_frac, R_frac = run_simulation(
        beta, zeta, alpha, initial_population, initial_infected, mobility_score
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, S_frac, label="Susceptible (S)", color="steelblue")
    ax.plot(t, Z_frac, label="Zombie (Z)", color="firebrick")
    ax.plot(t, R_frac, label="Removed (R)", color="seagreen")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Fraction of population")
    ax.set_title("SZR Outbreak Dynamics")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Simulation = ground truth. Neural network prediction shown above."
    )

    # -----------------------------------------------------------------------
    # SHAP explanation
    # -----------------------------------------------------------------------
    st.subheader("Why did the model predict this?")

    explainer = load_shap_explainer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(torch.FloatTensor(scaled_input))
    # sv shape: (1, 8, 3) — axis 2 is output index; 0 = peak_zombie_fraction
    shap_vals = sv[0, :, 0]                        # shape (8,)
    base_val = float(explainer.expected_value[0])

    # Sort by absolute magnitude so largest contributors appear at top
    order = np.argsort(np.abs(shap_vals))          # ascending → top of barh = largest
    sorted_vals = shap_vals[order]
    sorted_names = [_FEATURE_COLS[i] for i in order]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sorted_vals]

    fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
    ax_shap.barh(sorted_names, sorted_vals, color=colors)
    ax_shap.axvline(0, color="black", linewidth=0.8)
    ax_shap.set_xlabel("SHAP value (impact on peak zombie fraction)")
    ax_shap.set_title(
        f"Feature contributions  |  base value = {base_val:.4f}  "
        f"→  prediction = {pred_peak_fraction:.4f}"
    )
    fig_shap.tight_layout()
    st.pyplot(fig_shap)
    plt.close(fig_shap)

    st.caption(
        "SHAP values show each feature's push on the prediction away from "
        "the dataset average"
    )
