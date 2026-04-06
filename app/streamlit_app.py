import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.special import expit as _expit
import shap
import os
import sys

# ── Project root on path so model/ is importable regardless of CWD ───────────
_APP_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.szr_predictor import SZRPredictor, FEATURE_COLUMNS
from model.szr_model import run_simulation_v1 as run_simulation
from model.config import SCENARIOS as _SCENARIOS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OUTBREAK RESPONSE TERMINAL",
    page_icon="☣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Terminal CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=Share+Tech+Mono&display=swap');

/* ── Global background ─────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0a08 !important;
    color: #c8bfa8 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
/* ── Headers ───────────────────────────────────────── */
h1 { font-family: 'Special Elite', cursive !important; color: #e8e0d0 !important; letter-spacing: 4px !important; font-size: 1.6rem !important; }
h2 { font-family: 'Special Elite', cursive !important; color: #cc2200 !important; letter-spacing: 3px !important; font-size: 1.1rem !important; border-bottom: 1px solid rgba(204,34,0,.3); padding-bottom: 6px; }
h3 { font-family: 'Share Tech Mono', monospace !important; color: #8db829 !important; letter-spacing: 2px !important; font-size: .9rem !important; }

/* ── Metric cards ──────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(204,34,0,.07) !important;
    border: 1px solid rgba(204,34,0,.3) !important;
    border-radius: 2px !important;
    padding: 14px 16px !important;
}
[data-testid="metric-container"] label { color: #6b5a48 !important; font-size: .7rem !important; letter-spacing: 2px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #cc2200 !important; font-family: 'Special Elite', cursive !important; font-size: 1.6rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #8db829 !important; }

/* ── Buttons ───────────────────────────────────────── */
[data-testid="baseButton-primary"], button[kind="primary"] {
    background: rgba(204,34,0,.15) !important;
    border: 1px solid rgba(204,34,0,.6) !important;
    color: #cc2200 !important;
    font-family: 'Special Elite', cursive !important;
    letter-spacing: 3px !important;
    border-radius: 2px !important;
}
[data-testid="baseButton-secondary"], button[kind="secondary"] {
    background: rgba(141,184,41,.08) !important;
    border: 1px solid rgba(141,184,41,.4) !important;
    color: #8db829 !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 2px !important;
}

/* ── Sliders ───────────────────────────────────────── */
[data-testid="stSlider"] { accent-color: #8db829; }
[data-testid="stSlider"] p { color: #8b8070 !important; font-size: .75rem !important; letter-spacing: 1px !important; }

/* ── Divider ───────────────────────────────────────── */
hr { border-color: rgba(204,34,0,.25) !important; }

/* ── Progress bar ──────────────────────────────────── */
[data-testid="stProgress"] > div > div { background: #8db829 !important; }

/* ── Info / warning boxes ──────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(204,34,0,.08) !important;
    border: 1px solid rgba(204,34,0,.35) !important;
    border-radius: 2px !important;
    color: #c8bfa8 !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Expander ──────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(204,34,0,.2) !important;
    background: rgba(10,10,8,.9) !important;
    border-radius: 2px !important;
}

/* ── Tables / dataframes ───────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(204,34,0,.2) !important;
}

/* ── Hide Streamlit top white header bar ───────────── */
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
header { display: none !important; height: 0 !important; }
.main .block-container { padding-top: 0.5rem !important; }

/* ── Scanline overlay ──────────────────────────────── */
body::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,.04) 2px, rgba(0,0,0,.04) 3px
    );
    pointer-events: none;
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

# ── Load model + scaler ───────────────────────────────────────────────────────
# SZRPredictor is imported from model/szr_predictor.py — single definition, no drift.
MODEL_PATH  = os.path.join(_PROJECT_ROOT, "outputs", "best_model.pt")
SCALER_PATH = os.path.join(_PROJECT_ROOT, "outputs", "scaler.pkl")

@st.cache_resource
def load_model():
    model = SZRPredictor(input_dim=8, hidden_dims=[128, 256, 128], output_dim=3, dropout=0.2)
    try:
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    except RuntimeError as e:
        err = str(e)
        if "network" in err and "net" in err:
            raise RuntimeError(
                f"State dict key mismatch — the app uses 'self.network' but the "
                f"saved model may use a different attribute name. "
                f"Check model/szr_predictor.py and ensure the Sequential is "
                f"assigned to 'self.network'. Original error: {err}"
            )
        raise
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


# ── Matplotlib terminal style ─────────────────────────────────────────────────
def terminal_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0a0a08")
    ax.set_facecolor("#0d0d0a")
    ax.tick_params(colors="#5a5040", labelsize=8)
    ax.xaxis.label.set_color("#5a5040")
    ax.yaxis.label.set_color("#5a5040")
    for spine in ax.spines.values():
        spine.set_edgecolor((0.80, 0.13, 0.0, 0.3))   # matplotlib tuple: rgba(204,34,0,0.3)
        spine.set_linewidth(0.6)
    ax.grid(True, color=(1.0, 1.0, 1.0, 0.04), linewidth=0.5, linestyle="--")
    return fig, ax


# ── Feature names / ranges ────────────────────────────────────────────────────
# IMPORTANT: FEATURE_ORDER must match FEATURE_COLUMNS in model/szr_predictor.py
# exactly — same names, same order — otherwise the scaler applies the wrong
# mean/std to each input and every prediction will be wrong.
# Current order mirrors: beta, zeta, alpha, initial_population, initial_infected,
#                        mobility_score, infrastructure_score, health_score
FEATURE_META = {
    "beta":               {"label": "Transmission Rate (β)",        "min": 0.0001, "max": 1.0,  "default": 0.25, "step": 0.01, "help": "Rate at which zombies infect susceptibles"},
    "zeta":               {"label": "Removal Rate (ζ)",             "min": 0.0001, "max": 0.5,  "default": 0.10, "step": 0.01, "help": "Rate at which zombies are neutralised"},
    "alpha":              {"label": "Natural Death Rate (α)",        "min": 0.001,"max": 0.05, "default": 0.01, "step": 0.001,"help": "Background natural mortality rate"},
    "initial_population": {"label": "County Population",            "min": 5000, "max": 1200000,"default": 300000,"step": 1000, "help": "Starting susceptible population"},
    "initial_infected":   {"label": "Initial Infected (Z₀)",        "min": 1,    "max": 5000, "default": 10,    "step": 1,    "help": "Seed zombie count at outbreak start"},
    "mobility_score":     {"label": "Mobility / Escape Score (M)",  "min": 0.0,  "max": 1.0,  "default": 0.50, "step": 0.01, "help": "HSI-M: evacuation routes, vehicle access, transit"},
    "infrastructure_score":{"label":"Infrastructure Score (I)",     "min": 0.0,  "max": 1.0,  "default": 0.50, "step": 0.01, "help": "HSI-I: power, water, food self-sufficiency"},
    "health_score":       {"label": "Health & Fitness Score (H)",   "min": 0.0,  "max": 1.0,  "default": 0.50, "step": 0.01, "help": "HSI-H: physical capability, medical access"},
}
FEATURE_ORDER = FEATURE_COLUMNS  # guaranteed to match model/szr_predictor.py


# ── Containment colour helper ─────────────────────────────────────────────────
def containment_color(p):
    if p >= 0.60:
        return "#8db829", "CONTAINED"
    elif p >= 0.30:
        return "#d4820a", "UNSTABLE"
    else:
        return "#cc2200", "COLLAPSE"


# ── County presets (NC census + real HSI from integrate_real_hsi.py) ──────────
COUNTY_PRESETS = {
    "— Select a county —": None,
    "Wake (Raleigh)":             {"initial_population": 1132103, "initial_infected": 5, "mobility_score": 1.000, "infrastructure_score": 0.222, "health_score": 0.423},
    "Mecklenburg (Charlotte)":    {"initial_population": 1115403, "initial_infected": 5, "mobility_score": 0.921, "infrastructure_score": 0.218, "health_score": 0.574},
    "Guilford (Greensboro)":      {"initial_population": 539557,  "initial_infected": 5, "mobility_score": 0.835, "infrastructure_score": 0.260, "health_score": 0.675},
    "Durham":                     {"initial_population": 325101,  "initial_infected": 5, "mobility_score": 0.768, "infrastructure_score": 0.103, "health_score": 0.695},
    "Forsyth (Winston-Salem)":    {"initial_population": 383739,  "initial_infected": 5, "mobility_score": 0.796, "infrastructure_score": 0.214, "health_score": 0.194},
    "Cumberland (Fayetteville)":  {"initial_population": 335207,  "initial_infected": 5, "mobility_score": 0.774, "infrastructure_score": 0.242, "health_score": 0.154},
    "Buncombe (Asheville)":       {"initial_population": 269449,  "initial_infected": 5, "mobility_score": 0.830, "infrastructure_score": 0.134, "health_score": 0.843},
    "Pitt (Greenville/ECU)":      {"initial_population": 171196,  "initial_infected": 5, "mobility_score": 0.620, "infrastructure_score": 0.252, "health_score": 0.333},
    "New Hanover (Wilmington)":   {"initial_population": 228134,  "initial_infected": 5, "mobility_score": 0.775, "infrastructure_score": 0.170, "health_score": 0.721},
    "Alamance (Burlington)":      {"initial_population": 171779,  "initial_infected": 5, "mobility_score": 0.819, "infrastructure_score": 0.171, "health_score": 0.611},
    "Onslow (Jacksonville)":      {"initial_population": 203686,  "initial_infected": 5, "mobility_score": 0.884, "infrastructure_score": 0.316, "health_score": 0.656},
    "Union (Monroe)":             {"initial_population": 240109,  "initial_infected": 5, "mobility_score": 0.948, "infrastructure_score": 0.447, "health_score": 0.578},
    "Wayne (Goldsboro)":          {"initial_population": 117480,  "initial_infected": 5, "mobility_score": 0.693, "infrastructure_score": 0.304, "health_score": 0.390},
    "Cabarrus (Concord)":         {"initial_population": 226396,  "initial_infected": 5, "mobility_score": 0.897, "infrastructure_score": 0.161, "health_score": 0.288},
    "Orange (Chapel Hill)":       {"initial_population": 145919,  "initial_infected": 5, "mobility_score": 0.820, "infrastructure_score": 0.396, "health_score": 0.514},
}

# ── Outbreak scenario presets ─────────────────────────────────────────────────
# SHOW_PRESETS: built from model/config.py SCENARIOS — single source of truth.
# Epidemiological params (beta, kappa, alpha) come from config; only emoji,
# initial_infected, and description are defined here.
_SHOW_MAP = [
    ("🧟", "twd",    10, "Slow walkers, high removal — humans likely survive"),
    ("🍄", "tlou",   20, "Fungal spread, organised clickers — marginal survival"),
    ("🌍", "wwz",    50, "Fast movers, global scale — NC likely holds"),
    ("⚡", "28days",  5, "Rage virus, extreme spread — zombies win in NC"),
    ("🦠", "rabies",  2, "Real-world ceiling — humans dominate easily"),
]
SHOW_PRESETS = {"— Select a show scenario —": None}
SHOW_PRESETS.update({
    f"{emoji} {_SCENARIOS[key]['label']}": {
        "beta":             _SCENARIOS[key]["beta"],
        "zeta":             _SCENARIOS[key]["kappa"],
        "alpha":            _SCENARIOS[key]["alpha"],
        "initial_infected": z0,
        "label":            f"β={_SCENARIOS[key]['beta']:.4f} · α={_SCENARIOS[key]['alpha']:.2f} · {desc}",
        "group":            "show",
    }
    for emoji, key, z0, desc in _SHOW_MAP
})

# SEVERITY_PRESETS: operational outbreak levels (not show-specific)
SEVERITY_PRESETS = {
    "— Select a severity level —": None,
    "🟢 Early Detection": {
        "beta": 0.15, "zeta": 0.18, "alpha": 0.01, "initial_infected": 3,
        "label": "Low β · High ζ · Caught before community spread",
        "group": "severity",
    },
    "🟡 Active Spread": {
        "beta": 0.30, "zeta": 0.10, "alpha": 0.01, "initial_infected": 25,
        "label": "Moderate β · Standard removal rate",
        "group": "severity",
    },
    "🔴 Rapid Outbreak": {
        "beta": 0.55, "zeta": 0.08, "alpha": 0.01, "initial_infected": 100,
        "label": "High β · Overwhelmed response infrastructure",
        "group": "severity",
    },
    "☠  Total Collapse": {
        "beta": 0.80, "zeta": 0.04, "alpha": 0.01, "initial_infected": 500,
        "label": "Runaway infection · No effective containment",
        "group": "severity",
    },
    "🪖 Military Response": {
        "beta": 0.25, "zeta": 0.35, "alpha": 0.01, "initial_infected": 20,
        "label": "High removal · Fort Liberty / Camp Lejeune scenario",
        "group": "severity",
    },
    "🏥 Medical Containment": {
        "beta": 0.20, "zeta": 0.22, "alpha": 0.01, "initial_infected": 10,
        "label": "Active quarantine + treatment protocols",
        "group": "severity",
    },
}


# ── Session state defaults ────────────────────────────────────────────────────
DEFAULTS = {f: FEATURE_META[f]["default"] for f in FEATURE_ORDER}

def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = float(v)
    # Notification state — persists across reruns
    if "_toast_msg" not in st.session_state:
        st.session_state["_toast_msg"] = None
    if "_toast_icon" not in st.session_state:
        st.session_state["_toast_icon"] = "✅"

init_state()

# ── Fire any pending toast notification ──────────────────────────────────────
if st.session_state["_toast_msg"]:
    st.toast(st.session_state["_toast_msg"], icon=st.session_state["_toast_icon"])
    st.session_state["_toast_msg"] = None
    st.session_state["_toast_icon"] = "✅"


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='border-bottom:1px solid rgba(204,34,0,.3); padding:14px 0 12px; margin-bottom:16px; display:flex; align-items:center; justify-content:space-between;'>
  <div>
    <div style='font-family:"Special Elite",cursive; font-size:1.5rem; color:#e8e0d0; letter-spacing:4px;'>☣ OUTBREAK RESPONSE TERMINAL</div>
    <div style='font-size:.65rem; color:#6b5a48; letter-spacing:3px; margin-top:3px;'>NC DIVISION · CLASSIFIED LEVEL 4 · SURROGATE NEURAL MODEL ACTIVE</div>
  </div>
  <div style='text-align:right;'>
    <div style='font-size:.6rem; color:#5a5040; letter-spacing:2px;'>MODEL</div>
    <div style='font-family:"Special Elite",cursive; font-size:.9rem; color:#cc2200; letter-spacing:2px;'>SZRPredictor v2 · MLP [128→256→128]</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TWO-COLUMN CONTROL PANEL
# ═══════════════════════════════════════════════════════════════════════════════
ctrl_left, ctrl_right = st.columns([1, 1], gap="medium")

with ctrl_left:
    st.markdown("""<div style='border-right:1px solid rgba(204,34,0,.2); padding-right:8px;'>""",
                unsafe_allow_html=True)

    # ── Location ──────────────────────────────────────────────────────────────
    st.markdown("<div style='font-size:.7rem;color:#cc2200;letter-spacing:3px;margin-bottom:6px;'>◈ LOCATION</div>",
                unsafe_allow_html=True)
    county_choice = st.selectbox(
        "NC County",
        options=list(COUNTY_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Loads real population and estimated HSI scores for that county",
    )
    if county_choice != "— Select a county —":
        cp = COUNTY_PRESETS[county_choice]
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(
                f"<div style='font-size:.62rem;color:#5a5040;margin-top:2px;'>"
                f"{cp['initial_infected']} seed · pop {cp['initial_population']:,}</div>",
                unsafe_allow_html=True,
            )
        with col_b:
            if st.button("⬇ LOAD", type="secondary", use_container_width=True):
                try:
                    for k, v in cp.items():
                        st.session_state[k] = float(v)
                    st.session_state["_toast_msg"] = f"✔ {county_choice} loaded — pop {cp['initial_population']:,}"
                    st.session_state["_toast_icon"] = "✅"
                except Exception as e:
                    st.session_state["_toast_msg"] = f"Failed to load county data: {e}"
                    st.session_state["_toast_icon"] = "🚨"
                st.rerun()

    st.markdown("<div style='margin:10px 0 4px; border-top:1px solid rgba(204,34,0,.15);'></div>",
                unsafe_allow_html=True)

    # ── Zombie Show Scenarios ─────────────────────────────────────────────────
    st.markdown("<div style='font-size:.7rem;color:#cc2200;letter-spacing:3px;margin-bottom:6px;'>◈ ZOMBIE SHOW SCENARIO</div>",
                unsafe_allow_html=True)
    show_choice = st.selectbox(
        "Show scenario",
        options=list(SHOW_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Sets β, ζ, α from published epidemiological parameters for each show",
    )
    show_data = SHOW_PRESETS.get(show_choice)
    if show_data:
        st.markdown(
            f"<div style='font-size:.62rem;color:#d4820a;letter-spacing:1px;'>SHOW CANON</div>"
            f"<div style='font-size:.62rem;color:#8b8070;margin-top:2px;margin-bottom:4px;'>{show_data['label']}</div>",
            unsafe_allow_html=True,
        )
        if st.button("⬇ LOAD SHOW PARAMS", type="secondary", use_container_width=True):
            try:
                for k in ["beta", "zeta", "alpha", "initial_infected"]:
                    if k in show_data:
                        st.session_state[k] = float(show_data[k])
                st.session_state["_toast_msg"] = f"✔ {show_choice} loaded — β={show_data.get('beta', 'n/a'):.4f} · ζ={show_data.get('zeta', 'n/a'):.4f}"
                st.session_state["_toast_icon"] = "✅"
            except Exception as e:
                st.session_state["_toast_msg"] = f"Failed to load show params: {e}"
                st.session_state["_toast_icon"] = "🚨"
            st.rerun()

    st.markdown("<div style='margin:10px 0 4px; border-top:1px solid rgba(204,34,0,.15);'></div>",
                unsafe_allow_html=True)

    # ── Severity Levels ───────────────────────────────────────────────────────
    st.markdown("<div style='font-size:.7rem;color:#cc2200;letter-spacing:3px;margin-bottom:6px;'>◈ SEVERITY LEVEL</div>",
                unsafe_allow_html=True)
    severity_choice = st.selectbox(
        "Severity level",
        options=list(SEVERITY_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Operational outbreak severity — sets normalised SZR transmission and removal rates",
    )
    sev_data = SEVERITY_PRESETS.get(severity_choice)
    if sev_data:
        st.markdown(
            f"<div style='font-size:.62rem;color:#8db829;letter-spacing:1px;'>SEVERITY PRESET</div>"
            f"<div style='font-size:.62rem;color:#8b8070;margin-top:2px;margin-bottom:4px;'>{sev_data['label']}</div>",
            unsafe_allow_html=True,
        )
        if st.button("⬇ LOAD SEVERITY", type="secondary", use_container_width=True):
            try:
                for k in ["beta", "zeta", "alpha", "initial_infected"]:
                    if k in sev_data:
                        st.session_state[k] = float(sev_data[k])
                st.session_state["_toast_msg"] = f"✔ {severity_choice} loaded — {sev_data['label']}"
                st.session_state["_toast_icon"] = "✅"
            except Exception as e:
                st.session_state["_toast_msg"] = f"Failed to load severity preset: {e}"
                st.session_state["_toast_icon"] = "🚨"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with ctrl_right:
    # ── Fine-tune sliders ─────────────────────────────────────────────────────
    st.markdown("<div style='font-size:.7rem;color:#cc2200;letter-spacing:3px;margin-bottom:6px;'>◈ FINE-TUNE PARAMETERS</div>",
                unsafe_allow_html=True)

    inputs = {}
    sl1, sl2 = st.columns(2)
    feat_items = list(FEATURE_META.items())
    for idx, (feat, meta) in enumerate(feat_items):
        col = sl1 if idx % 2 == 0 else sl2
        with col:
            inputs[feat] = st.slider(
                meta["label"],
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(st.session_state.get(feat, meta["default"])),
                step=float(meta["step"]),
                help=meta["help"],
                key=feat,
            )

    st.markdown("<div style='margin:8px 0 4px; border-top:1px solid rgba(204,34,0,.15);'></div>",
                unsafe_allow_html=True)

    # ── Model version toggles ─────────────────────────────────────────────────
    st.markdown("<div style='font-size:.7rem;color:#cc2200;letter-spacing:3px;margin-bottom:6px;'>◈ MODEL VERSION</div>",
                unsafe_allow_html=True)
    tog1, tog2 = st.columns(2)
    with tog1:
        use_sigmoid = st.toggle(
            "Sigmoid κ (v2)",
            value=True,
            help="v2: sigmoid modifier captures threshold effects at HSI=0.5",
        )
    with tog2:
        use_tract_beta = st.toggle(
            "Tract-level β (v2)",
            value=True,
            help="v2: β varies by density (+40%) and mobility (-25%)",
        )
    st.markdown(
        "<div style='font-size:.6rem;color:#3a3028;margin-top:2px;line-height:1.5;'>"
        "v1: uniform β · linear κ &nbsp;|&nbsp; v2: tract β = f(density, mobility) · sigmoid κ"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin:10px 0 6px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("▶  RUN PREDICTION", type="primary", use_container_width=True)
    st.markdown("""
    <div style='font-size:.6rem; color:#3a3028; line-height:1.6; margin-top:6px;'>
    Model: SZRPredictor MLP [128→256→128] · Dataset: szr_synthetic.csv<br>
    Features: β, ζ, α, N₀, Z₀, HSI-H/M/I · Split: 80/10/10 · seed=42
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='border-top:1px solid rgba(204,34,0,.3); margin:16px 0 12px;'></div>",
            unsafe_allow_html=True)


# ── Sigmoid modifier helpers ──────────────────────────────────────────────────
def _sigmoid_kappa_modifier(hsi, scale=1.5, steepness=6.0):
    hsi = float(np.clip(hsi, 0.0, 1.0))
    return 1.0 + scale * (_expit(steepness * (hsi - 0.5)) - 0.5)

def _linear_kappa_modifier(hsi):
    return 1.0 + 0.5 * float(np.clip(hsi, 0.0, 1.0))


# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "☣  PREDICTION OUTPUT",
    "📡  ODE SIMULATION",
    "🧬  SHAP ANALYSIS",
    "📊  MODEL v1 vs v2",
])

if run_btn:
    # ── Build input tensor ────────────────────────────────────────────────────
    # Pass as DataFrame with named columns so StandardScaler doesn't warn
    x_raw_df = pd.DataFrame(
        [[inputs[f] for f in FEATURE_ORDER]],
        columns=FEATURE_ORDER,
    )

    try:
        scaler = load_scaler()
        x_scaled = scaler.transform(x_raw_df)
    except Exception:
        st.warning("⚠  Scaler not found — running unscaled. Retrain or check outputs/scaler.pkl.")
        x_scaled = x_raw_df.values

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    try:
        model = load_model()
        with torch.no_grad():
            raw_out = model(x_tensor).squeeze().numpy()
        peak_frac      = float(np.clip(raw_out[0], 0, 1))
        time_to_peak   = float(np.clip(raw_out[1], 0, 365))
        contain_logit  = float(raw_out[2])
        contain_prob   = float(torch.sigmoid(torch.tensor(contain_logit)).item())
        model_loaded   = True
        # ── Success toast ─────────────────────────────────────────────────────
        c_col, c_lbl = containment_color(contain_prob)
        st.toast(
            f"Prediction complete — Peak {peak_frac:.1%} · Day {time_to_peak:.0f} · {c_lbl}",
            icon="✅",
        )
    except Exception as e:
        st.error(f"Model load failed: {e}. Showing simulation-only mode.")
        model_loaded = False
        peak_frac, time_to_peak, contain_prob = None, None, None
        st.toast("Model unavailable — simulation ground truth only", icon="⚠️")

    # ── Run ODE ───────────────────────────────────────────────────────────────
    t_sim, sol = run_simulation(
        inputs["beta"], inputs["zeta"], inputs["alpha"],
        inputs["initial_population"], inputs["initial_infected"],
    )
    S, Z, R = sol[:, 0], sol[:, 1], sol[:, 2]
    sim_peak_frac    = float(np.max(Z) / inputs["initial_population"])
    sim_peak_day     = float(t_sim[np.argmax(Z)])
    sim_survive_frac = float(S[-1] / inputs["initial_population"])

    # ── Status strip — shows below control panel, above tabs ─────────────────
    status_color = "#8db829" if sim_survive_frac > 0.10 else "#cc2200"
    status_label = "CONTAINED" if sim_survive_frac > 0.60 else \
                   "UNSTABLE"  if sim_survive_frac > 0.10 else "COLLAPSE"
    st.markdown(
        f"<div style='"
        f"background:rgba({'141,184,41' if sim_survive_frac > 0.10 else '204,34,0'},.08);"
        f"border:1px solid {status_color}44;"
        f"padding:8px 16px;margin-bottom:10px;"
        f"display:flex;align-items:center;justify-content:space-between;'>"
        f"<div style='font-size:.7rem;color:{status_color};letter-spacing:2px;'>"
        f"◉ SIMULATION COMPLETE — STATUS: {status_label}</div>"
        f"<div style='font-size:.7rem;color:#5a5040;font-family:Share Tech Mono,monospace;'>"
        f"Peak {sim_peak_frac:.1%} · Day {sim_peak_day:.0f} · "
        f"Survivors {sim_survive_frac:.1%} · β={inputs['beta']:.4f} · ζ={inputs['zeta']:.4f}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 1 — PREDICTION
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("## ◈ NEURAL NETWORK PREDICTIONS")

        if model_loaded:
            c_color, c_label = containment_color(contain_prob)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PEAK ZOMBIE FRACTION", f"{peak_frac:.1%}",
                          delta=f"SIM: {sim_peak_frac:.1%}")
            with col2:
                st.metric("TIME TO PEAK (DAYS)", f"{time_to_peak:.0f}d",
                          delta=f"SIM: {sim_peak_day:.0f}d")
            with col3:
                st.metric("CONTAINMENT PROBABILITY", f"{contain_prob:.1%}", delta=c_label)

            st.markdown("---")
            st.markdown(f"### ◈ CONTAINMENT STATUS: <span style='color:{c_color};font-family:Special Elite,cursive;letter-spacing:3px;'>{c_label}</span>", unsafe_allow_html=True)
            st.progress(contain_prob)

            st.markdown("""
            <div style='font-size:.7rem; color:#3a3028; margin-top:8px;'>
            ↑ Neural network output · Model: SZRPredictor [128→256→128] · Experiment D
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Model unavailable — displaying simulation ground truth only.")

        st.markdown("---")
        st.markdown("## ◈ GROUND TRUTH — SIMULATION METRICS")
        s1, s2, s3 = st.columns(3)
        s1.metric("SIM PEAK ZOMBIE FRAC", f"{sim_peak_frac:.1%}")
        s2.metric("SIM PEAK DAY", f"{sim_peak_day:.0f}d")
        s3.metric("SIM SURVIVORS", f"{sim_survive_frac:.1%}")

        st.markdown("""
        <div style='font-size:.7rem; color:#3a3028; margin-top:4px;'>
        Simulation = ground truth. Neural network prediction shown above. HSI weights: H=0.15 · M=0.15 · I=0.20
        </div>""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 2 — ODE SIMULATION
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("## ◈ SZR EPIDEMIOLOGICAL SIMULATION")

        fig, ax = terminal_fig(figsize=(11, 4.5))
        ax.plot(t_sim, S / inputs["initial_population"], color="#8db829", linewidth=1.8, label="Susceptible (S)")
        ax.plot(t_sim, Z / inputs["initial_population"], color="#cc2200", linewidth=1.8, label="Zombie (Z)")
        ax.plot(t_sim, R / inputs["initial_population"], color="#5a5040", linewidth=1.2, label="Removed (R)", linestyle="--")
        ax.axvline(sim_peak_day, color="#cc2200", alpha=0.4, linewidth=0.8, linestyle=":")
        ax.text(sim_peak_day + 2, sim_peak_frac * 0.95, f"  PEAK D{sim_peak_day:.0f}",
                color="#cc2200", fontsize=7, fontfamily="monospace")
        ax.set_xlabel("Days since outbreak", fontfamily="monospace", fontsize=8)
        ax.set_ylabel("Fraction of population", fontfamily="monospace", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        legend = ax.legend(facecolor="#0d0d0a", edgecolor=(0.80, 0.13, 0.0, 0.3),
                           labelcolor="#8b8070", fontsize=8)
        st.pyplot(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("## ◈ COUNTY-LEVEL SCENARIO COMPARISON")

        COUNTIES = {
            "Wake (High Density)":       {"pop": 1117742, "inf": 50,  "mob": 0.58, "infra": 0.62, "health": 0.61},
            "Pitt (ECU — Moderate)":     {"pop": 184226,  "inf": 5,   "mob": 0.42, "infra": 0.44, "health": 0.46},
            "Buncombe (Rural/Terrain)":  {"pop": 274089,  "inf": 3,   "mob": 0.48, "infra": 0.52, "health": 0.55},
            "Cumberland (Military)":     {"pop": 337093,  "inf": 10,  "mob": 0.55, "infra": 0.58, "health": 0.60},
        }

        compare_rows = []
        for cname, cp in COUNTIES.items():
            ct, csol = run_simulation(inputs["beta"], inputs["zeta"], inputs["alpha"],
                                      cp["pop"], cp["inf"])
            cZ = csol[:, 1]
            compare_rows.append({
                "County": cname,
                "Peak Z%": f"{np.max(cZ)/cp['pop']:.1%}",
                "Peak Day": f"{ct[np.argmax(cZ)]:.0f}",
                "Survivors": f"{csol[-1,0]/cp['pop']:.1%}",
                "HSI-M": f"{cp['mob']:.2f}",
                "HSI-I": f"{cp['infra']:.2f}",
                "HSI-H": f"{cp['health']:.2f}",
            })
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 3 — SHAP
    # ═════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("## ◈ SHAP FEATURE IMPORTANCE")

        if model_loaded:
            try:
                def model_fn_peak(X_np):
                    t = torch.tensor(X_np.astype(np.float32))
                    with torch.no_grad():
                        return model(t).numpy()[:, 0]

                def model_fn_contain(X_np):
                    t = torch.tensor(X_np.astype(np.float32))
                    with torch.no_grad():
                        logits = model(t).numpy()[:, 2]
                    return 1 / (1 + np.exp(-logits))

                background = np.tile(x_scaled, (50, 1)) + np.random.normal(0, 0.05, (50, 8))
                background = background.astype(np.float32)

                exp_peak    = shap.Explainer(model_fn_peak,    background)
                exp_contain = shap.Explainer(model_fn_contain, background)
                sv_peak    = exp_peak(x_scaled)
                sv_contain = exp_contain(x_scaled)

                feat_labels = [FEATURE_META[f]["label"] for f in FEATURE_ORDER]

                fig2, axes = plt.subplots(1, 2, figsize=(11, 4))
                for ax2, sv, title, color in zip(
                    axes,
                    [sv_peak, sv_contain],
                    ["Peak Zombie Fraction", "Containment Probability"],
                    ["#cc2200", "#8db829"],
                ):
                    vals  = sv.values[0]
                    order = np.argsort(np.abs(vals))[::-1]
                    bars  = ax2.barh(
                        [feat_labels[i] for i in order],
                        [vals[i] for i in order],
                        color=[color if vals[i] > 0 else "#4a3020" for i in order],
                        edgecolor="none", height=0.6,
                    )
                    ax2.axvline(0, color="#3a3028", linewidth=0.8)
                    ax2.set_facecolor("#0d0d0a")
                    fig2.patch.set_facecolor("#0a0a08")
                    ax2.tick_params(colors="#5a5040", labelsize=7)
                    ax2.set_title(title, color="#8b8070", fontsize=8, fontfamily="monospace", pad=8)
                    for spine in ax2.spines.values():
                        spine.set_edgecolor((0.80, 0.13, 0.0, 0.2))
                        spine.set_linewidth(0.5)

                plt.tight_layout(pad=1.5)
                st.pyplot(fig2, use_container_width=True)
                st.markdown("""
                <div style='font-size:.7rem; color:#3a3028; margin-top:6px;'>
                SHAP values computed via KernelExplainer on 50 background samples.
                Positive = increases output · Negative = decreases output.
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")
        else:
            st.info("Load the model to enable SHAP analysis.")

        st.markdown("---")
        st.markdown("## ◈ ABLATION RESULTS")

        try:
            abl_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "ablation_results.csv")
            abl = pd.read_csv(abl_path)
            st.dataframe(abl, use_container_width=True, hide_index=True)
        except FileNotFoundError:
            st.markdown("""
            <div style='font-size:.8rem; color:#6b5a48; padding:10px; border:1px solid rgba(204,34,0,.2);'>
            ablation_results.csv not found at outputs/ablation_results.csv.<br>
            Run model/train.py to generate it.
            </div>""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  TAB 4 — MODEL v1 vs v2 COMPARISON
    # ═════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("## ◈ LINEAR vs SIGMOID κ MODIFIER")
        st.markdown("""
        <div style='font-size:.75rem; color:#6b5a48; margin-bottom:14px; line-height:1.7;'>
        v1 assumes killing effectiveness scales linearly with HSI.
        v2 uses a sigmoid — communities below a coordination threshold kill poorly;
        above it they become dramatically more effective. The inflection is at HSI=0.5.
        </div>""", unsafe_allow_html=True)

        hsi_range = np.linspace(0, 1, 200)
        linear_vals  = [_linear_kappa_modifier(h) for h in hsi_range]
        sigmoid_vals = [_sigmoid_kappa_modifier(h) for h in hsi_range]

        fig_mod, ax_mod = terminal_fig(figsize=(10, 3.5))
        ax_mod.plot(hsi_range, linear_vals,  color="#5a5040", linewidth=1.5,
                    linestyle="--", label="v1 linear:  κ_eff = κ_base × (1 + 0.5·HSI)")
        ax_mod.plot(hsi_range, sigmoid_vals, color="#cc2200", linewidth=2.0,
                    label="v2 sigmoid: κ_eff = κ_base × sigmoid_modifier(HSI)")
        ax_mod.axvline(0.5, color="#8db829", linewidth=0.7, alpha=0.5, linestyle=":")
        ax_mod.axhline(1.0, color="#3a3028", linewidth=0.5)
        ax_mod.text(0.52, 0.78, "inflection\nHSI=0.5", color="#8db829",
                    fontsize=7, fontfamily="monospace", transform=ax_mod.transAxes)
        ax_mod.set_xlabel("HSI score", fontfamily="monospace", fontsize=8)
        ax_mod.set_ylabel("κ multiplier", fontfamily="monospace", fontsize=8)
        legend = ax_mod.legend(facecolor="#0d0d0a", edgecolor=(0.80, 0.13, 0.0, 0.3),
                               labelcolor="#8b8070", fontsize=8)
        st.pyplot(fig_mod, use_container_width=True)

        st.markdown("---")
        st.markdown("## ◈ TRACT-LEVEL β VARIATION")
        st.markdown("""
        <div style='font-size:.75rem; color:#6b5a48; margin-bottom:14px; line-height:1.7;'>
        v1: β is uniform across all tracts regardless of density or mobility.<br>
        v2: β_eff(tract) = β_base × density_modifier × encounter_modifier.<br>
        This gives the M (Mobility) section direct influence on spread rate, not just kill rate.
        </div>""", unsafe_allow_html=True)

        beta_base_display = inputs["beta"]
        density_range = np.linspace(0, 1, 100)
        mob_levels = [0.2, 0.5, 0.8]
        mob_labels = ["Low mobility (0.2)", "Mid mobility (0.5)", "High mobility (0.8)"]
        mob_colors = ["#cc2200", "#d4820a", "#8db829"]

        fig_beta, ax_beta = terminal_fig(figsize=(10, 3.5))
        ax_beta.axhline(beta_base_display, color="#5a5040", linewidth=1.2,
                        linestyle="--", label=f"v1 uniform β = {beta_base_display:.3f}")
        for mob, label, color in zip(mob_levels, mob_labels, mob_colors):
            betas = [
                beta_base_display * (1.0 + 0.40 * d) * (1.0 - 0.25 * mob)
                for d in density_range
            ]
            ax_beta.plot(density_range, betas, color=color, linewidth=1.6, label=f"v2 {label}")
        ax_beta.set_xlabel("Normalised population density", fontfamily="monospace", fontsize=8)
        ax_beta.set_ylabel("β_eff", fontfamily="monospace", fontsize=8)
        legend2 = ax_beta.legend(facecolor="#0d0d0a", edgecolor=(0.80, 0.13, 0.0, 0.3),
                                 labelcolor="#8b8070", fontsize=8)
        st.pyplot(fig_beta, use_container_width=True)

        st.markdown("---")
        st.markdown("## ◈ NEW HSI SUB-FACTORS — FETCH STATUS")
        fetch_status = [
            {"Factor": "Hunting License Density",      "Category": "C", "Data Source": "NCWRC Annual Report",      "Status": "⚠ Manual download needed",   "Script": "fetch_category_c_extensions.py"},
            {"Factor": "Congregation Density",          "Category": "C", "Data Source": "USDA ERS Rural Atlas",     "Status": "⚠ Partial (proxy available)", "Script": "fetch_category_c_extensions.py"},
            {"Factor": "Volunteer Fire Dept Coverage",  "Category": "C", "Data Source": "USFA NFIRS",              "Status": "⚠ Fetch attempt in script",   "Script": "fetch_category_c_extensions.py"},
            {"Factor": "Waterway Barrier Score",        "Category": "G", "Data Source": "USGS NHD",                "Status": "⚠ Manual GIS download",       "Script": "fetch_category_g_extensions.py"},
            {"Factor": "National Forest Proximity",     "Category": "G", "Data Source": "USDA FS Boundaries",      "Status": "🟢 Auto-download available",  "Script": "fetch_category_g_extensions.py"},
            {"Factor": "Bridge Chokepoint Density",     "Category": "G", "Data Source": "FHWA NBI",                "Status": "🟢 Auto-download available",  "Script": "fetch_category_g_extensions.py"},
            {"Factor": "Agricultural Occupation Rate",  "Category": "E", "Data Source": "ACS C24010",              "Status": "🟢 Census API (key required)", "Script": "fetch_category_e_extensions.py"},
            {"Factor": "Ham Radio License Density",     "Category": "E", "Data Source": "FCC ULS",                 "Status": "🟢 Auto-download available",  "Script": "fetch_category_e_extensions.py"},
            {"Factor": "Physical Inactivity Rate (LPA)","Category": "H", "Data Source": "CDC PLACES 2023 (in repo)","Status": "✅ Ready — just not extracted","Script": "fetch_category_e_extensions.py"},
        ]
        st.dataframe(pd.DataFrame(fetch_status), use_container_width=True, hide_index=True)
        st.markdown("""
        <div style='font-size:.7rem; color:#3a3028; margin-top:6px;'>
        Run: python data/fetch/fetch_category_c_extensions.py --factor all<br>
        Run: python data/fetch/fetch_category_g_extensions.py --factor all<br>
        Run: python data/fetch/fetch_category_e_extensions.py --factor all
        </div>""", unsafe_allow_html=True)

else:
    # ── Idle state ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div style='
            border: 1px solid rgba(204,34,0,.3);
            background: rgba(204,34,0,.05);
            padding: 40px;
            text-align: center;
            margin-top: 40px;
        '>
            <div style='font-family:"Special Elite",cursive; font-size:2rem; color:#cc2200; letter-spacing:6px;'>☣</div>
            <div style='font-family:"Special Elite",cursive; font-size:1.1rem; color:#e8e0d0; letter-spacing:4px; margin-top:12px;'>
                AWAITING PARAMETERS
            </div>
            <div style='font-size:.75rem; color:#5a5040; letter-spacing:2px; margin-top:8px;'>
                Configure outbreak parameters above and press RUN PREDICTION
            </div>
        </div>
        """, unsafe_allow_html=True)
    with tab2:
        st.info("Run a prediction first to view the simulation.")
    with tab3:
        st.info("Run a prediction first to view SHAP analysis.")
    with tab4:
        st.info("Run a prediction first to view model comparison charts.")

