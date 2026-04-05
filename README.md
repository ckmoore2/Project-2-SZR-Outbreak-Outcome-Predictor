# SZR Outbreak Outcome Predictor

![CI](https://github.com/ckmoore2/Project-2-SZR-Outbreak-Outcome-Predictor/actions/workflows/ci.yml/badge.svg)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Cloud%20Run-blue)](https://szr-predictor-889200413982.us-east1.run.app)

A surrogate machine-learning model that predicts zombie outbreak outcomes
(peak infection rate, time to peak, and containment probability) from SZR
epidemiological parameters and HSI county-level scores — without running the
full differential-equation simulation at inference time. A PyTorch MLP is
trained on 10,000 synthetic ODE scenarios so that a single forward pass
replaces an expensive numerical integration, enabling real-time what-if
analysis in the Streamlit dashboard.

---

## Background

### The Original Paper — Munz et al. (2009)

The mathematical foundation for this project is the zombie epidemiology model
introduced in:

> **Munz, P., Hudea, I., Imad, J., & Smith?, R.J. (2009)**
> *When Zombies Attack!: Mathematical Modelling of an Outbreak of Zombie Infection*
> In: Infectious Disease Modelling Research Progress, Nova Science Publishers.

That paper was one of the first to apply standard compartmental epidemic
modelling to a fictional pathogen, using zombies as a pedagogical device to
make ODE dynamics accessible. It defined the **SZR** (Susceptible–Zombie–Removed)
system — a stripped-down variant of SIR where infected individuals do not
recover to immunity but instead become contagious until neutralised:

```
dS/dt = −β · S · Z
dZ/dt =  β · S · Z − ζ · Z
dR/dt =  ζ · Z
```

where β is the bite transmission rate and ζ is the zombie removal rate
(neutralisation by humans). The paper showed that without a sufficiently large
removal rate relative to transmission (ζ/β < 1), human extinction is the only
equilibrium — making it a useful teaching model for epidemic threshold theory.

The paper also extended the basic model with latency (SIZR), quarantine (SIZRQ),
and treatment (SIZRT) compartments, and demonstrated that only rapid, aggressive
intervention prevents collapse — a result that maps directly onto real outbreak
response doctrine.

### Project - HSI Scoring System

This project is an extension  that
built a **Human Survival Index (HSI)** scoring framework for all 100 North
Carolina counties. That project's contributions were:

- **Operationalised the SZR model for real geography** — parameterised β and κ
  using zombie fiction canon (The Walking Dead, The Last of Us, World War Z,
  28 Days Later, Rabies) as calibration anchors across a plausible outbreak
  severity range
- **Defined six HSI categories** scoring each county's structural resilience:
  H (Health & Fitness), M (Mobility & Escape), I (Infrastructure),
  E (Education & Awareness), C (Social & Community), G (Geographic & Environment)
- **Sourced and scored real census data** per category per county from ACS,
  CDC PLACES, FEMA, and state agency datasets — one section per team member
- **Applied a linear κ modifier** — higher HSI linearly increased zombie removal
  rate: `κ_eff = κ_base × (1 + 0.5 × HSI)`
- **Used uniform β** across all tracts and counties regardless of population
  density or mobility patterns
- **Ran the ODE directly at query time** — every what-if question required a
  full numerical integration, with no caching or approximation layer

The v1 model produced plausible county rankings but had two structural
limitations: the linear HSI modifier assumed proportional benefit with no
threshold effects, and querying was bottlenecked by ODE runtime whenever
large numbers of scenarios needed evaluation.

### This Project — What's Different

Project 2 keeps the SZR framework and NC county data from the HSI Scoring System and extends
it in three directions:

**1. Surrogate ML model replaces the ODE at inference time**

A PyTorch MLP (Experiment C: hidden dims [64, 128]) is trained on 10,000 ODE
simulations sampled across the full parameter space. At query time a single
forward pass (~0.1 ms) replaces a numerical integration (~50–200 ms), enabling
real-time interactive what-if analysis. The ablation suite (Experiments A–F)
quantifies exactly how much predictive power comes from the HSI features vs the
core SZR parameters, and how model complexity affects accuracy.

**2. Improved ODE mechanics (v2)**

Two weaknesses in the v1 model are corrected in the `code/` module:

- **Sigmoid κ modifier** — replaces the linear formula with a sigmoid centred
  at HSI = 0.5. This captures the real-world threshold effect where communities
  above a coordination tipping point become *dramatically* more effective, not
  just marginally better. Below the threshold, disorganised communities perform
  worse than the linear model predicted.
- **Tract-level β variation** — β is no longer uniform. Dense tracts get up to
  +40% on β (more S·Z encounters); highly mobile tracts get up to −25% (people
  flee encounters). This gives the Mobility category direct influence on spread
  rate, not just on removal rate.

**3. HSI weight sensitivity analysis**

HSI assigned category weights (H=0.15, M=0.15, I=0.20, E=0.10, C=0.25,
G=0.15) by expert judgement. Project 2 adds a systematic sweep (`code/sensitivity_analysis.py`)
that tests all pairwise weight combinations across three zombie scenarios and
maps how NC-wide survival fraction responds to weight choices — providing
empirical justification for the v2 weight adjustments and identifying which
categories have the largest marginal impact on outcomes.

| Aspect |HSI (v1) | Project 2 (v2) |
|--------|---------------|----------------|
| Inference | ODE at query time | Surrogate MLP (forward pass) |
| κ modifier | Linear: `1 + 0.5·HSI` | Sigmoid: threshold at HSI=0.5 |
| β | Uniform across tracts | Varies by density (+40%) and mobility (−25%) |
| HSI weights | Expert-assigned, fixed | Sensitivity-tested; v2 adjustments justified |
| New sub-factors | — | 9 new factors in development (C, G, E, H) |
| Deployment | — | Docker + Cloud Run (live demo) |
| Explainability | — | SHAP feature importance per output |

---

## Quick Start — Local Development (no Docker)

This is the fastest way to run the app on your own machine.

**1. Install dependencies**
```bash
pip install -r requirements.txt
```
Python 3.10+ recommended. A virtual environment is strongly advised:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Generate training data** *(one-time, ~2–5 min)*
```bash
python data/generate_data.py
```
Produces `data/szr_synthetic.csv` (10,000 ODE simulations).
Set `N_SCENARIOS=2000` for a faster first run:
```bash
N_SCENARIOS=2000 python data/generate_data.py
```

**3. Train the model** *(one-time, ~5–10 min)*
```bash
python model/train.py
```
Runs ablation experiments A–F and saves `outputs/best_model.pt` and
`outputs/scaler.pkl`. Both files are required before launching the app.

**4. Launch the dashboard**
```bash
streamlit run app/streamlit_app.py
```
Then open **http://localhost:8501** in your browser.

---

## Generated Files

These files are gitignored and must exist locally before running the app or
building a Docker image. Steps 2 and 3 above produce them.

| Script | Output | Notes |
|--------|--------|-------|
| `python data/generate_data.py` | `data/szr_synthetic.csv` | 10 k ODE simulations |
| `python model/train.py` | `outputs/best_model.pt`<br>`outputs/scaler.pkl` | Trained model + fitted scaler; both required by the app |
| `python data/synthesize_missing_hsi.py` | `data/dasc6010/nc_education.csv`<br>`data/dasc6010/nc_social.csv`<br>`data/dasc6010/nc_geographic.csv` | Fetches ACS data from Census API (requires internet); computes HSI scores E, C, G for all 100 NC counties — optional for the app, used by `data/nc_county_profiles.py` |

## Ablation Experiments

| ID | Description | Purpose |
|----|-------------|---------|
| A  | LinearRegression + LogisticRegression (scikit-learn) | Baseline |
| B  | MLP hidden_dims=[32] | Minimal neural net |
| C  | MLP hidden_dims=[64, 128] — **PRIMARY MODEL** | Best trade-off |
| D  | MLP hidden_dims=[128, 256, 128] | Deeper network |
| E  | MLP [64,128] **without** HSI features | Tests HSI importance |
| F  | MLP [64,128] with **only** HSI features | Tests HSI alone |

Experiments E and F isolate the contribution of the three HSI county scores
(mobility, infrastructure, health) relative to the core SZR parameters.

## Model Architecture (v2)

### SZR ODE

The simulation uses a frequency-dependent SZR system:

```
dS/dt = −β_eff · S · Z / N
dZ/dt =  β_eff · S · Z / N − (κ_eff + α) · Z
dR/dt =  (κ_eff + α) · Z
```

### Transmission Scaling — β

`beta` is scaled at two levels:

**Mobility modifier (training data & app simulation):**
```
effective_β = β × (1 − mobility_score)
```
`mobility_score` represents movement restrictions (0 = free movement, 1 = full
lockdown). A higher score reduces effective transmission. This modifier is
applied during data generation *and* in the app's ODE simulation so ground
truth always matches the model's training distribution.

**Tract-level β variation (v2, `code/szr_model.py`):**
```
β_eff(tract) = β_base × density_modifier × encounter_modifier
             = β_base × (1 + 0.40 · norm_density) × (1 − 0.25 · mobility_score)
```
Dense tracts get up to +40% on β; highly mobile tracts get up to −25%.

### Removal Scaling — κ (sigmoid modifier, v2)

v1 assumed killing effectiveness scales linearly with HSI:
`κ_eff = κ_base × (1 + 0.5 · HSI)` — range [1.0, 1.5] × κ_base.

v2 uses a sigmoid centred at HSI = 0.5:
```
κ_eff = κ_base × (1 + scale × (σ(steepness · (HSI − 0.5)) − 0.5))
```
- HSI = 0.0 → κ_eff ≈ 0.73 × κ_base (disorganised — kills poorly)
- HSI = 0.5 → κ_eff = 1.00 × κ_base (neutral)
- HSI = 1.0 → κ_eff ≈ 1.73 × κ_base (highly organised — much more effective)

The sigmoid captures the real-world threshold where organised communities
become dramatically more effective, not just marginally better.

Both modifiers are togglable in the app via the **Model Version** panel.

## HSI Categories & Weights

| Category | Label | v1 Weight | v2 Weight | Owner |
|----------|-------|-----------|-----------|-------|
| H | Health & Fitness | 0.15 | 0.15 | Curtis |
| M | Mobility & Escape | 0.15 | 0.15 | Curtis |
| I | Infrastructure | 0.20 | 0.20 | Curtis |
| E | Education & Awareness | 0.10 | 0.13 | Lennen |
| C | Social & Community | 0.25 | 0.22 | Kiana |
| G | Geographic & Environment | 0.15 | 0.15 | Rebecca |

v2 weights shift 0.03 from C to E, reflecting that new occupation and
agriculture sub-factors make the E section more robust.

## New HSI Sub-Factors

New sub-factors are registered in `code/config.py` and fetched by scripts in
`data/fetch/`. Each entry documents source, direction, and rationale.

| Factor | Category |  | Source | Status |
|--------|----------|  |--------|--------|
| Hunting License Density | C | | NCWRC Annual Report | Manual download needed |
| Congregation Density | C |  | USDA ERS Rural Atlas | Partial (proxy available) |
| Volunteer Fire Dept Coverage | C |  | USFA NFIRS | Fetch script ready |
| Waterway Barrier Score | G |  | USGS NHD | Manual GIS download |
| National Forest Proximity | G |  | USDA Forest Service | Auto-download available |
| Bridge Chokepoint Density | G |  | FHWA NBI | Auto-download available |
| Agricultural Occupation Rate | E |  | ACS C24010 | Census API (key required) |
| Ham Radio License Density | E |  | FCC ULS | Auto-download available |
| Physical Inactivity Rate (LPA) | H |  | CDC PLACES 2023 | Ready — not yet extracted |

Fetch scripts:
```bash
python data/fetch/fetch_category_c_extensions.py --factor all
python data/fetch/fetch_category_g_extensions.py --factor all
python data/fetch/fetch_category_e_extensions.py --factor all
```

## Sensitivity Analysis

`code/sensitivity_analysis.py` sweeps pairs of HSI category weights while
holding the remaining four fixed at v1 defaults. For each combination it
runs the SZR ODE on a 500-tract synthetic NC distribution and records
population-weighted survival fraction, containment rate, and median α_eff.

```bash
# Must be run from the code/ directory — sensitivity_analysis.py imports
# config and szr_model using relative imports that expect that working directory.
cd code
python sensitivity_analysis.py
```

Outputs to `outputs/sensitivity/` (relative to the project root):
- `<pair>_<scenario>.csv` — raw sweep results per pair × scenario
- `summary.csv` — best/worst/v1 survival across all sweeps
- `heatmaps.png` — heatmap grid (colour = survival fraction)

Configured sweep pairs: C×H, C×G, M×I, E×C across scenarios: 28 Days Later,
The Last of Us, World War Z.

## Code Modules (`code/`)

| File | Purpose |
|------|---------|
| `code/config.py` | Central config: scenario parameters, HSI weights (v1/v2), sigmoid/linear toggle, tract β weights, simulation settings, new sub-factor registry |
| `code/szr_model.py` | SZR ODE system with sigmoid κ modifier, tract-level β variation, tract/county simulation runners, v1 backwards-compatible wrapper |
| `code/sensitivity_analysis.py` | HSI weight sensitivity sweep with heatmap visualisation |

## App Features

The Streamlit app (`app/streamlit_app.py`) provides:

- **Location panel** — load real NC county population and HSI estimates
- **Zombie show scenarios** — canonical β/ζ/α from The Walking Dead, The Last of Us, World War Z, 28 Days Later, Rabies
- **Severity presets** — operational levels from Early Detection to Total Collapse
- **Fine-tune sliders** — manual control over all 8 model inputs
- **Model version toggles** — switch between v1 (linear κ, uniform β) and v2 (sigmoid κ, tract-level β) in real time
- **Tab 1 — Prediction** — neural network outputs vs ODE ground truth
- **Tab 2 — ODE Simulation** — full SZR trajectory plot + 4-county comparison table
- **Tab 3 — SHAP Analysis** — KernelExplainer feature importance for peak fraction and containment
- **Tab 4 — Model v1 vs v2** — κ modifier comparison chart, β tract variation chart, new sub-factor fetch status table

## Deployment

There are three ways to run the project via Docker, depending on your goal:

| Goal | Command | Platform |
|------|---------|----------|
| Run locally in Docker (fastest) | `bash deploy.sh --local` | Native (no emulation) |
| Deploy to Cloud Run | `bash deploy.sh` | Always `linux/amd64` |
| Manual Docker build | `docker build -t szr-predictor .` | Native |

### Before any Docker build

The model artifacts are not generated inside the Docker image — they are
copied in from your local `outputs/` directory. Run the training pipeline
first if you haven't already:

```bash
python data/generate_data.py
python model/train.py
```

This produces `outputs/best_model.pt` and `outputs/scaler.pkl`. Both files
must exist before running `docker build`. The Dockerfile does handle
`data/generate_data.py` and `data/synthesize_missing_hsi.py` itself (as `RUN`
steps), but `model/train.py` is not in the Dockerfile — the pre-trained
artifacts are baked in via `COPY`.

### Run locally in Docker

Builds using your machine's native architecture — no emulation, fastest
possible build. Does not require gcloud.

```bash
bash deploy.sh --local
docker run -p 8501:8080 szr-predictor-local
```

Then open **http://localhost:8501**.

### Deploy to Cloud Run

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) and
[gcloud CLI](https://cloud.google.com/sdk/docs/install) authenticated:

```bash
gcloud auth login
gcloud auth configure-docker
```

Then deploy:

```bash
bash deploy.sh
```

`deploy.sh` detects your host architecture and tells you whether it is
building natively or cross-compiling:

- **Intel/amd64 host** → native build, no emulation overhead
- **Apple Silicon / arm64 host** → cross-compile to `linux/amd64` via Docker
  buildx (required for Cloud Run — expect a slower build)

### Live URL

https://szr-predictor-889200413982.us-east1.run.app
