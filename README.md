# SZR Outbreak Outcome Predictor

A surrogate machine-learning model that predicts zombie outbreak outcomes
(peak infection rate, time to peak, and containment probability) from SZR
epidemiological parameters and HSI county-level scores — without running the
full differential-equation simulation at inference time. A PyTorch MLP is
trained on 10,000 synthetic ODE scenarios so that a single forward pass
replaces an expensive numerical integration, enabling real-time what-if
analysis in the Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate synthetic training data (≈ 10 k ODE simulations)
python data/generate_data.py

# 2. Train all ablation experiments and save best model + scaler
python model/train.py

# 3. Launch the interactive Streamlit dashboard
streamlit run app/streamlit_app.py
```

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

## Transmission Scaling Note

`beta` (base transmission rate) is scaled by `(1 - mobility_score)` during
both data generation and inference. `mobility_score` represents the level of
movement restrictions in the county (0 = unrestricted movement, 1 = full
lockdown). A higher restriction score therefore reduces the effective
transmission rate: `effective_β = β × (1 − mobility_score)`.

