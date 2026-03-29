# SZR Outbreak Outcome Predictor

## Project Purpose
Surrogate ML model predicting zombie outbreak outcomes from SZR
epidemiological parameters and HSI county scores. Replaces running
the full ODE simulation at inference time with a fast neural network.

## Run Order
1. python data/generate_data.py      # generates szr_synthetic.csv
2. python model/train.py             # trains all ablation experiments
3. streamlit run app/streamlit_app.py

## Key Design Decisions
- beta is scaled by (1 - mobility_score): mobility_score represents movement
  restrictions (0 = free movement, 1 = lockdown), so higher score lowers
  effective transmission: effective_beta = beta * (1 - mobility_score)
- HSI scores (mobility, infrastructure, health) are the novel features vs.
  standard SZR params — Experiment E vs C ablation tests their importance
- StandardScaler fitted on train set only, saved to outputs/scaler.pkl
- Best model saved to outputs/best_model.pt (Experiment C architecture)

## Apple Silicon Note
Always use --platform linux/amd64 for any Docker builds

## Targets
- peak_zombie_fraction: regression (MSE loss)
- time_to_peak: regression (MSE loss)
- containment: binary classification (BCEWithLogitsLoss)

## Ablation Experiments
- A: Linear/Logistic baseline (scikit-learn)
- B: MLP hidden_dims=[32]
- C: MLP hidden_dims=[64, 128] — PRIMARY MODEL
- D: MLP hidden_dims=[128, 256, 128]
- E: MLP [64,128] WITHOUT HSI features — tests HSI importance
- F: MLP [64,128] WITH ONLY HSI features — tests HSI alone

## Common Debugging Notes
- If generate_data.py produces NaNs, check for division by zero when
  initial_population is very small relative to initial_infected
- If containment class balance is worse than 80/20, adjust the
  stabilization threshold in generate_data.py
- Scaler must be fit on training data only — never on val or test
