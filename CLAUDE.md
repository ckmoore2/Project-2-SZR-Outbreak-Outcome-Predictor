# SZR Outbreak Outcome Predictor — Project Notes

## Run Order
1. `python data/rebuild_county_data.py`   — if `data/processed/` CSVs are missing
2. `python data/integrate_real_hsi.py`   — produces `hsi_distributions.json`
3. `python data/generate_data.py`        — produces `szr_synthetic.csv`
4. `python model/train.py`               — produces `best_model.pt` + `scaler.pkl`
5. `streamlit run app/streamlit_app.py`

## Key Design Constraints
- `FEATURE_COLUMNS` in `model/szr_predictor.py` is the single source of truth for
  feature order — `app/streamlit_app.py` imports it directly so they can never drift
- `self.network` is the Sequential attribute name in `SZRPredictor` — must match
  the saved state dict keys in `outputs/best_model.pt`
- HSI weights are defined in `model/config.py` as `HSI_WEIGHTS_V1` / `HSI_WEIGHTS_NAMED`
  and imported by all data scripts — never duplicate them
- `beta` is scaled by `(1 - mobility_score)`: mobility represents movement restrictions
  (0 = free movement, 1 = lockdown), so higher score lowers effective transmission
- Scaler must be fit on training data only — never on val or test

## Deployment
- GCP Project: `szr-outbreak-outcome-predictor` (number: 889200413982)
- Service account: `szr-deployer@szr-outbreak-outcome-predictor.iam.gserviceaccount.com`
- Cloud Run region: `us-east1`
- Mac Apple Silicon: always use `--platform linux/amd64` for Docker builds targeting Cloud Run
  (`bash deploy.sh` handles this automatically; use `bash deploy.sh --local` for native builds)

## Model Architecture (Experiment D — PRIMARY)
- `SZRPredictor(input_dim=8, hidden_dims=[128, 256, 128], output_dim=3, dropout=0.2)`
- `pos_weight=5.5` for `BCEWithLogitsLoss` on the containment target
- `StandardScaler` fitted on training features only, saved to `outputs/scaler.pkl`
- Output indices: `[0]` = peak_zombie_fraction, `[1]` = time_to_peak, `[2]` = containment logit

## Ablation Experiments
- A: Linear/Logistic baseline (scikit-learn)
- B: MLP `hidden_dims=[32]`
- C: MLP `hidden_dims=[64, 128]`
- D: MLP `hidden_dims=[128, 256, 128]` — PRIMARY MODEL (R²=0.937, MAE=0.928)
- E: MLP `[64, 128]` WITHOUT HSI features — tests HSI importance
- F: MLP `[64, 128]` WITH ONLY HSI features — tests HSI alone

## Common Debugging Notes
- If `generate_data.py` produces NaNs: check for division by zero when
  `initial_population` is very small relative to `initial_infected`
- If containment class balance is worse than 80/20: adjust the
  `CONTAINMENT_THRESHOLD` in `generate_data.py`
- If model load fails with key mismatch: the saved weights use `network.*` keys —
  confirm `SZRPredictor` assigns its `nn.Sequential` to `self.network`
- `model/szr_model.py` and `model/sensitivity_analysis.py` both insert `model/` into
  `sys.path` at import time so bare `from config import ...` resolves correctly
