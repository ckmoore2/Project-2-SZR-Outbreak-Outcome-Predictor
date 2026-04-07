"""
Generate parity plot, SHAP importance plot, and confusion matrix for Experiment D model.
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "model"))

from szr_predictor import SZRPredictor, load_and_preprocess

DATA_PATH = os.path.join(_ROOT, "data", "szr_synthetic.csv")
OUTPUTS_DIR = _HERE

# ---------------------------------------------------------------------------
# Reproduce the exact train/test split from train.py
# ---------------------------------------------------------------------------
print("Loading data and reproducing train/test split...")
X, y_reg, y_cls = load_and_preprocess(DATA_PATH)
feature_cols = list(X.columns)

X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = (
    train_test_split(X, y_reg, y_cls, test_size=0.20, random_state=42)
)
X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = (
    train_test_split(X_temp, y_reg_temp, y_cls_temp, test_size=0.50, random_state=42)
)

scaler = joblib.load(os.path.join(OUTPUTS_DIR, "scaler.pkl"))
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Load best model (Experiment D)
# ---------------------------------------------------------------------------
print("Loading Experiment D model...")
model = SZRPredictor(input_dim=8, hidden_dims=[128, 256, 128], output_dim=3)
model.load_state_dict(
    torch.load(os.path.join(OUTPUTS_DIR, "best_model.pt"),
               map_location="cpu", weights_only=True)
)
model.eval()

# ---------------------------------------------------------------------------
# Parity plot — peak_zombie_fraction
# ---------------------------------------------------------------------------
print("Generating parity plot...")
with torch.no_grad():
    preds = model(torch.FloatTensor(X_test_sc)).numpy()

y_true_peak = y_reg_test["peak_zombie_fraction"].values
y_pred_peak = np.clip(preds[:, 0], 0.0, 1.0)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true_peak, y_pred_peak, alpha=0.3, s=10, color="steelblue",
           label="Test samples")
lims = [min(y_true_peak.min(), y_pred_peak.min()),
        max(y_true_peak.max(), y_pred_peak.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual peak zombie fraction")
ax.set_ylabel("Predicted peak zombie fraction")
ax.set_title("Parity Plot — Peak Zombie Fraction (Experiment D)")
ax.legend(fontsize=9)
fig.tight_layout()
parity_path = os.path.join(OUTPUTS_DIR, "parity_plot.png")
fig.savefig(parity_path, dpi=150)
plt.close(fig)
print(f"  Saved: {parity_path}")

# ---------------------------------------------------------------------------
# SHAP summary bar plot
# ---------------------------------------------------------------------------
print("Computing SHAP values (DeepExplainer)...")

# Use 200-sample background from training set; explain 200 test samples
rng = np.random.default_rng(0)
bg_idx = rng.choice(len(X_train_sc), size=200, replace=False)
background = torch.FloatTensor(X_train_sc[bg_idx])
X_explain = torch.FloatTensor(X_test_sc[:200])

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_explain)

# shap_values is shape (200, 8, 3) or list of 3 arrays of shape (200, 8)
# We want output index 0 = peak_zombie_fraction
if isinstance(shap_values, list):
    sv_peak = shap_values[0]          # (200, 8)
else:
    sv_peak = shap_values[:, :, 0]    # (200, 8, 3) → (200, 8)

print(f"  SHAP values shape: {np.array(sv_peak).shape}")

fig2, ax2 = plt.subplots(figsize=(8, 5))
mean_abs_shap = np.abs(sv_peak).mean(axis=0)
order = np.argsort(mean_abs_shap)[::-1]
ax2.barh([feature_cols[i] for i in order[::-1]],
         mean_abs_shap[order[::-1]],
         color="steelblue")
ax2.set_xlabel("Mean |SHAP value| (impact on peak zombie fraction)")
ax2.set_title("SHAP Feature Importance — Experiment D\n(peak_zombie_fraction)")
fig2.tight_layout()
shap_path = os.path.join(OUTPUTS_DIR, "shap_importance.png")
fig2.savefig(shap_path, dpi=150)
plt.close(fig2)
print(f"  Saved: {shap_path}")

# ---------------------------------------------------------------------------
# Confusion matrix — containment classifier
# ---------------------------------------------------------------------------
print("Generating confusion matrix...")

y_true_cls = y_cls_test.values
y_cls_prob = torch.sigmoid(torch.FloatTensor(preds[:, 2])).numpy()
y_cls_pred = (y_cls_prob >= 0.5).astype(int)

cm = confusion_matrix(y_true_cls, y_cls_pred)
precision = precision_score(y_true_cls, y_cls_pred, zero_division=0)
recall    = recall_score(y_true_cls, y_cls_pred, zero_division=0)
f1        = f1_score(y_true_cls, y_cls_pred, zero_division=0)

labels = ["Collapse", "Contained"]

fig3, ax3 = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=labels, yticklabels=labels,
    linewidths=0.5, linecolor="gray",
    ax=ax3,
)
ax3.set_xlabel("Predicted label")
ax3.set_ylabel("True label")
ax3.set_title(
    "Containment Classifier — Test Set Confusion Matrix\n"
    f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}",
    fontsize=10,
)
fig3.tight_layout()
cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
fig3.savefig(cm_path, dpi=150)
plt.close(fig3)
print(f"  Saved: {cm_path}")

print("\nDone.")
