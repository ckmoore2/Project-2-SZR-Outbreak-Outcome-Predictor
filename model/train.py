"""
Train ablation experiments (A-F) for the SZR Outbreak Outcome Predictor.

Run order:
  1. python data/generate_data.py
  2. python model/train.py
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Allow importing from model/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))
from szr_predictor import (
    FEATURE_COLUMNS,
    SZRPredictor,
    load_and_preprocess,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_PATH = os.path.join(_ROOT, "data", "szr_synthetic.csv")
OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def prepare_splits(csv_path=DATA_PATH):
    X, y_reg, y_cls = load_and_preprocess(csv_path)

    # 80 / 10 / 10 split
    X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = (
        train_test_split(X, y_reg, y_cls, test_size=0.20, random_state=42)
    )
    X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = (
        train_test_split(X_temp, y_reg_temp, y_cls_temp,
                         test_size=0.50, random_state=42)
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "scaler.pkl"))
    print(f"Scaler saved to {OUTPUTS_DIR}/scaler.pkl")

    return (X_train_sc, X_val_sc, X_test_sc,
            y_reg_train, y_reg_val, y_reg_test,
            y_cls_train, y_cls_val, y_cls_test,
            scaler, list(X.columns))


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return f1, auc


# ---------------------------------------------------------------------------
# Experiment A: scikit-learn baseline
# ---------------------------------------------------------------------------

def run_experiment_a(X_train, X_test, y_reg_train, y_reg_test,
                     y_cls_train, y_cls_test):
    print("\n--- Experiment A: Linear/Logistic baseline ---")

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_reg_train)
    y_reg_pred = lin_reg.predict(X_test)

    mae, rmse, r2 = regression_metrics(y_reg_test, y_reg_pred)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_cls_train)
    y_cls_prob = log_reg.predict_proba(X_test)[:, 1]

    f1, auc = classification_metrics(y_cls_test.values, y_cls_prob)
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return {"experiment": "A", "description": "Linear/Logistic baseline",
            "MAE": mae, "RMSE": rmse, "R2": r2, "F1": f1, "AUC": auc}


# ---------------------------------------------------------------------------
# PyTorch training loop
# ---------------------------------------------------------------------------

def make_loaders(X_train, X_val, y_reg_train, y_reg_val,
                 y_cls_train, y_cls_val, batch_size=256):
    def to_tensors(X, y_reg, y_cls):
        return (
            torch.FloatTensor(X),
            torch.FloatTensor(y_reg.values),
            torch.FloatTensor(y_cls.values).unsqueeze(1),
        )

    tx, ty_reg, ty_cls = to_tensors(X_train, y_reg_train, y_cls_train)
    vx, vy_reg, vy_cls = to_tensors(X_val, y_reg_val, y_cls_val)

    train_loader = DataLoader(
        TensorDataset(tx, ty_reg, ty_cls), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(vx, vy_reg, vy_cls), batch_size=batch_size
    )
    return train_loader, val_loader


def train_pytorch_model(model, train_loader, val_loader,
                        epochs=100, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for X_b, y_reg_b, y_cls_b in train_loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = mse_loss(out[:, :2], y_reg_b) + bce_loss(out[:, 2:3], y_cls_b)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_b.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_b, y_reg_b, y_cls_b in val_loader:
                out = model(X_b)
                loss = mse_loss(out[:, :2], y_reg_b) + bce_loss(out[:, 2:3], y_cls_b)
                epoch_val_loss += loss.item() * X_b.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses


def evaluate_pytorch_model(model, X_test, y_reg_test, y_cls_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        out = model(X_t).numpy()

    y_reg_pred = out[:, :2]
    y_cls_prob = torch.sigmoid(torch.FloatTensor(out[:, 2])).numpy()

    mae, rmse, r2 = regression_metrics(y_reg_test.values, y_reg_pred)
    f1, auc = classification_metrics(y_cls_test.values, y_cls_prob)
    return mae, rmse, r2, f1, auc


def run_pytorch_experiment(label, description, hidden_dims,
                           X_train, X_val, X_test,
                           y_reg_train, y_reg_val, y_reg_test,
                           y_cls_train, y_cls_val, y_cls_test,
                           input_dim=None):
    print(f"\n--- Experiment {label}: {description} ---")
    if input_dim is None:
        input_dim = X_train.shape[1]

    model = SZRPredictor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=3,
        dropout=0.2,
    )

    train_loader, val_loader = make_loaders(
        X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val
    )
    train_losses, val_losses = train_pytorch_model(
        model, train_loader, val_loader
    )

    mae, rmse, r2, f1, auc = evaluate_pytorch_model(
        model, X_test, y_reg_test, y_cls_test
    )
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return (
        {"experiment": label, "description": description,
         "MAE": mae, "RMSE": rmse, "R2": r2, "F1": f1, "AUC": auc},
        model,
        train_losses,
        val_losses,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading and preprocessing data...")
    (X_train, X_val, X_test,
     y_reg_train, y_reg_val, y_reg_test,
     y_cls_train, y_cls_val, y_cls_test,
     scaler, feature_cols) = prepare_splits()

    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  "
          f"Test: {X_test.shape[0]}")

    results = []

    # ------------------------------------------------------------------
    # Experiment A
    # ------------------------------------------------------------------
    res_a = run_experiment_a(
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test
    )
    results.append(res_a)

    # ------------------------------------------------------------------
    # Experiments B-D (full feature set)
    # ------------------------------------------------------------------
    configs = [
        ("B", "MLP hidden=[32]", [32]),
        ("C", "MLP hidden=[64,128]", [64, 128]),
        ("D", "MLP hidden=[128,256,128]", [128, 256, 128]),
    ]

    model_c = None
    losses_c = None

    for label, desc, hdims in configs:
        res, model, tl, vl = run_pytorch_experiment(
            label, desc, hdims,
            X_train, X_val, X_test,
            y_reg_train, y_reg_val, y_reg_test,
            y_cls_train, y_cls_val, y_cls_test,
        )
        results.append(res)
        if label == "C":
            model_c = model
            losses_c = (tl, vl)

    # ------------------------------------------------------------------
    # Experiment E: drop HSI features
    # ------------------------------------------------------------------
    hsi_cols = ["mobility_score", "infrastructure_score", "health_score"]
    non_hsi_idx = [feature_cols.index(c)
                   for c in feature_cols if c not in hsi_cols]

    X_train_e = X_train[:, non_hsi_idx]
    X_val_e = X_val[:, non_hsi_idx]
    X_test_e = X_test[:, non_hsi_idx]

    res_e, _, _, _ = run_pytorch_experiment(
        "E", "MLP [64,128] WITHOUT HSI features", [64, 128],
        X_train_e, X_val_e, X_test_e,
        y_reg_train, y_reg_val, y_reg_test,
        y_cls_train, y_cls_val, y_cls_test,
        input_dim=len(non_hsi_idx),
    )
    results.append(res_e)

    # ------------------------------------------------------------------
    # Experiment F: ONLY HSI features
    # ------------------------------------------------------------------
    hsi_idx = [feature_cols.index(c) for c in hsi_cols]
    X_train_f = X_train[:, hsi_idx]
    X_val_f = X_val[:, hsi_idx]
    X_test_f = X_test[:, hsi_idx]

    res_f, _, _, _ = run_pytorch_experiment(
        "F", "MLP [64,128] ONLY HSI features", [64, 128],
        X_train_f, X_val_f, X_test_f,
        y_reg_train, y_reg_val, y_reg_test,
        y_cls_train, y_cls_val, y_cls_test,
        input_dim=len(hsi_idx),
    )
    results.append(res_f)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    # Best model (Experiment C)
    model_path = os.path.join(OUTPUTS_DIR, "best_model.pt")
    torch.save(model_c.state_dict(), model_path)
    print(f"\nBest model (Exp C) saved to {model_path}")

    # Ablation results table
    df_results = pd.DataFrame(results)
    ablation_path = os.path.join(OUTPUTS_DIR, "ablation_results.csv")
    df_results.to_csv(ablation_path, index=False)
    print(f"Ablation results saved to {ablation_path}")
    print("\n" + df_results.to_string(index=False))

    # Loss curve for Experiment C
    train_losses, val_losses = losses_c
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train loss")
    ax.plot(val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Experiment C — Training Loss Curve")
    ax.legend()
    fig.tight_layout()
    curve_path = os.path.join(OUTPUTS_DIR, "loss_curve.png")
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)
    print(f"Loss curve saved to {curve_path}")


if __name__ == "__main__":
    main()
