"""
SZR Predictor model definition and data preprocessing utilities.
"""

import pandas as pd
import torch
import torch.nn as nn


__all__ = [
    "SZRPredictor",
    "FEATURE_COLUMNS",
    "TARGET_REGRESSION",
    "TARGET_CLASSIFICATION",
    "load_and_preprocess",
]

FEATURE_COLUMNS = [
    "beta",
    "zeta",
    "alpha",
    "initial_population",
    "initial_infected",
    "mobility_score",
    "infrastructure_score",
    "health_score",
]

TARGET_REGRESSION = ["peak_zombie_fraction", "time_to_peak"]
TARGET_CLASSIFICATION = ["containment"]


class SZRPredictor(nn.Module):
    """
    Configurable MLP for predicting SZR outbreak outcomes.

    Outputs raw logits/values for both regression targets and the
    binary classification target (containment). No activation on the
    final layer — callers should apply sigmoid for containment probability.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_and_preprocess(csv_path: str):
    """
    Load the synthetic SZR CSV and split into features and targets.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with the 8 input columns.
    y_regression : pd.DataFrame
        Regression targets: peak_zombie_fraction and time_to_peak.
    y_classification : pd.Series
        Binary classification target: containment (0 or 1).
    """
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y_regression = df[TARGET_REGRESSION]
    y_classification = df[TARGET_CLASSIFICATION[0]]
    return X, y_regression, y_classification
