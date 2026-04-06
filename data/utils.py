"""
data/utils.py
─────────────
Shared utility functions for the data pipeline scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series to [0, 1].

    Returns a constant 0.5 Series when all values are equal, avoiding
    division-by-zero while preserving a meaningful mid-range default.
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)
