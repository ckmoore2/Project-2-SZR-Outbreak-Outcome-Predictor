"""
szr_model.py — SZR ODE model with v2 improvements.

Changes from v1:
  - sigmoid_kappa_modifier() replaces the linear (1 + 0.5 × HSI) formula
  - tract_beta() computes per-tract β based on population density and mobility
  - run_tract_simulation() runs the full ODE for a single census tract
  - run_county_simulation() aggregates tract results to county level
  - All v1 functions preserved for backwards compatibility
"""

import sys
import os
# Ensure model/ directory is on the path so bare `from config import ...` works
# regardless of whether this file is run directly or imported as model.szr_model.
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.integrate import odeint
from scipy.special import expit   # numerically stable sigmoid
from typing import Optional
import pandas as pd

from config import (
    KAPPA_MODIFIER,
    SIGMOID_SCALE,
    SIGMOID_STEEPNESS,
    BETA_VARY_BY_TRACT,
    BETA_DENSITY_WEIGHT,
    BETA_MOBILITY_WEIGHT,
    SIM_DAYS,
    SIM_STEPS_PER_DAY,
    CONTAINMENT_THRESHOLD,
)


# ══════════════════════════════════════════════════════════════════════════════
#  κ MODIFIER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def linear_kappa_modifier(hsi: float) -> float:
    """
    V1 modifier: κ_eff = κ_base × (1 + 0.5 × HSI)
    Range: [1.0, 1.5] × κ_base
    Preserved for backwards compatibility and ablation comparison.
    """
    return 1.0 + 0.5 * np.clip(hsi, 0.0, 1.0)


def sigmoid_kappa_modifier(
    hsi: float,
    scale: float = SIGMOID_SCALE,
    steepness: float = SIGMOID_STEEPNESS,
) -> float:
    """
    V2 modifier: κ_eff = κ_base × sigmoid_modifier(HSI)

    Uses a scaled sigmoid centred at HSI=0.5 so that:
      - HSI = 0.0  →  modifier ≈ 1 - scale/2  (poorly organised, kills few)
      - HSI = 0.5  →  modifier = 1.0           (neutral, same as base)
      - HSI = 1.0  →  modifier ≈ 1 + scale/2  (highly organised, kills many)

    The sigmoid inflection captures the real-world threshold effect where
    organised communities become dramatically more effective — not just
    marginally better — once they cross a coordination tipping point.

    Parameters
    ----------
    hsi        : float in [0, 1]
    scale      : total multiplicative range (default 1.5 → [0.25, 1.75] approx)
    steepness  : controls sharpness of the inflection (default 6)
    """
    hsi = float(np.clip(hsi, 0.0, 1.0))
    # expit(x) = 1 / (1 + e^-x)  — numerically stable
    raw = expit(steepness * (hsi - 0.5))       # range: ~(0, 1)
    # rescale from (0,1) to (1 - scale/2, 1 + scale/2)
    return 1.0 + scale * (raw - 0.5)


def kappa_modifier(hsi: float) -> float:
    """Dispatch to the configured modifier (set in config.py)."""
    if KAPPA_MODIFIER == "sigmoid":
        return sigmoid_kappa_modifier(hsi)
    return linear_kappa_modifier(hsi)


def effective_kappa(kappa_base: float, hsi: float) -> float:
    """κ_eff(tract) = κ_base × modifier(HSI)"""
    return kappa_base * kappa_modifier(hsi)


# ══════════════════════════════════════════════════════════════════════════════
#  β VARIATION BY TRACT
# ══════════════════════════════════════════════════════════════════════════════

def tract_beta(
    beta_base: float,
    norm_pop_density: float,
    mobility_score: float,
    density_weight: float = BETA_DENSITY_WEIGHT,
    mobility_weight: float = BETA_MOBILITY_WEIGHT,
) -> float:
    """
    Compute per-tract effective transmission rate.

    β_eff(tract) = β_base × density_modifier × encounter_modifier

    density_modifier  = 1 + density_weight × norm_pop_density
        High density → more S·Z encounters → higher effective β.
        Densest NC tracts (downtown Raleigh, Charlotte uptown) get up to
        +40% on β. Rural tracts near baseline.

    encounter_modifier = 1 - mobility_weight × mobility_score
        High mobility → people flee encounters → lower effective β.
        Most mobile tracts get up to -25% on β.
        This gives the M section a direct influence on spread dynamics,
        not just on κ.

    Parameters
    ----------
    beta_base        : scenario base transmission rate
    norm_pop_density : population density normalised to [0, 1] across NC tracts
    mobility_score   : HSI-M score for this tract, in [0, 1]
    """
    norm_pop_density = float(np.clip(norm_pop_density, 0.0, 1.0))
    mobility_score   = float(np.clip(mobility_score,   0.0, 1.0))

    density_modifier  = 1.0 + density_weight * norm_pop_density
    encounter_modifier = 1.0 - mobility_weight * mobility_score

    return beta_base * density_modifier * encounter_modifier


# ══════════════════════════════════════════════════════════════════════════════
#  ODE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def szr_ode(y, t, beta_eff, kappa_eff, alpha):
    """
    SZR ODE right-hand side.

    dS/dt = -β_eff · S · Z / N
    dZ/dt =  β_eff · S · Z / N  - (κ_eff + α) · Z
    dR/dt =  (κ_eff + α) · Z

    Parameters
    ----------
    y          : [S, Z, R]
    t          : time (days)
    beta_eff   : effective transmission rate for this tract
    kappa_eff  : effective removal rate for this tract (κ_base × modifier)
    alpha      : natural death / background removal rate
    """
    S, Z, R = y
    N = S + Z + R
    if N <= 0 or Z <= 0:
        return [0.0, 0.0, 0.0]
    dS = -beta_eff * S * Z / N
    dZ =  beta_eff * S * Z / N - (kappa_eff + alpha) * Z
    dR =  (kappa_eff + alpha) * Z
    return [dS, dZ, dR]


# ══════════════════════════════════════════════════════════════════════════════
#  TRACT-LEVEL SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_tract_simulation(
    population: float,
    initial_infected: float,
    beta_base: float,
    kappa_base: float,
    alpha: float,
    hsi: float,
    norm_pop_density: float = 0.5,
    mobility_score: float = 0.5,
    days: int = SIM_DAYS,
) -> dict:
    """
    Run the SZR ODE for a single census tract using v2 parameters.

    Returns a dict with:
      t               : time array
      S, Z, R         : population arrays
      peak_frac       : max(Z) / N
      time_to_peak    : day of peak zombie population
      survival_frac   : S[-1] / N (fraction of population surviving)
      contained       : bool — S[-1]/N > CONTAINMENT_THRESHOLD
      beta_eff        : effective β used
      kappa_eff       : effective κ used
      alpha_eff       : κ_eff / β_eff  (governing survival parameter)
    """
    population = max(float(population), 1.0)
    initial_infected = min(float(initial_infected), population - 1)

    # Compute tract-specific rates
    if BETA_VARY_BY_TRACT:
        beta_eff = tract_beta(beta_base, norm_pop_density, mobility_score)
    else:
        beta_eff = beta_base

    kappa_eff = effective_kappa(kappa_base, hsi)
    alpha_eff = kappa_eff / beta_eff if beta_eff > 0 else 0.0

    # Initial conditions
    S0 = population - initial_infected
    Z0 = initial_infected
    R0 = 0.0
    t  = np.linspace(0, days, days * SIM_STEPS_PER_DAY)

    sol = odeint(szr_ode, [S0, Z0, R0], t, args=(beta_eff, kappa_eff, alpha))
    S, Z, R = sol[:, 0], sol[:, 1], sol[:, 2]

    peak_idx     = int(np.argmax(Z))
    peak_frac    = float(np.max(Z) / population)
    time_to_peak = float(t[peak_idx])
    survival_frac = float(S[-1] / population)
    contained    = survival_frac > CONTAINMENT_THRESHOLD

    return {
        "t": t, "S": S, "Z": Z, "R": R,
        "peak_frac": peak_frac,
        "time_to_peak": time_to_peak,
        "survival_frac": survival_frac,
        "contained": contained,
        "beta_eff": beta_eff,
        "kappa_eff": kappa_eff,
        "alpha_eff": alpha_eff,
        "hsi": hsi,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  COUNTY / STATE AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

def run_county_simulation(
    tract_df: pd.DataFrame,
    beta_base: float,
    kappa_base: float,
    alpha: float,
    initial_infected_per_tract: int = 1,
) -> pd.DataFrame:
    """
    Run tract-level simulations for all tracts in a county/state DataFrame
    and return a results DataFrame.

    Expected tract_df columns:
      GEOID, population, hsi, norm_pop_density, score_M (mobility)

    Returns tract_df with added result columns:
      peak_frac, time_to_peak, survival_frac, contained,
      beta_eff, kappa_eff, alpha_eff
    """
    required = {"GEOID", "population", "hsi", "norm_pop_density", "score_M"}
    missing  = required - set(tract_df.columns)
    if missing:
        raise ValueError(f"tract_df missing columns: {missing}")

    results = []
    for _, row in tract_df.iterrows():
        res = run_tract_simulation(
            population=row["population"],
            initial_infected=initial_infected_per_tract,
            beta_base=beta_base,
            kappa_base=kappa_base,
            alpha=alpha,
            hsi=row["hsi"],
            norm_pop_density=row["norm_pop_density"],
            mobility_score=row["score_M"],
        )
        results.append({
            "GEOID":         row["GEOID"],
            "peak_frac":     res["peak_frac"],
            "time_to_peak":  res["time_to_peak"],
            "survival_frac": res["survival_frac"],
            "contained":     res["contained"],
            "beta_eff":      res["beta_eff"],
            "kappa_eff":     res["kappa_eff"],
            "alpha_eff":     res["alpha_eff"],
        })

    return tract_df.merge(pd.DataFrame(results), on="GEOID", how="left")


# ══════════════════════════════════════════════════════════════════════════════
#  V1 BACKWARDS-COMPATIBLE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation_v1(beta, zeta, alpha, population, initial_infected, days=180):
    """
    Original v1 interface — uniform β, linear κ modifier, no HSI.
    Preserved for comparison and Streamlit slider mode.
    """
    S0 = population - initial_infected
    Z0 = initial_infected
    R0 = 0.0
    t  = np.linspace(0, days, days * SIM_STEPS_PER_DAY)
    sol = odeint(szr_ode, [S0, Z0, R0], t, args=(beta, zeta, alpha))
    return t, sol


# ══════════════════════════════════════════════════════════════════════════════
#  MODIFIER COMPARISON UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def compare_modifiers(hsi_values: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Return a DataFrame comparing linear vs sigmoid κ modifier values
    across the HSI range — useful for visualisation and validation.
    """
    if hsi_values is None:
        hsi_values = np.linspace(0, 1, 101)

    return pd.DataFrame({
        "hsi":    hsi_values,
        "linear":  [linear_kappa_modifier(h) for h in hsi_values],
        "sigmoid": [sigmoid_kappa_modifier(h) for h in hsi_values],
        "delta":   [sigmoid_kappa_modifier(h) - linear_kappa_modifier(h)
                    for h in hsi_values],
    })
