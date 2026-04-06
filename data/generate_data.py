"""
data/generate_data.py
─────────────────────
Generates the synthetic SZR training dataset for the surrogate model.

v2 changes vs original:
  - HSI sub-scores are sampled from the REAL NC county distribution
    (loaded from data/processed/hsi_distributions.json, produced by
    data/integrate_real_hsi.py) rather than wide uniform ranges.
  - Beta and kappa are derived from HSI using the same formula as the
    simulation notebook, not randomized independently.
  - Five zombie scenario profiles are included as a stratification
    variable to ensure coverage across outbreak types.
  - Outputs data/szr_synthetic.csv (same schema as before — train.py
    requires no changes).

Run order:
  1. python data/integrate_real_hsi.py    ← produces hsi_distributions.json
  2. python data/generate_data.py         ← produces szr_synthetic.csv
  3. python model/train.py
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from model.config import HSI_WEIGHTS_NAMED as HSI_WEIGHTS  # single source of truth
DIST_PATH = os.path.join(_ROOT, "data", "processed", "hsi_distributions.json")
OUT_PATH  = os.path.join(_ROOT, "data", "szr_synthetic.csv")

N_SAMPLES    = 10_000   # total synthetic scenarios
RANDOM_STATE = 42
DAYS         = 180
RNG          = np.random.default_rng(RANDOM_STATE)


# ── Zombie scenario profiles ──────────────────────────────────────────────────
# Each profile defines a (beta_base, kappa_base) pair matching the show canon.
# Scenarios are sampled proportionally so all outbreak types are represented.
SCENARIOS = {
    "walking_dead": {"beta": 2.80e-3, "kappa": 2.52e-3, "weight": 0.20},
    "last_of_us":   {"beta": 6.00e-3, "kappa": 3.90e-3, "weight": 0.20},
    "world_war_z":  {"beta": 3.60e-3, "kappa": 2.88e-3, "weight": 0.20},
    "28_days":      {"beta": 9.00e-3, "kappa": 4.50e-3, "weight": 0.25},
    "rabies":       {"beta": 0.50e-3, "kappa": 4.75e-4, "weight": 0.15},
}


# ── Fallback distributions (used when hsi_distributions.json not found) ───────
FALLBACK_DIST = {
    "health_score":         {"mean": 0.50, "std": 0.15, "min": 0.10, "max": 0.90},
    "mobility_score":       {"mean": 0.50, "std": 0.18, "min": 0.05, "max": 0.98},
    "infrastructure_score": {"mean": 0.50, "std": 0.16, "min": 0.08, "max": 0.95},
    "education_score":      {"mean": 0.50, "std": 0.14, "min": 0.10, "max": 0.92},
    "social_score":         {"mean": 0.50, "std": 0.17, "min": 0.05, "max": 0.95},
    "geo_score":            {"mean": 0.50, "std": 0.20, "min": 0.00, "max": 1.00},
}

CONTAINMENT_THRESHOLD = 0.10   # S/N > 10% at day 180 = contained


# ── SZR ODE (frequency-dependent — matches streamlit_app.py) ─────────────────
def szr_ode(t, y, beta, kappa, alpha=0.01):
    S, Z, R = y
    N = S + Z + R
    if N <= 0:
        return [0.0, 0.0, 0.0]
    dS = -beta * S * Z / N
    dZ = beta * S * Z / N - (kappa + alpha) * Z
    dR = (kappa + alpha) * Z
    return [dS, dZ, dR]


def run_simulation(population, beta, kappa, initial_infected, alpha=0.01):
    """Run SZR and return outcome metrics."""
    S0 = max(float(population) - float(initial_infected), 1.0)
    Z0 = float(initial_infected)
    t_span = (0, DAYS)
    t_eval = np.linspace(0, DAYS, DAYS * 2)

    sol = solve_ivp(
        szr_ode,
        t_span=t_span,
        y0=[S0, Z0, 0.0],
        t_eval=t_eval,
        args=(beta, kappa, alpha),
        method="RK45",
        rtol=1e-6, atol=1e-9,
    )

    S, Z = sol.y[0], sol.y[1]
    N    = float(population)
    peak_idx = int(np.argmax(Z))

    return {
        "peak_zombie_fraction": float(np.clip(Z[peak_idx] / N, 0, 1)),
        "time_to_peak":         float(sol.t[peak_idx]),
        "containment":          int(S[-1] / N > CONTAINMENT_THRESHOLD),
    }


def load_distributions():
    """Load real NC HSI distributions or fall back to defaults."""
    if os.path.exists(DIST_PATH):
        with open(DIST_PATH) as f:
            dist = json.load(f)
        print(f"Loaded real NC HSI distributions from {DIST_PATH}")
        return dist, True
    else:
        print(f"WARNING: {DIST_PATH} not found.")
        print("  Run data/integrate_real_hsi.py first for real NC distributions.")
        print("  Falling back to wide uniform distributions.")
        return FALLBACK_DIST, False


def sample_hsi_score(name, dist, real=True):
    """
    Sample a single HSI sub-score.
    If real distributions are available, sample from a clipped normal
    anchored to the actual NC county mean and std.
    Otherwise sample from uniform [0, 1].
    """
    if real and name in dist:
        d = dist[name]
        val = RNG.normal(loc=d["mean"], scale=d["std"])
        return float(np.clip(val, d.get("min", 0.0), d.get("max", 1.0)))
    return float(RNG.uniform(0.0, 1.0))


def compute_hsi(scores):
    """Composite HSI from sub-scores dict."""
    return float(np.clip(
        sum(HSI_WEIGHTS[k] * v for k, v in scores.items()),
        0.0, 1.0
    ))


def derive_beta_kappa(hsi, beta_base, kappa_base, a=0.3, b=0.5):
    """
    Per-scenario beta and kappa modified by HSI.
    Higher HSI → lower beta (better at avoiding contact)
               → higher kappa (better at neutralising zombies)
    """
    beta  = float(np.clip(beta_base  * (1.0 - a * hsi), beta_base  * 0.2, beta_base))
    kappa = float(np.clip(kappa_base * (1.0 + b * hsi), kappa_base,        kappa_base * 2.0))
    return beta, kappa


def main() -> None:
    print("=" * 60)
    print("SZR Synthetic Dataset Generator v2")
    print(f"Target: {N_SAMPLES:,} scenarios → {OUT_PATH}")
    print("=" * 60)

    dist, real = load_distributions()

    # Build scenario sample counts proportional to weights
    scenario_names = list(SCENARIOS.keys())
    weights        = np.array([SCENARIOS[s]["weight"] for s in scenario_names])
    counts         = np.round(weights / weights.sum() * N_SAMPLES).astype(int)
    counts[-1]     = N_SAMPLES - counts[:-1].sum()   # ensure exact total

    rows = []
    total = 0

    for scenario_name, n in zip(scenario_names, counts):
        sc = SCENARIOS[scenario_name]
        print(f"\nGenerating {n:,} samples for scenario: {scenario_name}")

        for i in range(n):
            # Sample HSI sub-scores from real NC distribution
            hsi_scores = {k: sample_hsi_score(k, dist, real) for k in HSI_WEIGHTS}
            hsi        = compute_hsi(hsi_scores)

            # Derive effective beta/kappa for this county-scenario combination
            beta, kappa = derive_beta_kappa(hsi, sc["beta"], sc["kappa"])

            # Sample population and initial infected
            population       = int(RNG.integers(5_000, 1_200_000))
            initial_infected = int(RNG.integers(1, min(500, population // 100 + 1)))

            # Run simulation
            try:
                outcome = run_simulation(population, beta, kappa, initial_infected)
            except Exception:
                continue

            rows.append({
                # Input features (must match FEATURE_COLUMNS in szr_predictor.py)
                "beta":                beta,
                "zeta":               kappa,
                "alpha":              0.01,
                "initial_population": population,
                "initial_infected":   initial_infected,
                "mobility_score":     hsi_scores["mobility_score"],
                "infrastructure_score": hsi_scores["infrastructure_score"],
                "health_score":       hsi_scores["health_score"],
                # Additional HSI scores (not currently in model features but
                # available for future ablations)
                "education_score":    hsi_scores["education_score"],
                "social_score":       hsi_scores["social_score"],
                "geo_score":          hsi_scores["geo_score"],
                "hsi":                hsi,
                # Scenario metadata
                "scenario":           scenario_name,
                # Targets
                "peak_zombie_fraction": outcome["peak_zombie_fraction"],
                "time_to_peak":         outcome["time_to_peak"],
                "containment":          outcome["containment"],
            })

            total += 1
            if (total % 1000) == 0:
                print(f"  {total:,} / {N_SAMPLES:,} complete...", end="\r")

    df = pd.DataFrame(rows)
    print(f"\n\nGenerated {len(df):,} valid scenarios")
    print(f"Containment rate: {df['containment'].mean():.1%}")
    print(f"Peak zombie frac — mean: {df['peak_zombie_fraction'].mean():.3f}  "
          f"std: {df['peak_zombie_fraction'].std():.3f}")
    print(f"Time to peak     — mean: {df['time_to_peak'].mean():.1f}d  "
          f"std: {df['time_to_peak'].std():.1f}d")

    print(f"\nScenario breakdown:")
    print(df.groupby("scenario")[["peak_zombie_fraction","containment"]].mean().round(3))

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print("\nNext: python model/train.py")


if __name__ == "__main__":
    main()
