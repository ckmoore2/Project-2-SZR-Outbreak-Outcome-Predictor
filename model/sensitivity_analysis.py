"""
sensitivity_analysis.py — HSI weight sensitivity sweep.

Sweeps pairs of HSI category weights against each other while holding
the remaining categories fixed at V1 defaults. For each weight combination
and each scenario, runs the SZR ODE on a representative NC tract distribution
and records:
  - NC-wide survival fraction (population-weighted mean)
  - Containment rate (% of tracts with S[-1]/N > threshold)
  - Median α_eff across tracts

Outputs:
  outputs/sensitivity/<pair>_<scenario>.csv   — raw sweep results
  outputs/sensitivity/summary.csv             — aggregated findings
  outputs/sensitivity/heatmaps.png            — visual grid
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from copy import deepcopy

from config import (
    HSI_WEIGHTS_V1,
    SCENARIOS,
    SENSITIVITY_CONFIG,
    CONTAINMENT_THRESHOLD,
    PATHS,
)
from szr_model import run_tract_simulation


# ── Synthetic tract distribution (used when real tract data is unavailable) ───
# Draws from NC census statistics to create a representative 500-tract sample.
# Replace with real tract_df once data/processed/ is fully populated.

RNG = np.random.default_rng(42)

def synthetic_nc_tracts(n: int = 500) -> pd.DataFrame:
    """
    Generate a synthetic but statistically representative NC tract distribution.

    HSI sub-scores drawn from Beta distributions calibrated to approximate
    actual NC tract distributions from the team's real data outputs.
    Population drawn from NC tract size distribution (median ~3,800).
    """
    return pd.DataFrame({
        "GEOID":            [f"37{i:06d}" for i in range(n)],
        "population":       RNG.lognormal(mean=8.2, sigma=0.6, size=n).astype(int),
        "norm_pop_density": RNG.beta(a=1.5, b=4.0, size=n),   # right-skewed — most tracts rural
        "score_H":          RNG.beta(a=3.0, b=3.5, size=n),
        "score_M":          RNG.beta(a=2.8, b=3.2, size=n),
        "score_I":          RNG.beta(a=2.5, b=3.0, size=n),
        "score_E":          RNG.beta(a=2.0, b=4.0, size=n),
        "score_C":          RNG.beta(a=2.2, b=3.8, size=n),
        "score_G":          RNG.beta(a=3.0, b=3.0, size=n),
    })


def compute_hsi(row: pd.Series, weights: dict) -> float:
    """Compute HSI for a tract given a weight dict."""
    raw = (
        weights["H"] * row["score_H"] +
        weights["M"] * row["score_M"] +
        weights["I"] * row["score_I"] +
        weights["E"] * row["score_E"] +
        weights["C"] * row["score_C"] +
        weights["G"] * row["score_G"]
    )
    return float(np.clip(raw, 0.0, 1.0))


def run_sweep_for_pair(
    cat_a: str,
    cat_b: str,
    scenario_key: str,
    tract_df: pd.DataFrame,
    n_steps: int = SENSITIVITY_CONFIG["n_steps"],
) -> pd.DataFrame:
    """
    Sweep weights for cat_a and cat_b while holding others fixed at V1 values.
    At each (w_a, w_b) point, the remaining weight is redistributed
    proportionally among the other four categories.

    Returns a DataFrame with columns:
      w_a, w_b, survival_frac, containment_rate, median_alpha_eff
    """
    scenario = SCENARIOS[scenario_key]
    beta_base  = scenario["beta"]
    kappa_base = scenario["kappa"]
    alpha      = scenario["alpha"]

    other_cats = [c for c in "HMIECG" if c not in (cat_a, cat_b)]
    v1_other_total = sum(HSI_WEIGHTS_V1[c] for c in other_cats)

    step_vals = np.linspace(0.0, 1.0, n_steps)
    records = []

    for w_a in step_vals:
        for w_b in step_vals:
            if w_a + w_b > 1.0:
                continue   # invalid — remaining weight would be negative

            remaining = 1.0 - w_a - w_b
            weights = deepcopy(HSI_WEIGHTS_V1)
            weights[cat_a] = w_a
            weights[cat_b] = w_b

            # Redistribute remaining weight proportionally among other cats
            if v1_other_total > 0:
                for c in other_cats:
                    weights[c] = remaining * (HSI_WEIGHTS_V1[c] / v1_other_total)
            else:
                share = remaining / len(other_cats)
                for c in other_cats:
                    weights[c] = share

            # Simulate all tracts
            survival_fracs = []
            alpha_effs     = []
            contained_count = 0

            for _, row in tract_df.iterrows():
                hsi = compute_hsi(row, weights)
                res = run_tract_simulation(
                    population=row["population"],
                    initial_infected=1,
                    beta_base=beta_base,
                    kappa_base=kappa_base,
                    alpha=alpha,
                    hsi=hsi,
                    norm_pop_density=row["norm_pop_density"],
                    mobility_score=row["score_M"],
                )
                survival_fracs.append(res["survival_frac"])
                alpha_effs.append(res["alpha_eff"])
                if res["contained"]:
                    contained_count += 1

            pop_weights = tract_df["population"].values / tract_df["population"].sum()
            weighted_survival = float(np.dot(survival_fracs, pop_weights))
            containment_rate  = contained_count / len(tract_df)
            median_alpha      = float(np.median(alpha_effs))

            records.append({
                f"w_{cat_a.lower()}": round(w_a, 2),
                f"w_{cat_b.lower()}": round(w_b, 2),
                "survival_frac":   round(weighted_survival, 4),
                "containment_rate": round(containment_rate, 4),
                "median_alpha_eff": round(median_alpha, 4),
            })

    return pd.DataFrame(records)


def run_all_sweeps(tract_df: pd.DataFrame = None) -> dict:
    """
    Run all configured weight pair sweeps across all configured scenarios.
    Returns a dict: {(cat_a, cat_b, scenario_key): DataFrame}
    Saves CSVs and summary to outputs/sensitivity/.
    """
    os.makedirs(SENSITIVITY_CONFIG["output_path"], exist_ok=True)

    if tract_df is None:
        print("Real tract data not found — using synthetic NC distribution (n=500).")
        tract_df = synthetic_nc_tracts(500)

    all_results = {}
    summary_rows = []

    for (cat_a, cat_b) in SENSITIVITY_CONFIG["sweep_pairs"]:
        for scenario_key in SENSITIVITY_CONFIG["scenarios_to_test"]:
            print(f"  Sweeping {cat_a} × {cat_b} — scenario: {scenario_key} ...", end=" ")

            df = run_sweep_for_pair(cat_a, cat_b, scenario_key, tract_df)

            fname = f"{cat_a.lower()}{cat_b.lower()}_{scenario_key}.csv"
            fpath = os.path.join(SENSITIVITY_CONFIG["output_path"], fname)
            df.to_csv(fpath, index=False)
            print(f"saved → {fpath}")

            all_results[(cat_a, cat_b, scenario_key)] = df

            # Summary: effect of moving from V1 weights to max-C and max-G extremes
            v1_row = df[
                (df[f"w_{cat_a.lower()}"].round(1) == round(HSI_WEIGHTS_V1[cat_a], 1)) &
                (df[f"w_{cat_b.lower()}"].round(1) == round(HSI_WEIGHTS_V1[cat_b], 1))
            ]
            best_row = df.sort_values("survival_frac", ascending=False).head(1)
            worst_row = df.sort_values("survival_frac").head(1)

            summary_rows.append({
                "pair":           f"{cat_a}×{cat_b}",
                "scenario":       scenario_key,
                "v1_survival":    v1_row["survival_frac"].values[0] if len(v1_row) else None,
                "best_survival":  best_row["survival_frac"].values[0],
                "worst_survival": worst_row["survival_frac"].values[0],
                f"best_w_{cat_a.lower()}": best_row[f"w_{cat_a.lower()}"].values[0],
                f"best_w_{cat_b.lower()}": best_row[f"w_{cat_b.lower()}"].values[0],
                "sensitivity_range": round(
                    best_row["survival_frac"].values[0] -
                    worst_row["survival_frac"].values[0], 4
                ),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(SENSITIVITY_CONFIG["output_path"], "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")

    return all_results


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_heatmaps(all_results: dict, save_path: str = None):
    """
    Generate heatmap grid: one subplot per (pair, scenario) combination.
    Colour = population-weighted survival fraction.
    """
    pairs     = SENSITIVITY_CONFIG["sweep_pairs"]
    scenarios = SENSITIVITY_CONFIG["scenarios_to_test"]
    nrows, ncols = len(scenarios), len(pairs)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.patch.set_facecolor("#0a0a08")

    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, scenario_key in enumerate(scenarios):
        for col_idx, (cat_a, cat_b) in enumerate(pairs):
            ax  = axes[row_idx][col_idx]
            key = (cat_a, cat_b, scenario_key)
            df  = all_results.get(key)

            ax.set_facecolor("#0d0d0a")
            if df is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color="#5a5040", transform=ax.transAxes)
                continue

            col_a = f"w_{cat_a.lower()}"
            col_b = f"w_{cat_b.lower()}"
            pivot = df.pivot_table(
                index=col_b, columns=col_a, values="survival_frac", aggfunc="mean"
            )

            im = ax.imshow(
                pivot.values,
                cmap="RdYlGn", vmin=0, vmax=1,
                origin="lower", aspect="auto",
                extent=[0, 1, 0, 1],
            )

            # Mark V1 weight position
            ax.axvline(HSI_WEIGHTS_V1[cat_a], color="white", linewidth=0.8,
                       linestyle="--", alpha=0.5)
            ax.axhline(HSI_WEIGHTS_V1[cat_b], color="white", linewidth=0.8,
                       linestyle="--", alpha=0.5)
            ax.plot(HSI_WEIGHTS_V1[cat_a], HSI_WEIGHTS_V1[cat_b],
                    "w+", markersize=10, markeredgewidth=1.5)

            ax.set_xlabel(f"weight {cat_a}", color="#8b8070", fontsize=8,
                          fontfamily="monospace")
            ax.set_ylabel(f"weight {cat_b}", color="#8b8070", fontsize=8,
                          fontfamily="monospace")
            ax.set_title(
                f"{cat_a}×{cat_b} | {SCENARIOS[scenario_key]['label']}",
                color="#c8bfa8", fontsize=8, fontfamily="monospace",
            )
            ax.tick_params(colors="#5a5040", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor((0.80, 0.13, 0.0, 0.3))

            plt.colorbar(im, ax=ax, fraction=0.04, label="survival frac")

    plt.suptitle(
        "HSI Weight Sensitivity — Population-Weighted NC Survival\n"
        "White + = V1 weights. Dashed lines = V1 axes.",
        color="#e8e0d0", fontsize=10, fontfamily="monospace", y=1.02,
    )
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(SENSITIVITY_CONFIG["output_path"], "heatmaps.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a08")
    print(f"Heatmaps saved → {save_path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SZR HSI Sensitivity Analysis")
    print("=" * 60)

    # Try to load real tract data, fall back to synthetic
    tract_data_path = os.path.join(PATHS["processed_data"], "nc_hsi_tract_scores.csv")
    if os.path.exists(tract_data_path):
        print(f"Loading real tract data from {tract_data_path}")
        tract_df = pd.read_csv(tract_data_path)
        required_cols = {"GEOID","population","hsi","norm_pop_density","score_M",
                         "score_H","score_I","score_E","score_C","score_G"}
        if not required_cols.issubset(set(tract_df.columns)):
            print("  Missing columns — falling back to synthetic distribution.")
            tract_df = None
    else:
        print(f"Tract data not found at {tract_data_path} — using synthetic.")
        tract_df = None

    results = run_all_sweeps(tract_df)
    plot_heatmaps(results)

    print("\nKey findings:")
    summary = pd.read_csv(
        os.path.join(SENSITIVITY_CONFIG["output_path"], "summary.csv")
    )
    print(summary.to_string(index=False))
