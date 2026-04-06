"""
data/integrate_real_hsi.py
──────────────────────────
Extracts real HSI scores from the team's merged county DataFrame
(produced by Model_Simulation_1.ipynb) and outputs two artefacts:

  data/processed/nc_county_hsi_real.csv   — one row per NC county with all
                                             six normalised HSI sub-scores,
                                             computed beta/kappa, and
                                             simulation ground-truth results.

  data/processed/hsi_distributions.json  — per-score min/max/mean/std used
                                             to anchor generate_data.py so
                                             synthetic training samples are
                                             drawn from the real NC distribution
                                             rather than wide uniform ranges.

Usage
-----
    python data/integrate_real_hsi.py

Expects the merged DataFrame to be available at:
    data/processed/merged_county_df.csv

If that file is not present, run the team's main notebook first
(Model_Simulation_1.ipynb) and export the merged df:

    df.to_csv("data/processed/merged_county_df.csv", index=False)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

MERGED_CSV = os.path.join(PROCESSED_DIR, "merged_county_df.csv")
OUT_HSI    = os.path.join(PROCESSED_DIR, "nc_county_hsi_real.csv")
OUT_DIST   = os.path.join(PROCESSED_DIR, "hsi_distributions.json")


# ── HSI column mapping ────────────────────────────────────────────────────────
# Maps canonical score names used in generate_data.py → actual column names
# in the merged DataFrame. Update right side if notebook column names change.
SCORE_MAP = {
    "health_score":       "score_H",
    "mobility_score":     "score_M",
    "infrastructure_score": "score_I",
    "education_score":    "edu_awareness_score",
    "social_score":       "final_score",          # prepper/vet/permit composite
    "geo_score":          "mean_survivability",    # NOAA interpolated
}

# HSI weights (must match config.py)
HSI_WEIGHTS = {
    "health_score":         0.15,
    "mobility_score":       0.15,
    "infrastructure_score": 0.20,
    "education_score":      0.10,
    "social_score":         0.25,
    "geo_score":            0.15,
}

# SZR scenario base parameters (28 Days Later — matches generate_data.py default)
BETA0  = 0.009
KAPPA0 = 0.0045


# ── SZR simulation (frequency-dependent — matches streamlit_app.py) ───────────
def szr_ode(t, y, beta, kappa):
    S, Z, R = y
    N = S + Z + R
    if N <= 0:
        return [0.0, 0.0, 0.0]
    dS = -beta * S * Z / N
    dZ = (beta - kappa) * S * Z / N
    dR = kappa * S * Z / N
    return [dS, dZ, dR]


def simulate_county(population, beta, kappa, z0=1, days=180):
    """Run SZR for a single county. Returns solution object."""
    S0 = max(population - z0, 0)
    sol = solve_ivp(
        szr_ode,
        t_span=(0, days),
        y0=[S0, float(z0), 0.0],
        t_eval=np.linspace(0, days, days * 4),
        args=(beta, kappa),
        method="RK45",
        rtol=1e-6, atol=1e-9,
    )
    return sol


def compute_hsi(row):
    """Compute composite HSI from the six sub-scores."""
    raw = sum(HSI_WEIGHTS[k] * float(row.get(k, 0.5)) for k in HSI_WEIGHTS)
    return float(np.clip(raw, 0.0, 1.0))


def derive_parameters(hsi, beta0=BETA0, kappa0=KAPPA0, a=0.3, b=0.5):
    """
    Derive per-county beta and kappa from HSI.
    Higher HSI → lower beta (communities better at avoiding contact)
               → higher kappa (communities better at neutralising zombies)

    beta  = beta0  × (1 - a × hsi)   clipped to [beta0×0.2, beta0]
    kappa = kappa0 × (1 + b × hsi)   clipped to [kappa0,    kappa0×2]
    """
    beta  = float(np.clip(beta0  * (1.0 - a * hsi), beta0  * 0.2, beta0))
    kappa = float(np.clip(kappa0 * (1.0 + b * hsi), kappa0,        kappa0 * 2.0))
    return beta, kappa


# ── Main ──────────────────────────────────────────────────────────────────────

def build_from_processed_csvs():
    """
    Fallback: build the merged DataFrame directly from the team's individual
    processed CSVs (nc_health.csv, nc_mobility.csv, nc_infrastructure.csv,
    nc_geography.csv, nc_social.csv, nc_education.csv) when the notebook's
    merged_county_df.csv is not available.

    Column expectations per file:
      nc_health.csv         → GEOID/county, score_H
      nc_mobility.csv       → GEOID/county, score_M
      nc_infrastructure.csv → GEOID/county, score_I
      nc_geography.csv      → GEOID/county, mean_survivability (or score_G)
      nc_social.csv         → GEOID/county, final_score (or score_C)
      nc_education.csv      → GEOID/county, edu_awareness_score (or score_E)
    """
    processed = os.path.join(_ROOT, "data", "processed")
    files = {
        "health_score":         ("nc_health.csv",         ["score_H", "score_h", "health_score"]),
        "mobility_score":       ("nc_mobility.csv",        ["score_M", "score_m", "mobility_score"]),
        "infrastructure_score": ("nc_infrastructure.csv",  ["score_I", "score_i", "infrastructure_score"]),
        "geo_score":            ("nc_geography.csv",       ["mean_survivability", "score_G", "geo_score"]),
        "social_score":         ("nc_social.csv",          ["final_score", "score_C", "social_score"]),
        "education_score":      ("nc_education.csv",       ["edu_awareness_score", "score_E", "education_score"]),
    }

    # Try to load a population reference file
    pop_file = os.path.join(processed, "nc_health.csv")
    if not os.path.exists(pop_file):
        pop_file = os.path.join(processed, "nc_mobility.csv")

    # Find the county identifier column
    def find_county_col(df):
        for c in ["county", "county_name", "NAME", "GEOID", "county_key", "GEO_ID"]:
            if c in df.columns:
                return c
        return df.columns[0]

    def find_score_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # Try case-insensitive
        lower = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    base_df = None
    for score_name, (fname, col_candidates) in files.items():
        fpath = os.path.join(processed, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found — will use median fill for {score_name}")
            continue
        df_src = pd.read_csv(fpath, dtype=str)
        county_col = find_county_col(df_src)
        score_col  = find_score_col(df_src, col_candidates)
        if score_col is None:
            print(f"  WARNING: could not find score column in {fname} — tried {col_candidates}")
            print(f"    Available columns: {list(df_src.columns)}")
            continue
        df_src["_county_key"] = df_src[county_col].astype(str).str.upper().str.strip()
        df_src[score_name]    = pd.to_numeric(df_src[score_col], errors="coerce")
        subset = df_src[["_county_key", score_name]].drop_duplicates("_county_key")
        print(f"  Loaded {fname}: {len(subset)} counties, col='{score_col}'")
        if base_df is None:
            base_df = subset
        else:
            base_df = base_df.merge(subset, on="_county_key", how="outer")

    if base_df is None or len(base_df) == 0:
        raise RuntimeError(
            "No processed CSV files found in data/processed/.\n"
            "Please ensure at least nc_health.csv and nc_mobility.csv exist,\n"
            "or export merged_county_df.csv from Model_Simulation_1.ipynb."
        )

    # Add placeholder population (use 50000 if no pop data available)
    base_df["population"] = 50000
    base_df["county"]     = base_df["_county_key"]
    print(f"\nBuilt merged DataFrame from processed CSVs: {len(base_df)} counties")
    return base_df


def build_county_hsi_table(merged_path=MERGED_CSV, from_processed=False):
    """
    Load county data from either:
      (a) the merged notebook CSV (merged_county_df.csv) — preferred
      (b) individual processed CSVs in data/processed/            — fallback

    Normalises sub-scores, computes composite HSI, derives beta/kappa,
    runs per-county SZR simulation, and returns a clean result DataFrame.
    """
    if from_processed or not os.path.exists(merged_path):
        if not from_processed:
            print(f"merged_county_df.csv not found — falling back to processed CSVs.")
        df = build_from_processed_csvs()
    else:
        df = pd.read_csv(merged_path)
        print(f"Loaded merged DataFrame: {df.shape[0]} rows × {df.shape[1]} cols")

    # ── Normalise sub-scores to [0, 1] ───────────────────────────────────────
    records = []
    for _, row in df.iterrows():
        county = str(row.get("county", row.get("county_key", row.get("_county_key", "UNKNOWN"))))
        pop    = float(row.get("population", 50000))

        sub = {}
        for canonical, col in SCORE_MAP.items():
            # Try canonical name first, then the raw column name
            val = row.get(canonical, row.get(col, np.nan))
            sub[canonical] = float(val) if not pd.isna(val) else np.nan

        missing = sum(1 for v in sub.values() if np.isnan(v))
        if missing > 4:
            print(f"  Skipping {county} — {missing} missing sub-scores")
            continue

        records.append({"county": county, "population": pop, **sub})

    result_df = pd.DataFrame(records)

    # Fill NaN sub-scores with column median across NC counties
    for col in SCORE_MAP:
        if col in result_df.columns:
            med = result_df[col].median()
            n_filled = result_df[col].isna().sum()
            result_df[col] = result_df[col].fillna(med)
            if n_filled > 0:
                print(f"  {col}: filled {n_filled} NaN(s) with median={med:.4f}")

    # Renormalise each sub-score to strict [0, 1] across NC
    for col in SCORE_MAP:
        mn, mx = result_df[col].min(), result_df[col].max()
        if mx > mn:
            result_df[col] = (result_df[col] - mn) / (mx - mn)
        else:
            result_df[col] = 0.5

    # ── Composite HSI ─────────────────────────────────────────────────────────
    result_df["hsi"] = result_df.apply(compute_hsi, axis=1)

    # ── Derive per-county beta / kappa ────────────────────────────────────────
    result_df[["beta", "kappa"]] = result_df["hsi"].apply(
        lambda h: pd.Series(derive_parameters(h))
    )

    # ── Run SZR simulation for each county ───────────────────────────────────
    sim_rows = []
    for _, row in result_df.iterrows():
        sol = simulate_county(row["population"], row["beta"], row["kappa"])
        S, Z = sol.y[0], sol.y[1]
        N    = row["population"]
        peak_idx = int(np.argmax(Z))
        sim_rows.append({
            "survival_fraction": float(S[-1] / N),
            "peak_zombie_frac":  float(Z[peak_idx] / N),
            "time_to_peak_days": float(sol.t[peak_idx]),
            "contained":         int(S[-1] / N > 0.10),
        })

    sim_df = pd.DataFrame(sim_rows)
    result_df = pd.concat([result_df.reset_index(drop=True), sim_df], axis=1)

    return result_df


def build_hsi_distributions(result_df):
    """
    Compute per-score statistics used to anchor the synthetic data generator.
    Returns a dict ready to be written to hsi_distributions.json.
    """
    dist = {}
    for col in list(SCORE_MAP.keys()) + ["hsi", "beta", "kappa"]:
        if col not in result_df.columns:
            continue
        s = result_df[col].dropna()
        dist[col] = {
            "min":  round(float(s.min()),  6),
            "max":  round(float(s.max()),  6),
            "mean": round(float(s.mean()), 6),
            "std":  round(float(s.std()),  6),
            "p10":  round(float(s.quantile(0.10)), 6),
            "p90":  round(float(s.quantile(0.90)), 6),
        }
    return dist


def generate_county_presets(result_df, top_n=15):
    """
    Print Streamlit COUNTY_PRESETS dict entries using real HSI values.
    Copy the output into app/streamlit_app.py.
    """
    # Select key counties by name match
    key_counties = [
        "WAKE", "MECKLENBURG", "GUILFORD", "DURHAM", "FORSYTH",
        "CUMBERLAND", "BUNCOMBE", "PITT", "NEW HANOVER", "ALAMANCE",
        "ONSLOW", "WAYNE", "UNION", "CABARRUS", "ORANGE",
    ]
    print("\n# ── COUNTY_PRESETS (replace in app/streamlit_app.py) ──────────────")
    print("COUNTY_PRESETS = {")
    print('    "— Select a county —": None,')

    for name in key_counties:
        row = result_df[result_df["county"].str.upper().str.contains(name)]
        if row.empty:
            continue
        row = row.iloc[0]
        label = row["county"].replace(" County, North Carolina", "").replace(", North Carolina", "")
        print(
            f'    "{label}": {{'
            f'"initial_population": {int(row["population"])}, '
            f'"initial_infected": 5, '
            f'"mobility_score": {row["mobility_score"]:.3f}, '
            f'"infrastructure_score": {row["infrastructure_score"]:.3f}, '
            f'"health_score": {row["health_score"]:.3f}, '
            f'"education_score": {row["education_score"]:.3f}, '
            f'"social_score": {row["social_score"]:.3f}, '
            f'"geo_score": {row["geo_score"]:.3f}, '
            f'"hsi": {row["hsi"]:.3f}}},'
        )
    print("}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NC HSI Integration")
    parser.add_argument(
        "--from-processed", action="store_true",
        help=(
            "Build from individual processed CSVs in data/processed/ "
            "(nc_health.csv, nc_mobility.csv, etc.) instead of the merged notebook CSV. "
            "Use this when Model_Simulation_1.ipynb cannot be run locally."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NC HSI Integration — Real County Data")
    print("=" * 60)

    result_df = build_county_hsi_table(from_processed=args.from_processed)

    # Save county HSI table
    result_df.to_csv(OUT_HSI, index=False)
    print(f"\nCounty HSI table saved → {OUT_HSI}  ({len(result_df)} counties)")

    # Print summary
    print(f"\nHSI range: {result_df['hsi'].min():.3f} – {result_df['hsi'].max():.3f}")
    print(f"Survival range: {result_df['survival_fraction'].min():.1%} – {result_df['survival_fraction'].max():.1%}")
    print(f"\nTop 5 surviving counties:")
    top = result_df.nlargest(5, "survival_fraction")[["county","hsi","survival_fraction","peak_zombie_frac"]]
    print(top.to_string(index=False))
    print(f"\nBottom 5 surviving counties:")
    bot = result_df.nsmallest(5, "survival_fraction")[["county","hsi","survival_fraction","peak_zombie_frac"]]
    print(bot.to_string(index=False))

    # Save distributions
    dist = build_hsi_distributions(result_df)
    with open(OUT_DIST, "w") as f:
        json.dump(dist, f, indent=2)
    print(f"\nHSI distributions saved → {OUT_DIST}")

    # Print county presets for Streamlit
    generate_county_presets(result_df)

    print("\nNext steps:")
    print("  1. Copy COUNTY_PRESETS output above into app/streamlit_app.py")
    print("  2. Run: python data/generate_data.py   (uses hsi_distributions.json)")
    print("  3. Run: python model/train.py")


if __name__ == "__main__":
    main()
