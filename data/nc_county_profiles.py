"""
nc_county_profiles.py
=====================
Derives county-level HSI scores for the 5 NC app profiles by averaging
census-tract scores from the DASC 6010 team's output CSVs stored in
data/dasc6010/.

# mobility_score = 1 - score_M because score_M measures escape
# capacity (higher = more mobile) but SZR mobility_score models
# transmission spread (higher = more spread). Polarity inverted.

Returned dict shape matches the COUNTY_PROFILES format in streamlit_app.py:
each value is a dict of slider-key → float, ready to be merged with the
population/beta/zeta/alpha values that remain hardcoded.
"""

import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DASC_DIR = os.path.join(_HERE, "dasc6010")

# ---------------------------------------------------------------------------
# County FIPS → display name (only the 5 app-profile counties)
# ---------------------------------------------------------------------------
_COUNTY_FIPS = {
    "37119": "Mecklenburg (Charlotte)",
    "37183": "Wake (Raleigh)",
    "37055": "Dare (Outer Banks)",
    "37177": "Tyrrell (Rural)",
    "37155": "Robeson (Mixed)",
}

# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------

def _load_scores() -> pd.DataFrame:
    """
    Read the three section CSVs, derive county FIPS from the first 5 digits
    of GEOID, average tracts within each county, and return a tidy DataFrame
    with one row per county.
    """
    df_m = pd.read_csv(os.path.join(_DASC_DIR, "nc_mobility.csv"),
                       dtype={"GEOID": str})
    df_h = pd.read_csv(os.path.join(_DASC_DIR, "nc_health.csv"),
                       dtype={"GEOID": str})
    df_i = pd.read_csv(os.path.join(_DASC_DIR, "nc_infrastructure.csv"),
                       dtype={"GEOID": str})

    for df in (df_m, df_h, df_i):
        df["county_fips"] = df["GEOID"].str[:5]

    # Aggregate to county level
    mob   = df_m.groupby("county_fips")["score_M"].mean().rename("score_M")
    health = df_h.groupby("county_fips")["score_H"].mean().rename("score_H")
    infra  = df_i.groupby("county_fips")["score_I"].mean().rename("score_I")

    merged = pd.concat([mob, health, infra], axis=1).reset_index()
    merged.columns = ["county_fips", "score_M", "score_H", "score_I"]

    # Invert mobility polarity: score_M = escape capacity (high = mobile),
    # but SZR mobility_score = spread risk (high = more transmission).
    merged["mobility_score"]       = (1.0 - merged["score_M"]).round(4)
    merged["health_score"]         = merged["score_H"].round(4)
    merged["infrastructure_score"] = merged["score_I"].round(4)

    return merged.set_index("county_fips")


def build_hsi_overrides() -> dict:
    """
    Return a dict mapping each county display name to its three HSI scores.
    Only the keys that differ from the hardcoded defaults are included so
    the caller can safely update() the existing profile dicts.
    """
    scores = _load_scores()
    overrides = {}
    for fips, name in _COUNTY_FIPS.items():
        if fips not in scores.index:
            continue
        row = scores.loc[fips]
        overrides[name] = {
            "mobility_score":       float(row["mobility_score"]),
            "health_score":         float(row["health_score"]),
            "infrastructure_score": float(row["infrastructure_score"]),
        }
    return overrides


# ---------------------------------------------------------------------------
# CLI: print derived values when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    overrides = build_hsi_overrides()
    print(f"{'County':<28} {'mobility':>10} {'health':>10} {'infra':>10}")
    print("-" * 62)
    for name, vals in overrides.items():
        print(
            f"{name:<28}"
            f"  {vals['mobility_score']:>8.4f}"
            f"  {vals['health_score']:>8.4f}"
            f"  {vals['infrastructure_score']:>8.4f}"
        )
