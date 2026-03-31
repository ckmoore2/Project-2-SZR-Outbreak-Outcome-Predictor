"""
nc_county_profiles.py
=====================
Derives county-level HSI scores for the 5 NC app profiles by averaging
census-tract scores from the DASC 6010 team's output CSVs stored in
data/dasc6010/.

Mobility derivation
-------------------
mobility_score is derived from transit_dependence (% households without
vehicles) rather than 1 - score_M.  The original 1 - score_M approach
compressed all NC counties into a 0.03–0.08 band because NC is
universally car-dependent (score_M ≈ 0.92–0.97 statewide).

transit_dependence gives meaningful inter-county spread (1.64–13.29%
across NC's 100 counties).  It is min-max normalised across ALL NC
counties so the resulting mobility_score spans the full [0, 1] range:

    mobility_score = (county_mean_td - nc_min_td) / (nc_max_td - nc_min_td)

Higher transit_dependence → more residents without cars → more constrained
movement → higher spread risk in SZR terms → higher mobility_score.
No polarity inversion required; the direction is already correct.

Fallback: if transit_dependence is absent, falls back to
    mobility_score = 1 - score_M  (re-normalised the same way)

Health and infrastructure
-------------------------
health_score         = mean(score_H)  used directly
infrastructure_score = mean(score_I)  used directly

Returned dict shape matches the COUNTY_PROFILES format in streamlit_app.py:
each value is a dict of slider-key → float, ready to be merged with the
population/beta/zeta/alpha values that remain hardcoded.
"""

import os
import warnings

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
    with one row per county containing mobility_score, health_score, and
    infrastructure_score.
    """
    df_m = pd.read_csv(os.path.join(_DASC_DIR, "nc_mobility.csv"),
                       dtype={"GEOID": str})
    df_h = pd.read_csv(os.path.join(_DASC_DIR, "nc_health.csv"),
                       dtype={"GEOID": str})
    df_i = pd.read_csv(os.path.join(_DASC_DIR, "nc_infrastructure.csv"),
                       dtype={"GEOID": str})

    for df in (df_m, df_h, df_i):
        df["county_fips"] = df["GEOID"].str[:5]

    # ------------------------------------------------------------------
    # Mobility — prefer transit_dependence; fall back to 1 - score_M
    # ------------------------------------------------------------------
    if "transit_dependence" in df_m.columns:
        # Aggregate to county level across ALL counties (needed for
        # global min/max normalisation)
        county_td = df_m.groupby("county_fips")["transit_dependence"].mean()
        nc_min, nc_max = county_td.min(), county_td.max()
        mob = ((county_td - nc_min) / (nc_max - nc_min)).rename("mobility_score")
        _mobility_source = (
            f"transit_dependence (min-max [0,1] across {len(county_td)} NC counties; "
            f"raw range {nc_min:.2f}–{nc_max:.2f}%)"
        )
    else:
        raw_mob = df_m.groupby("county_fips")["score_M"].mean()
        inverted = 1.0 - raw_mob
        nc_min, nc_max = inverted.min(), inverted.max()
        mob = ((inverted - nc_min) / (nc_max - nc_min)).rename("mobility_score")
        _mobility_source = "1 - score_M  (re-normalised; transit_dependence not found)"

    # ------------------------------------------------------------------
    # Health and infrastructure — aggregate and use directly
    # ------------------------------------------------------------------
    health = df_h.groupby("county_fips")["score_H"].mean().rename("health_score")
    infra  = df_i.groupby("county_fips")["score_I"].mean().rename("infrastructure_score")

    merged = pd.concat([mob, health, infra], axis=1).reset_index()
    merged.columns = ["county_fips", "mobility_score", "health_score",
                      "infrastructure_score"]

    for col in ("mobility_score", "health_score", "infrastructure_score"):
        merged[col] = merged[col].round(4)

    # Store source description for CLI reporting
    merged.attrs["mobility_source"] = _mobility_source

    return merged.set_index("county_fips")


def _load_esg_scores() -> pd.DataFrame:
    """
    Load county-level E, S, G scores from the synthesized CSVs.
    Returns a DataFrame indexed by 5-digit county FIPS, or an empty DataFrame
    if the files have not yet been generated (scores default to 0.5 in caller).

    Run  python data/synthesize_missing_hsi.py  to generate these files.
    """
    paths = {
        "score_E": os.path.join(_DASC_DIR, "nc_education.csv"),
        "score_S": os.path.join(_DASC_DIR, "nc_social.csv"),
        "score_G": os.path.join(_DASC_DIR, "nc_geographic.csv"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        warnings.warn(
            f"ESG CSV(s) not found: {missing}. "
            "Run python data/synthesize_missing_hsi.py to generate them. "
            "score_E / score_S / score_G will default to 0.5.",
            stacklevel=3,
        )
        return pd.DataFrame()

    edu = pd.read_csv(paths["score_E"], dtype={"GEOID": str})[["GEOID", "score_E"]]
    soc = pd.read_csv(paths["score_S"], dtype={"GEOID": str})[["GEOID", "score_S"]]
    geo = pd.read_csv(paths["score_G"], dtype={"GEOID": str})[["GEOID", "score_G"]]

    merged = (
        edu.merge(soc, on="GEOID", how="outer")
           .merge(geo, on="GEOID", how="outer")
           .fillna(0.5)
    )
    return merged.set_index("GEOID")


def build_hsi_overrides() -> dict:
    """
    Return a dict mapping each county display name to all six HSI scores:
      mobility_score, health_score, infrastructure_score (from DASC 6010 CSVs)
      score_E, score_S, score_G (from synthesize_missing_hsi.py output)

    score_E/S/G default to 0.5 when the synthesized CSVs are absent.
    """
    scores = _load_scores()
    esg    = _load_esg_scores()

    overrides = {}
    for fips, name in _COUNTY_FIPS.items():
        if fips not in scores.index:
            continue
        row = scores.loc[fips]
        hmi = {
            "mobility_score":       float(row["mobility_score"]),
            "health_score":         float(row["health_score"]),
            "infrastructure_score": float(row["infrastructure_score"]),
        }
        if not esg.empty and fips in esg.index:
            esg_row = esg.loc[fips]
            esg_vals = {
                "score_E": float(esg_row["score_E"]),
                "score_S": float(esg_row["score_S"]),
                "score_G": float(esg_row["score_G"]),
            }
        else:
            esg_vals = {"score_E": 0.5, "score_S": 0.5, "score_G": 0.5}
        overrides[name] = {**hmi, **esg_vals}
    return overrides


# ---------------------------------------------------------------------------
# CLI: print derived values when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scores = _load_scores()
    overrides = build_hsi_overrides()

    print(f"\nMobility source: {scores.attrs.get('mobility_source', 'unknown')}\n")
    print(f"{'County':<28} {'mobility':>8} {'health':>8} {'infra':>8} "
          f"{'E':>8} {'S':>8} {'G':>8}")
    print("-" * 82)
    for name, vals in overrides.items():
        print(
            f"{name:<28}"
            f"  {vals['mobility_score']:>6.4f}"
            f"  {vals['health_score']:>6.4f}"
            f"  {vals['infrastructure_score']:>6.4f}"
            f"  {vals.get('score_E', 0.5):>6.4f}"
            f"  {vals.get('score_S', 0.5):>6.4f}"
            f"  {vals.get('score_G', 0.5):>6.4f}"
        )
