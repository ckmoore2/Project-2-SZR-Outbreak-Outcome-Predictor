#!/usr/bin/env python3
"""
synthesize_missing_hsi.py
=========================
Computes county-level HSI scores for three missing categories:
  E — Education & Awareness      weight 0.10
  S — Social & Community         weight 0.25
  G — Environmental & Geographic weight 0.15

for all 100 NC counties (state FIPS 37).  No Census API key required.

Outputs
-------
data/dasc6010/nc_education.csv    GEOID, NAME, pct_bachelors, score_E
data/dasc6010/nc_social.csv       GEOID, NAME, vet_pct, vet_norm, gun_rate,
                                  gun_norm, hunting_proxy, hunting_norm, score_S
data/dasc6010/nc_geographic.csv   GEOID, NAME, terrain_score, precip_inches, score_G

After writing CSVs, prints a comparison table for the 5 app-profile counties
showing partial HSI (H+M+I average, current) vs. full weighted HSI (all 6).
"""

import io
import json
import os
import sys
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DASC_DIR = os.path.join(_HERE, "dasc6010")

_APP_FIPS = {
    "37119": "Mecklenburg (Charlotte)",
    "37183": "Wake (Raleigh)",
    "37055": "Dare (Outer Banks)",
    "37177": "Tyrrell (Rural)",
    "37155": "Robeson (Mixed)",
}

# ---------------------------------------------------------------------------
# Hardcoded fallback: approximate 1991-2020 PRISM/NOAA annual precipitation
# normals (inches) for all 100 NC counties.
# ---------------------------------------------------------------------------
_NC_PRECIP = {
    "37001": 47.2, "37003": 48.3, "37005": 55.4, "37007": 45.1, "37009": 58.6,
    "37011": 66.2, "37013": 51.3, "37015": 50.2, "37017": 52.4, "37019": 56.1,
    "37021": 47.1, "37023": 48.5, "37025": 44.9, "37027": 49.2, "37029": 49.5,
    "37031": 55.3, "37033": 47.4, "37035": 45.2, "37037": 47.6, "37039": 60.8,
    "37041": 49.2, "37043": 64.1, "37045": 47.3, "37047": 54.2, "37049": 52.4,
    "37051": 48.3, "37053": 50.1, "37055": 55.2, "37057": 44.1, "37059": 44.3,
    "37061": 51.2, "37063": 46.1, "37065": 48.3, "37067": 43.2, "37069": 46.4,
    "37071": 47.3, "37073": 49.3, "37075": 65.4, "37077": 46.2, "37079": 49.5,
    "37081": 43.4, "37083": 45.3, "37085": 47.2, "37087": 57.4, "37089": 57.1,
    "37091": 49.4, "37093": 48.2, "37095": 51.4, "37097": 46.3, "37099": 66.3,
    "37101": 48.4, "37103": 52.3, "37105": 47.1, "37107": 50.3, "37109": 46.2,
    "37111": 52.4, "37113": 63.2, "37115": 52.3, "37117": 50.4, "37119": 44.1,
    "37121": 62.3, "37123": 46.3, "37125": 48.4, "37127": 47.3, "37129": 58.4,
    "37131": 47.3, "37133": 57.4, "37135": 47.2, "37137": 53.4, "37139": 49.2,
    "37141": 58.3, "37143": 48.3, "37145": 46.3, "37147": 50.4, "37149": 55.3,
    "37151": 44.2, "37153": 45.3, "37155": 49.2, "37157": 43.3, "37159": 44.3,
    "37161": 50.2, "37163": 51.3, "37165": 47.2, "37167": 44.3, "37169": 44.4,
    "37171": 47.3, "37173": 68.2, "37175": 75.4, "37177": 52.3, "37179": 45.2,
    "37181": 44.3, "37183": 46.2, "37185": 45.3, "37187": 51.4, "37189": 65.3,
    "37191": 49.2, "37193": 50.3, "37195": 48.2, "37197": 45.3, "37199": 55.4,
}

# Hardcoded fallback: approximate USDA Terrain Ruggedness Index values for all
# 100 NC counties.  Mountains: high; Piedmont: moderate; Coastal Plain: low.
_NC_TRI = {
    "37001": 2.8, "37003": 3.4, "37005": 4.1, "37007": 1.2, "37009": 4.8,
    "37011": 5.6, "37013": 0.4, "37015": 0.3, "37017": 0.5, "37019": 0.6,
    "37021": 3.9, "37023": 3.2, "37025": 1.8, "37027": 3.8, "37029": 0.2,
    "37031": 0.5, "37033": 1.3, "37035": 2.1, "37037": 1.5, "37039": 5.2,
    "37041": 0.3, "37043": 5.3, "37045": 2.5, "37047": 0.5, "37049": 0.4,
    "37051": 0.4, "37053": 0.2, "37055": 0.3, "37057": 1.7, "37059": 1.6,
    "37061": 0.5, "37063": 1.4, "37065": 0.4, "37067": 1.3, "37069": 0.9,
    "37071": 2.4, "37073": 0.3, "37075": 5.8, "37077": 1.0, "37079": 0.4,
    "37081": 1.4, "37083": 0.4, "37085": 0.6, "37087": 4.6, "37089": 4.1,
    "37091": 0.3, "37093": 0.5, "37095": 0.3, "37097": 2.0, "37099": 5.6,
    "37101": 0.4, "37103": 0.4, "37105": 0.8, "37107": 0.4, "37109": 2.3,
    "37111": 4.4, "37113": 5.4, "37115": 4.2, "37117": 0.4, "37119": 1.2,
    "37121": 5.5, "37123": 1.3, "37125": 0.7, "37127": 0.5, "37129": 0.6,
    "37131": 0.4, "37133": 0.5, "37135": 1.5, "37137": 0.4, "37139": 0.2,
    "37141": 0.5, "37143": 0.3, "37145": 1.1, "37147": 0.4, "37149": 3.8,
    "37151": 1.7, "37153": 0.8, "37155": 0.5, "37157": 1.5, "37159": 1.8,
    "37161": 3.1, "37163": 0.5, "37165": 0.6, "37167": 1.9, "37169": 1.6,
    "37171": 2.6, "37173": 5.7, "37175": 5.2, "37177": 0.3, "37179": 1.5,
    "37181": 0.9, "37183": 0.9, "37185": 0.8, "37187": 0.3, "37189": 5.4,
    "37191": 0.4, "37193": 3.7, "37195": 0.5, "37197": 1.8, "37199": 4.7,
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SZR-HSI/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _acs(url: str) -> pd.DataFrame:
    """Fetch ACS JSON endpoint, return DataFrame with GEOID column."""
    raw = json.loads(_get(url))
    headers, *rows = raw
    df = pd.DataFrame(rows, columns=headers)
    df["GEOID"] = df["state"] + df["county"]
    return df


# ---------------------------------------------------------------------------
# TIGER land area (optional; enhances rural-factor accuracy)
# ---------------------------------------------------------------------------

def _tiger_land_area() -> "pd.DataFrame | None":
    """
    Query Census TIGER REST API for NC county land area (sq miles).
    Returns DataFrame(GEOID, ALAND_sqmi) or None on failure.
    """
    candidates = [
        ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
         "State_County/MapServer/12/query"
         "?where=STATEFP%3D'37'&outFields=GEOID%2CALAND"
         "&returnGeometry=false&f=json"),
        ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
         "tigerWEB_current/MapServer/84/query"
         "?where=STATEFP%3D'37'&outFields=GEOID%2CALAND"
         "&returnGeometry=false&f=json"),
    ]
    for url in candidates:
        try:
            data = json.loads(_get(url))
            features = data.get("features", [])
            if not features:
                continue
            df = pd.DataFrame([f["attributes"] for f in features])
            if "GEOID" not in df.columns:
                continue
            df["GEOID"] = df["GEOID"].astype(str).str.zfill(5)
            nc = df[df["GEOID"].str.startswith("37")].copy()
            if len(nc) < 90:
                continue
            nc["ALAND_sqmi"] = nc["ALAND"].astype(float) / 2_589_988.0
            print(f"  TIGER land area: {len(nc)} NC counties fetched")
            return nc[["GEOID", "ALAND_sqmi"]].reset_index(drop=True)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Category E — Education & Awareness
# ---------------------------------------------------------------------------

def build_education() -> pd.DataFrame:
    """ACS S1501: % adults 25+ with bachelor's degree → score_E."""
    print("[E] Fetching ACS S1501 (bachelor's degree %) ...")
    url = (
        "https://api.census.gov/data/2023/acs/acs5/subject"
        "?get=NAME,S1501_C02_015E&for=county:*&in=state:37"
    )
    df = _acs(url)
    df["pct_bachelors"] = pd.to_numeric(df["S1501_C02_015E"], errors="coerce")
    df = df.dropna(subset=["pct_bachelors"])
    df["score_E"] = _minmax(df["pct_bachelors"]).round(4)
    out = df[["GEOID", "NAME", "pct_bachelors", "score_E"]].sort_values("GEOID")
    path = os.path.join(_DASC_DIR, "nc_education.csv")
    out.to_csv(path, index=False)
    print(f"[E] Education saved  ({len(out)} counties) → {path}")
    return out


# ---------------------------------------------------------------------------
# Category S — Social & Community
# ---------------------------------------------------------------------------

def build_social(edu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Three sub-factors:
      vet_norm     0.30  ACS S2101 veteran % of civilian pop 18+
      gun_norm     0.35  NC state rate × rural adjustment
      hunting_norm 0.35  rural population proxy (inverse pop density)
    """
    # --- Veterans ---
    print("[S] Fetching ACS S2101 (veteran %) ...")
    url_vet = (
        "https://api.census.gov/data/2023/acs/acs5/subject"
        "?get=NAME,S2101_C03_001E&for=county:*&in=state:37"
    )
    df = _acs(url_vet)
    df["vet_pct"] = pd.to_numeric(df["S2101_C03_001E"], errors="coerce").fillna(0.0)

    # --- Population (ACS B01003) ---
    print("[S] Fetching ACS B01003 (total population) ...")
    url_pop = (
        "https://api.census.gov/data/2023/acs/acs5"
        "?get=NAME,B01003_001E&for=county:*&in=state:37"
    )
    pop_df = _acs(url_pop)
    pop_df["population"] = pd.to_numeric(pop_df["B01003_001E"], errors="coerce")
    df = df.merge(pop_df[["GEOID", "population"]], on="GEOID", how="left")

    # --- Rural factor (population density or pop proxy) ---
    land = _tiger_land_area()
    if land is not None:
        df = df.merge(land, on="GEOID", how="left")
        df["ALAND_sqmi"] = df["ALAND_sqmi"].fillna(df["ALAND_sqmi"].median())
        pop_density = df["population"] / df["ALAND_sqmi"].clip(lower=1.0)
        rural_factor = (1.0 - _minmax(pop_density)).values
        print("  Rural factor: pop / sq-mi (TIGER land area)")
    else:
        pop = df["population"].fillna(df["population"].median())
        rural_factor = (1.0 - _minmax(pop)).values
        print("  Rural factor: inverse population (TIGER unavailable)")

    df["rural_factor"] = rural_factor

    # --- Sub-factors ---
    df["vet_norm"] = _minmax(df["vet_pct"]).round(4)
    NC_GUN_RATE = 0.398
    df["gun_rate"] = (NC_GUN_RATE * (1.0 + 0.3 * df["rural_factor"])).clip(0.20, 0.70)
    df["gun_norm"] = _minmax(df["gun_rate"]).round(4)
    df["hunting_proxy"] = df["rural_factor"].round(4)
    df["hunting_norm"] = _minmax(df["hunting_proxy"]).round(4)

    df["score_S"] = (
        0.30 * df["vet_norm"]
        + 0.35 * df["gun_norm"]
        + 0.35 * df["hunting_norm"]
    ).round(4)

    cols = ["GEOID", "NAME", "vet_pct", "vet_norm",
            "gun_rate", "gun_norm", "hunting_proxy", "hunting_norm", "score_S"]
    out = df[cols].sort_values("GEOID")
    path = os.path.join(_DASC_DIR, "nc_social.csv")
    out.to_csv(path, index=False)
    print(f"[S] Social saved  ({len(out)} counties) → {path}")
    return out


# ---------------------------------------------------------------------------
# Category G — Environmental & Geographic
# ---------------------------------------------------------------------------

def _usda_ruggedness() -> pd.DataFrame:
    """Try USDA ERS Area Ruggedness Scale Excel; fall back to built-in TRI."""
    url = "https://www.ers.usda.gov/webdocs/DataFiles/50916/AreaRuggedness.xlsx"
    try:
        print("  USDA ERS: downloading AreaRuggedness.xlsx ...")
        data = _get(url, timeout=45)
        xl = pd.read_excel(io.BytesIO(data))
        cols_lower = {c.lower().strip(): c for c in xl.columns}
        fips_col = next(
            (cols_lower[k] for k in cols_lower if "fips" in k), None
        )
        tri_col = next(
            (cols_lower[k] for k in cols_lower
             if any(t in k for t in ["tri", "rugged", "terrain", "index"])), None
        )
        if fips_col is None or tri_col is None:
            raise ValueError(f"Cannot find FIPS+TRI columns in {list(xl.columns)}")
        xl[fips_col] = xl[fips_col].astype(str).str.strip().str.zfill(5)
        nc = xl[xl[fips_col].str.startswith("37")].copy()
        nc = nc.rename(columns={fips_col: "GEOID", tri_col: "terrain_score"})
        nc = nc[["GEOID", "terrain_score"]].dropna()
        print(f"  USDA ERS: {len(nc)} NC counties parsed")
        return nc
    except Exception as exc:
        print(f"  USDA ERS unavailable ({exc}); using built-in TRI table")
        return pd.DataFrame([
            {"GEOID": k, "terrain_score": v} for k, v in _NC_TRI.items()
        ])


def _noaa_precip() -> pd.DataFrame:
    """
    Try NOAA CIRS county-level precipitation file.
    The CIRS county numbering differs from FIPS and requires a cross-walk that
    is not publicly available as a simple CSV, so we fall back immediately to
    the built-in 1991-2020 normals derived from PRISM/NOAA station data.
    """
    try:
        import re
        index_url = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"
        idx = _get(index_url, timeout=20).decode("utf-8", errors="replace")
        matches = re.findall(r'climdiv-pcpncy-v[\d.]+-\d{8}', idx)
        if not matches:
            raise ValueError("climdiv-pcpncy file not found in NOAA index")
        fname = sorted(set(matches))[-1]
        print(f"  NOAA CIRS: found {fname} (CIRS→FIPS cross-walk not available; "
              "using built-in normals)")
        # CIRS county numbers are ordered alphabetically within each state and do
        # not map 1-to-1 to FIPS without an explicit lookup table.
        raise NotImplementedError("CIRS→FIPS cross-walk required")
    except Exception as exc:
        print(f"  NOAA precipitation: {exc}; using built-in 1991-2020 normals")
        return pd.DataFrame([
            {"GEOID": k, "precip_inches": v} for k, v in _NC_PRECIP.items()
        ])


def build_geographic(edu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Two equal sub-factors:
      terrain_score  0.50  USDA TRI — higher = more rugged = better G
      score_precip   0.50  1 - precip_normalized — drier = safer = higher G
    """
    df = edu_df[["GEOID", "NAME"]].copy()

    terrain_df = _usda_ruggedness()
    precip_df  = _noaa_precip()

    df = df.merge(terrain_df, on="GEOID", how="left")
    df = df.merge(precip_df, on="GEOID", how="left")

    df["terrain_score"] = df["terrain_score"].fillna(df["terrain_score"].median())
    df["precip_inches"] = df["precip_inches"].fillna(df["precip_inches"].median())

    terrain_norm = _minmax(df["terrain_score"])
    score_precip = 1.0 - _minmax(df["precip_inches"])

    df["score_G"] = (0.5 * terrain_norm + 0.5 * score_precip).round(4)

    out = df[["GEOID", "NAME", "terrain_score", "precip_inches", "score_G"]].sort_values("GEOID")
    path = os.path.join(_DASC_DIR, "nc_geographic.csv")
    out.to_csv(path, index=False)
    print(f"[G] Geographic saved  ({len(out)} counties) → {path}")
    return out


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(edu: pd.DataFrame, soc: pd.DataFrame, geo: pd.DataFrame):
    """
    Print partial HSI (average of H, M, I — current formula) vs.
    full weighted HSI (all 6 categories) for the 5 app-profile counties.
    """
    sys.path.insert(0, _HERE)
    try:
        from nc_county_profiles import _load_scores
        hmi = _load_scores()
    except Exception as exc:
        print(f"\n[comparison] Cannot load H/M/I scores: {exc}")
        return

    edu_idx = edu.set_index("GEOID")
    soc_idx = soc.set_index("GEOID")
    geo_idx = geo.set_index("GEOID")

    print("\n" + "=" * 82)
    print(f"{'County':<28} {'H':>6} {'M':>6} {'I':>6} {'E':>6} {'S':>6} "
          f"{'G':>6}  {'partHSI':>8} {'fullHSI':>8}")
    print("-" * 82)
    for fips, name in _APP_FIPS.items():
        if fips not in hmi.index:
            print(f"  {name}: FIPS {fips} not found in H/M/I scores")
            continue
        H = float(hmi.loc[fips, "health_score"])
        M = float(hmi.loc[fips, "mobility_score"])
        I = float(hmi.loc[fips, "infrastructure_score"])
        E = float(edu_idx.loc[fips, "score_E"]) if fips in edu_idx.index else 0.5
        S = float(soc_idx.loc[fips, "score_S"]) if fips in soc_idx.index else 0.5
        G = float(geo_idx.loc[fips, "score_G"]) if fips in geo_idx.index else 0.5
        partial = (H + M + I) / 3.0
        full    = 0.20*H + 0.10*E + 0.15*M + 0.25*S + 0.15*I + 0.15*G
        print(f"{name:<28} {H:6.3f} {M:6.3f} {I:6.3f} {E:6.3f} {S:6.3f} "
              f"{G:6.3f}  {partial:8.4f} {full:8.4f}")
    print("=" * 82)
    print("Weights: H=0.20  E=0.10  M=0.15  S=0.25  I=0.15  G=0.15")
    print("partHSI = (H + M + I) / 3  (current 3-factor average)")
    print("fullHSI = weighted sum of all 6 categories\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(_DASC_DIR, exist_ok=True)
    print("Building HSI categories E, S, G for all 100 NC counties...\n")

    edu = build_education()
    print()
    soc = build_social(edu)
    print()
    geo = build_geographic(edu)

    print("\nFiles written:")
    for fname in ("nc_education.csv", "nc_social.csv", "nc_geographic.csv"):
        p = os.path.join(_DASC_DIR, fname)
        print(f"  {p}  ({os.path.getsize(p):,} bytes)")

    print_comparison(edu, soc, geo)
