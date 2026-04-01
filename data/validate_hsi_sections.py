#!/usr/bin/env python3
"""
validate_hsi_sections.py
========================
Validates all six HSI section CSVs, prints human-readable summaries
(score ranges, top/bottom counties, SZR interpretation), runs a full
merge check, and prints a side-by-side HSI table for the 5 app counties.

Usage:
    python data/validate_hsi_sections.py
"""

import os
import sys
import warnings

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DASC = os.path.join(_HERE, "dasc6010")

# ---------------------------------------------------------------------------
# NC county FIPS → short name (all 100 counties)
# ---------------------------------------------------------------------------
_NAMES = {
    "37001": "Alamance",    "37003": "Alexander",   "37005": "Alleghany",
    "37007": "Anson",       "37009": "Ashe",         "37011": "Avery",
    "37013": "Beaufort",    "37015": "Bertie",       "37017": "Bladen",
    "37019": "Brunswick",   "37021": "Buncombe",     "37023": "Burke",
    "37025": "Cabarrus",    "37027": "Caldwell",     "37029": "Camden",
    "37031": "Carteret",    "37033": "Caswell",      "37035": "Catawba",
    "37037": "Chatham",     "37039": "Cherokee",     "37041": "Chowan",
    "37043": "Clay",        "37045": "Cleveland",    "37047": "Columbus",
    "37049": "Craven",      "37051": "Cumberland",   "37053": "Currituck",
    "37055": "Dare",        "37057": "Davidson",     "37059": "Davie",
    "37061": "Duplin",      "37063": "Durham",       "37065": "Edgecombe",
    "37067": "Forsyth",     "37069": "Franklin",     "37071": "Gaston",
    "37073": "Gates",       "37075": "Graham",       "37077": "Granville",
    "37079": "Greene",      "37081": "Guilford",     "37083": "Halifax",
    "37085": "Harnett",     "37087": "Haywood",      "37089": "Henderson",
    "37091": "Hertford",    "37093": "Hoke",         "37095": "Hyde",
    "37097": "Iredell",     "37099": "Jackson",      "37101": "Johnston",
    "37103": "Jones",       "37105": "Lee",          "37107": "Lenoir",
    "37109": "Lincoln",     "37111": "McDowell",     "37113": "Macon",
    "37115": "Madison",     "37117": "Martin",       "37119": "Mecklenburg",
    "37121": "Mitchell",    "37123": "Montgomery",   "37125": "Moore",
    "37127": "Nash",        "37129": "New Hanover",  "37131": "Northampton",
    "37133": "Onslow",      "37135": "Orange",       "37137": "Pamlico",
    "37139": "Pasquotank",  "37141": "Pender",       "37143": "Perquimans",
    "37145": "Person",      "37147": "Pitt",         "37149": "Polk",
    "37151": "Randolph",    "37153": "Richmond",     "37155": "Robeson",
    "37157": "Rockingham",  "37159": "Rowan",        "37161": "Rutherford",
    "37163": "Sampson",     "37165": "Scotland",     "37167": "Stanly",
    "37169": "Stokes",      "37171": "Surry",        "37173": "Swain",
    "37175": "Transylvania","37177": "Tyrrell",      "37179": "Union",
    "37181": "Vance",       "37183": "Wake",         "37185": "Warren",
    "37187": "Washington",  "37189": "Watauga",      "37191": "Wayne",
    "37193": "Wilkes",      "37195": "Wilson",       "37197": "Yadkin",
    "37199": "Yancey",
}

_APP_FIPS = {
    "37119": "Mecklenburg (Charlotte)",
    "37183": "Wake (Raleigh)",
    "37055": "Dare (Outer Banks)",
    "37177": "Tyrrell (Rural)",
    "37155": "Robeson (Mixed)",
}

_PASS = "\033[32mPASS\033[0m"
_FAIL = "\033[31mFAIL\033[0m"
_WARN = "\033[33mWARN\033[0m"

_errors = []


def _check(condition: bool, msg_pass: str, msg_fail: str):
    tag = _PASS if condition else _FAIL
    print(f"  [{tag}] {msg_pass if condition else msg_fail}")
    if not condition:
        _errors.append(msg_fail)


def _name(fips: str) -> str:
    return _NAMES.get(fips, fips)


def _top_bottom(county_series: pd.Series, n: int = 3):
    """Return (top_n names, bottom_n names) for a county-indexed Series."""
    s = county_series.dropna().sort_values(ascending=False)
    top    = [_name(f) for f in s.head(n).index]
    bottom = [_name(f) for f in s.tail(n).index]
    return top, bottom


def _load_tract(fname: str) -> "pd.DataFrame | None":
    path = os.path.join(_DASC, fname)
    if not os.path.exists(path):
        print(f"  [{_FAIL}] File not found: {path}")
        _errors.append(f"Missing {fname}")
        return None
    df = pd.read_csv(path, dtype={"GEOID": str})
    df["county_fips"] = df["GEOID"].str[:5]
    return df


def _load_county(fname: str) -> "pd.DataFrame | None":
    path = os.path.join(_DASC, fname)
    if not os.path.exists(path):
        print(f"  [{_FAIL}] File not found: {path}")
        _errors.append(f"Missing {fname}")
        return None
    df = pd.read_csv(path, dtype={"GEOID": str})
    df = df.rename(columns={"GEOID": "county_fips"})
    return df


def _divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Section H — nc_health.csv (tract-level)
# ---------------------------------------------------------------------------

def validate_H() -> "pd.Series | None":
    _divider("Section H — nc_health.csv")
    df = _load_tract("nc_health.csv")
    if df is None:
        return None

    req = ["GEOID", "score_H"]
    for col in req:
        _check(col in df.columns, f"Column '{col}' present", f"Missing column '{col}'")
    _check(df["score_H"].between(0, 1).all(),
           "score_H values in [0, 1]",
           f"score_H out of range: {df['score_H'].min():.3f}–{df['score_H'].max():.3f}")
    _check(not df["score_H"].isna().any(),
           "No NaN in score_H", "NaN values found in score_H")

    county_h = df.groupby("county_fips")["score_H"].mean()
    top, bot = _top_bottom(county_h)

    print(f"\n  Section H Summary:")
    print(f"    Tracts: {len(df):,} across {df['county_fips'].nunique()} counties")
    print(f"    score_H range: {county_h.min():.2f} – {county_h.max():.2f}  "
          f"(mean {county_h.mean():.2f})")
    print(f"    Best counties  (highest health):  {', '.join(top)}")
    print(f"    Worst counties (lowest health):   {', '.join(bot)}")
    print(f"    SZR interpretation: Higher score_H = healthier, more mobile population")
    print(f"      = harder for zombies to catch = lower effective beta")
    return county_h


# ---------------------------------------------------------------------------
# Section M — nc_mobility.csv (tract-level)
# ---------------------------------------------------------------------------

def validate_M() -> "pd.Series | None":
    _divider("Section M — nc_mobility.csv")
    df = _load_tract("nc_mobility.csv")
    if df is None:
        return None

    req = ["GEOID", "score_M"]
    for col in req:
        _check(col in df.columns, f"Column '{col}' present", f"Missing column '{col}'")

    has_td = "transit_dependence" in df.columns
    _check(has_td,
           "transit_dependence column present (used for mobility_score)",
           "transit_dependence missing — mobility_score will fall back to 1-score_M")

    _check(df["score_M"].between(0, 1).all(),
           "score_M values in [0, 1]",
           f"score_M out of range: {df['score_M'].min():.3f}–{df['score_M'].max():.3f}")

    county_m  = df.groupby("county_fips")["score_M"].mean()
    top_m, bot_m = _top_bottom(county_m)

    print(f"\n  Section M Summary:")
    print(f"    Tracts: {len(df):,} across {df['county_fips'].nunique()} counties")
    print(f"    score_M range: {county_m.min():.2f} – {county_m.max():.2f}  "
          f"(mean {county_m.mean():.2f})")

    if has_td:
        county_td = df.groupby("county_fips")["transit_dependence"].mean()
        top_td, bot_td = _top_bottom(county_td)
        print(f"    transit_dependence range: "
              f"{county_td.min():.1f}% – {county_td.max():.1f}%  "
              f"(used as spread proxy)")
        print(f"    Most isolated (high transit dep.): {', '.join(top_td)}")
        print(f"    Most mobile   (low transit dep.):  {', '.join(bot_td)}")
    else:
        print(f"    Most mobile counties (high score_M):  {', '.join(top_m)}")
        print(f"    Least mobile counties (low score_M):  {', '.join(bot_m)}")

    print(f"    SZR interpretation: mobility_score = normalized transit_dependence;")
    print(f"      higher = more constrained movement = higher zombie spread risk")
    return county_m


# ---------------------------------------------------------------------------
# Section I — nc_infrastructure.csv (tract-level)
# ---------------------------------------------------------------------------

def validate_I() -> "pd.Series | None":
    _divider("Section I — nc_infrastructure.csv")
    df = _load_tract("nc_infrastructure.csv")
    if df is None:
        return None

    req = ["GEOID", "score_I"]
    for col in req:
        _check(col in df.columns, f"Column '{col}' present", f"Missing column '{col}'")
    _check(df["score_I"].between(0, 1).all(),
           "score_I values in [0, 1]",
           f"score_I out of range: {df['score_I'].min():.3f}–{df['score_I'].max():.3f}")
    _check(not df["score_I"].isna().any(),
           "No NaN in score_I", "NaN values found in score_I")

    county_i = df.groupby("county_fips")["score_I"].mean()
    top, bot = _top_bottom(county_i)

    print(f"\n  Section I Summary:")
    print(f"    Tracts: {len(df):,} across {df['county_fips'].nunique()} counties")
    print(f"    score_I range: {county_i.min():.2f} – {county_i.max():.2f}  "
          f"(mean {county_i.mean():.2f})")
    print(f"    Best counties  (most prepared):   {', '.join(top)}")
    print(f"    Worst counties (least prepared):  {', '.join(bot)}")
    print(f"    SZR interpretation: Higher score_I = more off-grid, COVID-compliant,")
    print(f"      prepper-dense = better long-term survival capacity")
    return county_i


# ---------------------------------------------------------------------------
# Section E — nc_education.csv (county-level)
# ---------------------------------------------------------------------------

def validate_E() -> "pd.Series | None":
    _divider("Section E — nc_education.csv")
    df = _load_county("nc_education.csv")
    if df is None:
        return None

    req = ["county_fips", "pct_bachelors", "score_E"]
    for col in req:
        _check(col in df.columns, f"Column '{col}' present", f"Missing column '{col}'")

    _check(len(df) == 100,
           f"100 NC counties present ({len(df)} rows)",
           f"Expected 100 counties, got {len(df)}")
    _check(df["score_E"].between(0, 1).all(),
           "score_E values in [0, 1]",
           f"score_E out of range: {df['score_E'].min():.3f}–{df['score_E'].max():.3f}")
    _check(not df["score_E"].isna().any(),
           "No NaN in score_E", "NaN values found in score_E")

    county_e = df.set_index("county_fips")["score_E"]
    top, bot = _top_bottom(county_e)
    pct = df.set_index("county_fips")["pct_bachelors"]

    print(f"\n  Section E Summary:")
    print(f"    Counties: {len(df)}")
    print(f"    pct_bachelors range: {pct.min():.1f}% – {pct.max():.1f}%  "
          f"(mean {pct.mean():.1f}%)")
    print(f"    score_E range: {county_e.min():.2f} – {county_e.max():.2f}  "
          f"(mean {county_e.mean():.2f})")
    print(f"    Highest education:  {', '.join(top)}")
    print(f"    Lowest education:   {', '.join(bot)}")
    print(f"    SZR interpretation: Higher score_E = more educated/aware population")
    print(f"      = faster coordinated response = lower effective beta")
    return county_e


# ---------------------------------------------------------------------------
# Section S — nc_social.csv (county-level)
# ---------------------------------------------------------------------------

def validate_S() -> "pd.Series | None":
    _divider("Section S — nc_social.csv")
    df = _load_county("nc_social.csv")
    if df is None:
        return None

    req = ["county_fips", "vet_pct", "gun_rate", "hunting_proxy", "score_S"]
    for col in req:
        _check(col in df.columns, f"Column '{col}' present", f"Missing column '{col}'")

    _check(len(df) == 100,
           f"100 NC counties present ({len(df)} rows)",
           f"Expected 100 counties, got {len(df)}")
    _check(df["score_S"].between(0, 1).all(),
           "score_S values in [0, 1]",
           f"score_S out of range: {df['score_S'].min():.3f}–{df['score_S'].max():.3f}")
    _check(not df["score_S"].isna().any(),
           "No NaN in score_S", "NaN values found in score_S")

    county_s = df.set_index("county_fips")["score_S"]
    top, bot = _top_bottom(county_s)
    gun = df.set_index("county_fips")["gun_rate"]

    print(f"\n  Section S Summary:")
    print(f"    Counties: {len(df)}")
    print(f"    gun_rate range:  {gun.min():.1%} – {gun.max():.1%}  "
          f"(mean {gun.mean():.1%})")
    print(f"    score_S range:   {county_s.min():.2f} – {county_s.max():.2f}  "
          f"(mean {county_s.mean():.2f})")
    print(f"    Highest social (rural/armed): {', '.join(top)}")
    print(f"    Lowest social  (urban):       {', '.join(bot)}")
    print(f"    SZR interpretation: Higher score_S = more veterans, guns, hunters")
    print(f"      = higher kappa (kill rate) = faster zombie removal")
    return county_s


# ---------------------------------------------------------------------------
# Section G — nc_geographic.csv (county-level)
# ---------------------------------------------------------------------------

def validate_G() -> "pd.Series | None":
    _divider("Section G — nc_geographic.csv")
    df = _load_county("nc_geographic.csv")
    if df is None:
        return None

    req = ["county_fips", "terrain_score", "score_G"]
    for col in req:
        _check(col in df.columns, f"Column '{col}' present", f"Missing column '{col}'")

    _check(len(df) == 100,
           f"100 NC counties present ({len(df)} rows)",
           f"Expected 100 counties, got {len(df)}")
    _check(df["score_G"].between(0, 1).all(),
           "score_G values in [0, 1]",
           f"score_G out of range: {df['score_G'].min():.3f}–{df['score_G'].max():.3f}")
    _check(not df["score_G"].isna().any(),
           "No NaN in score_G", "NaN values found in score_G")

    county_g   = df.set_index("county_fips")["score_G"]
    terrain    = df.set_index("county_fips")["terrain_score"]
    top, bot   = _top_bottom(county_g)

    print(f"\n  Section G Summary:")
    print(f"    Counties: {len(df)}")
    print(f"    terrain_score range: {terrain.min():.2f} – {terrain.max():.2f}  "
          f"(mean {terrain.mean():.2f})")
    print(f"    score_G range:       {county_g.min():.2f} – {county_g.max():.2f}  "
          f"(mean {county_g.mean():.2f})")
    print(f"    Most rugged / defensible:  {', '.join(top)}")
    print(f"    Least rugged / exposed:    {', '.join(bot)}")
    print(f"    SZR interpretation: Higher score_G = more rugged terrain")
    print(f"      = natural movement barrier = lower effective beta for zombie spread")
    return county_g


# ---------------------------------------------------------------------------
# Merge check
# ---------------------------------------------------------------------------

def validate_merge(county_h, county_m, county_i,
                   county_e, county_s, county_g):
    _divider("Full merge check — all 6 sections")

    frames = {"H": county_h, "M": county_m, "I": county_i}
    esg_labels = {"E": county_e, "S": county_s, "G": county_g}

    missing_esg = [k for k, v in esg_labels.items() if v is None]
    present_esg = {k: v for k, v in esg_labels.items() if v is not None}
    if missing_esg:
        print(f"  [{_WARN}] ESG files missing for: {missing_esg} — partial merge only")
    frames.update(present_esg)

    if not frames:
        print(f"  [{_FAIL}] No sections loaded; cannot merge")
        _errors.append("No sections available for merge")
        return

    merged = pd.DataFrame({k: v for k, v in frames.items()})
    n_counties = len(merged)
    n_complete = merged.dropna().shape[0]

    _check(n_counties >= 95,
           f"{n_counties} counties in merged table (≥ 95)",
           f"Only {n_counties} counties after merge (expected ≥ 95)")
    _check(n_complete >= 90,
           f"{n_complete} counties have all available scores complete",
           f"Only {n_complete} fully complete rows (expected ≥ 90)")

    present_cols = list(frames.keys())
    print(f"  Sections merged: {present_cols}")
    print(f"  Counties total:  {n_counties}")
    print(f"  Fully complete:  {n_complete}")

    return merged


# ---------------------------------------------------------------------------
# 5-county HSI summary
# ---------------------------------------------------------------------------

def print_app_county_table():
    _divider("5-App-County HSI Summary")
    sys.path.insert(0, _HERE)
    try:
        from nc_county_profiles import build_hsi_overrides
        overrides = build_hsi_overrides()
    except Exception as exc:
        print(f"  [{_FAIL}] Cannot load overrides: {exc}")
        return

    avail = [k for k in ("score_E", "score_S", "score_G")
             if any(k in v for v in overrides.values())]
    has_esg = bool(avail)

    hdr_cols = ["H", "M", "I"] + (["E", "S", "G"] if has_esg else []) + ["fullHSI"]
    fmt = f"  {{:<28}}" + "  {:>6}" * len(hdr_cols)
    print(fmt.format("County", *hdr_cols))
    print("  " + "-" * (28 + 9 * len(hdr_cols)))

    for fips, label in _APP_FIPS.items():
        if label not in overrides:
            print(f"  {label:<28}  (not in overrides)")
            continue
        v = overrides[label]
        H = v["health_score"]
        M = v["mobility_score"]
        I = v["infrastructure_score"]
        E = v.get("score_E", 0.5)
        S = v.get("score_S", 0.5)
        G = v.get("score_G", 0.5)
        full = 0.20*H + 0.10*E + 0.15*M + 0.25*S + 0.15*I + 0.15*G
        vals = [H, M, I] + ([E, S, G] if has_esg else []) + [full]
        print(fmt.format(label, *[f"{x:.3f}" for x in vals]))

    print(f"\n  Weights: H=0.20  E=0.10  M=0.15  S=0.25  I=0.15  G=0.15")
    if not has_esg:
        print(f"  [{_WARN}] E/S/G unavailable — fullHSI uses 0.5 defaults")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("HSI Section Validator")
    print("=" * 60)

    county_h = validate_H()
    county_m = validate_M()
    county_i = validate_I()
    county_e = validate_E()
    county_s = validate_S()
    county_g = validate_G()

    validate_merge(county_h, county_m, county_i, county_e, county_s, county_g)
    print_app_county_table()

    print(f"\n{'='*60}")
    if _errors:
        print(f"  RESULT: {len(_errors)} error(s) found:")
        for e in _errors:
            print(f"    • {e}")
        sys.exit(1)
    else:
        print("  RESULT: All checks passed.")
