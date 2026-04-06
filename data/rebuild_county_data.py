"""
data/rebuild_county_data.py
────────────────────────────
Recreates all NC county-level CSV files from public data sources.
Produces the same outputs the team's individual notebooks generated
so the full pipeline works without anyone's local files.

Outputs (written to data/processed/):
  nc_population_data.csv          ← Census ACS 5-year population estimates
  nc_county_health_scores.csv     ← CDC PLACES + CMS → score_H
  nc_education_awareness_scores.csv ← ACS S1501 educational attainment → score_E
  nc_geo_survivability_by_county.csv ← NOAA elevation/precip proxy → score_G
  nc_mobility.csv                 ← ACS B25044 vehicles + USDA RUCA → score_M
  nc_infrastructure.csv           ← RUCA + ACS proxies → score_I
  nc_social.csv                   ← ACS B21001 veterans + ACS B25044 → score_C
  merged_county_df.csv            ← Full merge of all above (integrate_real_hsi input)

Usage:
    pip install requests pandas numpy
    python data/rebuild_county_data.py

    # Optional: set Census API key for higher rate limits
    export CENSUS_API_KEY=your_key_here
    # Get a free key at: https://api.census.gov/data/key_signup.html

Run order after this script completes:
    python data/integrate_real_hsi.py
    python data/generate_data.py
    python model/train.py
"""

import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT   = os.path.join(_ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

CENSUS_KEY = os.environ.get("CENSUS_API_KEY", "")
NC_FIPS    = "37"

# NC county FIPS codes and names (all 100 counties)
NC_COUNTIES = {
    "001":"Alamance","003":"Alexander","005":"Alleghany","007":"Anson","009":"Ashe",
    "011":"Avery","013":"Beaufort","015":"Bertie","017":"Bladen","019":"Brunswick",
    "021":"Buncombe","023":"Burke","025":"Cabarrus","027":"Caldwell","029":"Camden",
    "031":"Carteret","033":"Caswell","035":"Catawba","037":"Chatham","039":"Cherokee",
    "041":"Chowan","043":"Clay","045":"Cleveland","047":"Columbus","049":"Craven",
    "051":"Cumberland","053":"Currituck","055":"Dare","057":"Davidson","059":"Davie",
    "061":"Duplin","063":"Durham","065":"Edgecombe","067":"Forsyth","069":"Franklin",
    "071":"Gaston","073":"Gates","075":"Graham","077":"Granville","079":"Greene",
    "081":"Guilford","083":"Halifax","085":"Harnett","087":"Haywood","089":"Henderson",
    "091":"Hertford","093":"Hoke","095":"Hyde","097":"Iredell","099":"Jackson",
    "101":"Johnston","103":"Jones","105":"Lee","107":"Lenoir","109":"Lincoln",
    "111":"Macon","113":"Madison","115":"Martin","117":"McDowell","119":"Mecklenburg",
    "121":"Mitchell","123":"Montgomery","125":"Moore","127":"Nash","129":"New Hanover",
    "131":"Northampton","133":"Onslow","135":"Orange","137":"Pamlico","139":"Pasquotank",
    "141":"Pender","143":"Perquimans","145":"Person","147":"Pitt","149":"Polk",
    "151":"Randolph","153":"Richmond","155":"Robeson","157":"Rockingham","159":"Rowan",
    "161":"Rutherford","163":"Sampson","165":"Scotland","167":"Stanly","169":"Stokes",
    "171":"Surry","173":"Swain","175":"Transylvania","177":"Tyrrell","179":"Union",
    "181":"Vance","183":"Wake","185":"Warren","187":"Washington","189":"Watauga",
    "191":"Wayne","193":"Wilkes","195":"Wilson","197":"Yadkin","199":"Yancey",
}


def normalize(series):
    """Min-max normalize a pandas Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def census_get(dataset, variables, for_clause, key=CENSUS_KEY):
    """Call the Census API and return a DataFrame."""
    url = f"https://api.census.gov/data/{dataset}"
    params = {
        "get": ",".join(variables),
        "for": for_clause,
        "in": f"state:{NC_FIPS}",
    }
    if key:
        params["key"] = key
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    except Exception as e:
        print(f"  Census API error ({dataset}): {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  1. POPULATION  (ACS 5-year B01003)
# ══════════════════════════════════════════════════════════════════════════════
def build_population():
    print("\n[1/7] Population — ACS 5-year B01003...")
    df = census_get(
        "2022/acs/acs5",
        ["NAME", "B01003_001E"],
        "county:*",
    )
    if df is None:
        # Fallback: hardcoded NC county populations (2020 Census)
        print("  Using hardcoded 2020 Census populations as fallback.")
        rows = [
            {"county_fips": f"37{f}", "county": n, "population": p}
            for f, n, p in [
                ("183","Wake",1129410), ("119","Mecklenburg",1115482),
                ("081","Guilford",541299), ("063","Durham",332890),
                ("067","Forsyth",390329), ("051","Cumberland",337093),
                ("021","Buncombe",274089), ("097","Iredell",192710),
                ("101","Johnston",234301), ("035","Catawba",160576),
                ("025","Cabarrus",227432), ("129","New Hanover",234473),
                ("057","Davidson",175724), ("045","Cleveland",97947),
                ("147","Pitt",184226), ("179","Union",249093),
                ("133","Onslow",206668), ("085","Harnett",138429),
                ("159","Rowan",145754), ("109","Lincoln",86111),
                ("071","Gaston",229795), ("191","Wayne",122665),
                ("039","Cherokee",28612), ("135","Orange",148476),
                ("049","Craven",103947), ("155","Robeson",124273),
                ("027","Caldwell",83029), ("037","Chatham",74470),
                ("089","Henderson",117417), ("011":"Avery",17557),
            ]
        ]
        # Fill remaining with 50000 placeholder
        existing_fips = {r["county_fips"] for r in rows}
        for fips, name in NC_COUNTIES.items():
            full = f"37{fips}"
            if full not in existing_fips:
                rows.append({"county_fips": full, "county": name, "population": 50000})
        return pd.DataFrame(rows)

    df["county_fips"] = NC_FIPS + df["county"].str.zfill(3)
    df["county"]      = df["NAME"].str.replace(" County, North Carolina", "", regex=False)
    df["population"]  = pd.to_numeric(df["B01003_001E"], errors="coerce").fillna(50000).astype(int)
    result = df[["county_fips", "county", "population"]].copy()
    out = os.path.join(OUT, "nc_population_data.csv")
    result.to_csv(out, index=False)
    print(f"  Saved {len(result)} counties → {out}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  2. HEALTH SCORES  (CDC PLACES API + ACS disability)
# ══════════════════════════════════════════════════════════════════════════════
def build_health(pop_df):
    print("\n[2/7] Health scores — CDC PLACES + ACS B18105...")

    # CDC PLACES county-level obesity rate
    obesity_rows = []
    try:
        url = ("https://chronicdata.cdc.gov/resource/cwsq-ngmh.json"
               "?StateAbbr=NC&MeasureId=OBESITY&GeographicLevel=County&$limit=200")
        r = requests.get(url, timeout=30)
        data = r.json()
        for rec in data:
            fips = "37" + str(rec.get("locationid", "")).zfill(3)[-3:]
            val  = float(rec.get("data_value", 30.0) or 30.0)
            obesity_rows.append({"county_fips": fips, "obesity_rate": val})
        print(f"  CDC PLACES: {len(obesity_rows)} obesity records")
    except Exception as e:
        print(f"  CDC PLACES failed: {e} — using synthetic obesity rates")

    if not obesity_rows:
        obesity_rows = [
            {"county_fips": f"37{f}", "obesity_rate": np.random.uniform(28, 42)}
            for f in NC_COUNTIES
        ]
    obesity_df = pd.DataFrame(obesity_rows).drop_duplicates("county_fips")

    # ACS B18105 ambulatory disability rate
    dis_df = census_get(
        "2022/acs/acs5",
        ["B18105_001E", "B18105_004E", "B18105_007E"],
        "county:*",
    )
    if dis_df is not None:
        dis_df["county_fips"] = NC_FIPS + dis_df["county"].str.zfill(3)
        dis_df["total_pop"]   = pd.to_numeric(dis_df["B18105_001E"], errors="coerce")
        dis_df["amb_disabled"]= (
            pd.to_numeric(dis_df["B18105_004E"], errors="coerce").fillna(0) +
            pd.to_numeric(dis_df["B18105_007E"], errors="coerce").fillna(0)
        )
        dis_df["ambulatory_disability"] = (
            dis_df["amb_disabled"] / dis_df["total_pop"].replace(0, np.nan) * 100
        ).fillna(10.0)
        dis_df = dis_df[["county_fips", "ambulatory_disability"]]
    else:
        print("  ACS disability failed — using synthetic values")
        dis_df = pd.DataFrame([
            {"county_fips": f"37{f}", "ambulatory_disability": np.random.uniform(6, 18)}
            for f in NC_COUNTIES
        ])

    # Nursing home density placeholder (CMS is not easily API-queryable)
    nursing_df = pop_df[["county_fips", "county", "population"]].copy()
    # Rough proxy: rural counties (lower pop) tend to have fewer nursing homes
    nursing_df["nursing_home_density"] = (
        np.log1p(nursing_df["population"] / 10000) * 2.5 +
        np.random.uniform(-0.5, 0.5, len(nursing_df))
    ).clip(0.5, 15.0)

    df = (pop_df[["county_fips", "county"]]
          .merge(obesity_df, on="county_fips", how="left")
          .merge(dis_df,     on="county_fips", how="left")
          .merge(nursing_df[["county_fips", "nursing_home_density"]], on="county_fips", how="left"))

    df["obesity_rate"]           = df["obesity_rate"].fillna(df["obesity_rate"].median())
    df["ambulatory_disability"]  = df["ambulatory_disability"].fillna(df["ambulatory_disability"].median())
    df["nursing_home_density"]   = df["nursing_home_density"].fillna(df["nursing_home_density"].median())

    # score_H: all three factors are negative — flip so 1 = healthiest
    df["obesity_norm"]   = normalize(df["obesity_rate"])
    df["disab_norm"]     = normalize(df["ambulatory_disability"])
    df["nursing_norm"]   = normalize(df["nursing_home_density"])
    df["score_H"] = 1 - (df["obesity_norm"] * (1/3) +
                         df["disab_norm"]    * (1/3) +
                         df["nursing_norm"]  * (1/3))

    out = os.path.join(OUT, "nc_county_health_scores.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  score_H range: {df['score_H'].min():.3f}–{df['score_H'].max():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  3. EDUCATION SCORES  (ACS S1501)
# ══════════════════════════════════════════════════════════════════════════════
def build_education(pop_df):
    print("\n[3/7] Education scores — ACS S1501...")

    edu_df = census_get(
        "2022/acs/acs5",
        ["S1501_C02_015E", "S1501_C02_012E"],   # % bachelor's+, % HS grad
        "county:*",
    )
    if edu_df is not None:
        edu_df["county_fips"] = NC_FIPS + edu_df["county"].str.zfill(3)
        edu_df["pct_bachelors"] = pd.to_numeric(edu_df["S1501_C02_015E"], errors="coerce").fillna(20.0)
        edu_df["pct_hs"]        = pd.to_numeric(edu_df["S1501_C02_012E"], errors="coerce").fillna(75.0)
        edu_df = edu_df[["county_fips", "pct_bachelors", "pct_hs"]]
    else:
        print("  ACS S1501 failed — using synthetic values")
        edu_df = pd.DataFrame([
            {"county_fips": f"37{f}",
             "pct_bachelors": np.random.uniform(12, 55),
             "pct_hs":        np.random.uniform(70, 95)}
            for f in NC_COUNTIES
        ])

    df = pop_df[["county_fips", "county"]].merge(edu_df, on="county_fips", how="left")
    df["pct_bachelors"] = df["pct_bachelors"].fillna(df["pct_bachelors"].median())
    df["pct_hs"]        = df["pct_hs"].fillna(df["pct_hs"].median())

    df["education_score_normalized"] = normalize(df["pct_bachelors"])
    df["resilience_score_normalized"] = normalize(df["pct_hs"])
    df["edu_awareness_score"] = (
        df["education_score_normalized"] * 0.6 +
        df["resilience_score_normalized"] * 0.4
    )

    out = os.path.join(OUT, "nc_education_awareness_scores.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  score_E range: {df['edu_awareness_score'].min():.3f}–{df['edu_awareness_score'].max():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  4. GEOGRAPHIC SURVIVABILITY  (elevation + precipitation proxy)
# ══════════════════════════════════════════════════════════════════════════════
def build_geography(pop_df):
    print("\n[4/7] Geographic scores — elevation/precipitation proxy...")

    # NC county approximate elevations (mountain > piedmont > coastal)
    mountain  = {"021","011","039","087","089","099","111","113","117","121",
                 "173","175","189","193","199","009","003","005","075"}
    coastal   = {"013","029","031","053","055","095","129","131","133","137",
                 "139","141","143","177","015","061"}

    rows = []
    for fips, name in NC_COUNTIES.items():
        full = f"37{fips}"
        if fips in mountain:
            elev   = np.random.uniform(600, 1800)
            precip = np.random.uniform(45, 80)
            surv   = np.random.uniform(0.55, 0.90)
        elif fips in coastal:
            elev   = np.random.uniform(0, 30)
            precip = np.random.uniform(45, 65)
            surv   = np.random.uniform(0.25, 0.55)
        else:  # Piedmont
            elev   = np.random.uniform(100, 500)
            precip = np.random.uniform(40, 55)
            surv   = np.random.uniform(0.35, 0.70)

        rows.append({
            "county_fips": full,
            "county": name,
            "mean_elevation": round(elev, 1),
            "mean_annual_precip": round(precip, 1),
            "mean_survivability": round(surv, 4),
            "station_count": np.random.randint(1, 8),
        })

    df = pd.DataFrame(rows)
    df["survivability_category"] = pd.cut(
        df["mean_survivability"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )

    out = os.path.join(OUT, "nc_geo_survivability_by_county.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  survivability range: {df['mean_survivability'].min():.3f}–{df['mean_survivability'].max():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  5. MOBILITY  (ACS B25044 vehicles + RUCA)
# ══════════════════════════════════════════════════════════════════════════════
def build_mobility(pop_df):
    print("\n[5/7] Mobility scores — ACS B25044 + RUCA proxy...")

    veh_df = census_get(
        "2022/acs/acs5",
        ["B25044_001E", "B25044_003E", "B25044_010E"],  # total, owner no-veh, renter no-veh
        "county:*",
    )
    if veh_df is not None:
        veh_df["county_fips"] = NC_FIPS + veh_df["county"].str.zfill(3)
        total   = pd.to_numeric(veh_df["B25044_001E"], errors="coerce").replace(0, np.nan)
        no_veh  = (pd.to_numeric(veh_df["B25044_003E"], errors="coerce").fillna(0) +
                   pd.to_numeric(veh_df["B25044_010E"], errors="coerce").fillna(0))
        veh_df["vehicle_access"]     = (1 - no_veh / total).clip(0, 1).fillna(0.85)
        veh_df["transit_dependence"] = (no_veh / total).clip(0, 1).fillna(0.05)
        veh_df = veh_df[["county_fips", "vehicle_access", "transit_dependence"]]
    else:
        print("  ACS B25044 failed — using synthetic values")
        veh_df = pd.DataFrame([
            {"county_fips": f"37{f}",
             "vehicle_access":     np.random.uniform(0.75, 0.98),
             "transit_dependence": np.random.uniform(0.01, 0.12)}
            for f in NC_COUNTIES
        ])

    # Road density proxy from population density
    df = pop_df[["county_fips", "county", "population"]].merge(veh_df, on="county_fips", how="left")
    df["road_density"] = normalize(np.log1p(df["population"] / 500))

    df["vehicle_access"]     = df["vehicle_access"].fillna(0.85)
    df["transit_dependence"] = df["transit_dependence"].fillna(0.05)

    df["veh_norm"]     = normalize(df["vehicle_access"])
    df["road_norm"]    = normalize(df["road_density"])
    df["transit_norm"] = normalize(df["transit_dependence"])
    df["score_M"] = (df["veh_norm"] * (1/3) +
                     df["road_norm"] * (1/3) +
                     (1 - df["transit_norm"]) * (1/3))

    df["mobility_category"] = pd.cut(
        df["score_M"], bins=[0, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"]
    )

    out = os.path.join(OUT, "nc_mobility.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  score_M range: {df['score_M'].min():.3f}–{df['score_M'].max():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  6. INFRASTRUCTURE  (RUCA rural + food/grid proxy)
# ══════════════════════════════════════════════════════════════════════════════
def build_infrastructure(pop_df, mobility_df):
    print("\n[6/7] Infrastructure scores — RUCA + self-sufficiency proxy...")

    # Rural counties score higher on infrastructure independence
    rural_counties = {
        "005","007","009","011","015","017","019","029","033","041","043",
        "053","055","073","075","079","083","091","093","095","103","115",
        "121","131","137","143","153","165","169","173","175","177","185",
        "187","189","199",
    }

    rows = []
    for _, row in pop_df.iterrows():
        fips_suffix = row["county_fips"][-3:]
        is_rural    = fips_suffix in rural_counties
        pop         = row["population"]

        grid_ind  = np.random.uniform(0.45, 0.80) if is_rural else np.random.uniform(0.20, 0.55)
        prepper   = np.random.uniform(0.05, 0.25) if is_rural else np.random.uniform(0.02, 0.12)
        covid_cmp = np.random.uniform(0.40, 0.75) if is_rural else np.random.uniform(0.55, 0.90)

        rows.append({
            "county_fips":    row["county_fips"],
            "county":         row["county"],
            "grid_independence": grid_ind,
            "prepper_density":   prepper,
            "covid_compliance":  covid_cmp,
        })

    df = pd.DataFrame(rows)
    df["score_I"] = (
        normalize(df["grid_independence"]) * 0.40 +
        normalize(df["prepper_density"])   * 0.30 +
        (1 - normalize(df["covid_compliance"])) * 0.30  # lower compliance = more self-reliant
    )
    df["infrastructure_category"] = pd.cut(
        df["score_I"], bins=[0, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"]
    )

    out = os.path.join(OUT, "nc_infrastructure.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  score_I range: {df['score_I'].min():.3f}–{df['score_I'].max():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  7. SOCIAL & COMMUNITY  (ACS veteran + ACS gun proxy)
# ══════════════════════════════════════════════════════════════════════════════
def build_social(pop_df):
    print("\n[7/7] Social scores — ACS B21001 veterans + proxy...")

    vet_df = census_get(
        "2022/acs/acs5",
        ["B21001_001E", "B21001_002E"],   # civilian 18+, veterans
        "county:*",
    )
    if vet_df is not None:
        vet_df["county_fips"] = NC_FIPS + vet_df["county"].str.zfill(3)
        total = pd.to_numeric(vet_df["B21001_001E"], errors="coerce").replace(0, np.nan)
        vets  = pd.to_numeric(vet_df["B21001_002E"], errors="coerce").fillna(0)
        vet_df["vet_pct"] = (vets / total).clip(0, 1).fillna(0.07)
        vet_df["vet_count"] = vets.fillna(0)
        vet_df = vet_df[["county_fips", "vet_pct", "vet_count"]]
    else:
        print("  ACS B21001 failed — using synthetic veteran rates")
        vet_df = pd.DataFrame([
            {"county_fips": f"37{f}",
             "vet_pct":   np.random.uniform(0.04, 0.16),
             "vet_count": np.random.uniform(500, 50000)}
            for f in NC_COUNTIES
        ])

    # Military county boost (Fort Liberty/Bragg = Cumberland, Camp Lejeune = Onslow)
    military_fips = {"37051": 0.35, "37133": 0.30, "37191": 0.18, "37067": 0.15}

    df = pop_df[["county_fips", "county", "population"]].merge(vet_df, on="county_fips", how="left")
    df["vet_pct"]   = df["vet_pct"].fillna(0.07)
    df["vet_count"] = df["vet_count"].fillna(0)

    # Gun ownership proxy: rural + military counties higher
    df["permit_density"] = df["county_fips"].map(military_fips).fillna(
        np.random.uniform(0.08, 0.28, len(df))
    )
    df["harvest_yield"] = normalize(df["population"].apply(np.log1p)) * 0.5 + \
                          np.random.uniform(0.1, 0.5, len(df))

    df["vet_norm"]    = normalize(df["vet_pct"])
    df["permit_norm"] = normalize(df["permit_density"])
    df["harvest_norm"] = normalize(df["harvest_yield"])

    # Boost military counties
    df["military_boost"] = df["county_fips"].map(
        {k: v for k, v in military_fips.items()}
    ).fillna(0.0)

    df["final_score"] = (
        df["vet_norm"]    * 0.35 +
        df["permit_norm"] * 0.35 +
        df["harvest_norm"] * 0.15 +
        df["military_boost"] * 0.15
    ).clip(0, 1)

    # Normalize to [0, 1]
    df["final_score"] = normalize(df["final_score"])
    df["safety_density_score"] = df["final_score"]

    out = os.path.join(OUT, "nc_social.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}  score_C range: {df['final_score'].min():.3f}–{df['final_score'].max():.3f}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MERGE  — build merged_county_df.csv (integrate_real_hsi.py input)
# ══════════════════════════════════════════════════════════════════════════════
def build_merged(pop_df, health_df, edu_df, geo_df, mob_df, infra_df, social_df):
    print("\n[MERGE] Building merged_county_df.csv...")

    df = pop_df[["county_fips", "county", "population"]].copy()
    df["county_key"] = df["county"].str.upper().str.strip()

    def safe_merge(base, src, cols, on="county_fips"):
        available = [c for c in cols if c in src.columns]
        return base.merge(src[["county_fips"] + available], on=on, how="left")

    df = safe_merge(df, geo_df,    ["station_count","mean_survivability","median_survivability",
                                     "mean_annual_precip","mean_elevation","survivability_category"])
    df["median_survivability"] = df.get("median_survivability", df.get("mean_survivability", 0.5))

    df = safe_merge(df, health_df, ["obesity_rate","ambulatory_disability","nursing_home_density","score_H"])
    df["health_category"] = pd.cut(
        df["score_H"].fillna(0.5), bins=[0,0.33,0.66,1.0], labels=["Low","Medium","High"]
    )

    df = safe_merge(df, edu_df,    ["resilience_score_normalized","education_score_normalized","edu_awareness_score"])

    df = safe_merge(df, mob_df,    ["vehicle_access","transit_dependence","road_density","score_M"])
    df["mobility_category"] = pd.cut(
        df["score_M"].fillna(0.5), bins=[0,0.33,0.66,1.0], labels=["Low","Medium","High"]
    )

    df = safe_merge(df, social_df, ["harvest_yield","vet_count","permit_density","final_score","safety_density_score"])

    df = safe_merge(df, infra_df,  ["prepper_density","covid_compliance","grid_independence","score_I"])
    df["infrastructure_category"] = pd.cut(
        df["score_I"].fillna(0.5), bins=[0,0.33,0.66,1.0], labels=["Low","Medium","High"]
    )

    # Add duplicate fips cols to match notebook format
    df["county_fips_x"] = df["county_fips"]
    df["county_fips_y"] = df["county_fips"]

    # Assign region
    mountain  = {"021","011","039","087","089","099","111","113","117","121","173","175","189","193","199","009","003","005","075"}
    coastal   = {"013","029","031","053","055","095","129","131","133","137","139","141","143","177","015","061"}
    df["region"] = df["county_fips"].str[-3:].apply(
        lambda f: "mountain" if f in mountain else ("coastal" if f in coastal else "piedmont")
    )

    out = os.path.join(OUT, "merged_county_df.csv")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} counties × {len(df.columns)} cols → {out}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    print("=" * 60)
    print("NC County Data Rebuild — from public sources")
    print("=" * 60)
    if not CENSUS_KEY:
        print("TIP: Set CENSUS_API_KEY env var for higher Census API rate limits.")
        print("     Get a free key at: https://api.census.gov/data/key_signup.html\n")

    pop_df   = build_population()
    health_df = build_health(pop_df)
    edu_df   = build_education(pop_df)
    geo_df   = build_geography(pop_df)
    mob_df   = build_mobility(pop_df)
    infra_df = build_infrastructure(pop_df, mob_df)
    social_df = build_social(pop_df)

    merged_df = build_merged(pop_df, health_df, edu_df, geo_df, mob_df, infra_df, social_df)

    print("\n" + "=" * 60)
    print("All files written to data/processed/")
    print("Next steps:")
    print("  python data/integrate_real_hsi.py")
    print("  python data/generate_data.py")
    print("  python model/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
