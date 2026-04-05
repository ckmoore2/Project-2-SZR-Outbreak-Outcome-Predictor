"""
fetch_category_e_extensions.py
Fetch scripts for Category E (Education) and H (Health) extensions:
  1. Ham radio operator license density — FCC ULS (public database)
  2. Agricultural occupation rate — ACS C24010
  3. Physical inactivity rate — CDC PLACES (already in repo, just not extracted)

Usage:
    python data/fetch/fetch_category_e_extensions.py [--factor all|ham|agriculture|inactivity]

Dependencies:
    pip install requests pandas
"""

import os
import io
import zipfile
import argparse
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

NC_STATE_FIPS = "37"

# Census API key — set CENSUS_API_KEY env variable or paste key here
# Get a free key at: https://api.census.gov/data/key_signup.html
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "YOUR_API_KEY_HERE")


# ══════════════════════════════════════════════════════════════════════════════
#  1. HAM RADIO LICENSES — FCC Universal Licensing System
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ham_radio():
    """
    FCC amateur radio (ham) licenses are public record, published in the
    FCC Universal Licensing System (ULS) database.

    The complete license database is available as a weekly bulk download:
      https://www.fcc.gov/uls/transactions/daily-weekly-files

    File: l_amat.zip — Amateur Radio license extract
    Key table: EN (Entity) — contains callsign, name, address, zip code
    Key table: HD (Header) — contains license status (A = active)

    Scoring (TODO — add to lennen_education.ipynb):
      Step 1: Filter to NC licensees (state = 'NC') with status = 'A' (active)
      Step 2: Geocode by zip code → map to census tract (using zcta_to_tract crosswalk)
      Step 3: ham_density = active_licenses / tract_population * 10000
      Step 4: ham_score = normalize(ham_density)
      Direction: POSITIVE (more ham operators = better emergency comms)
      Weight within E: 0.20

    Why it matters: Ham operators provide emergency communication independence
    when cellular and internet infrastructure fails — a genuine grid-down
    survival advantage. No current E sub-factor captures this.
    """
    out_dir = os.path.join(RAW_DIR, "fcc_ham")
    os.makedirs(out_dir, exist_ok=True)
    print("Fetching FCC ham radio license database...")

    # FCC ULS weekly extract — Amateur Radio (EN + HD tables)
    fcc_url = "https://data.fcc.gov/download/pub/uls/complete/l_amat.zip"
    out_zip = os.path.join(out_dir, "l_amat.zip")

    try:
        print(f"  Downloading from {fcc_url} (may take 1-2 min — ~300MB)...")
        resp = requests.get(fcc_url, timeout=120, stream=True)
        if resp.status_code == 200:
            total = 0
            with open(out_zip, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    total += len(chunk)
            print(f"  ✓ Downloaded → {out_zip} ({total//1024//1024}MB)")

            # Extract and filter to NC
            with zipfile.ZipFile(out_zip) as z:
                names = z.namelist()
                print(f"  Files in archive: {names}")

                # EN table: licensee entity info
                en_file = next((n for n in names if n.upper().startswith("EN")), None)
                hd_file = next((n for n in names if n.upper().startswith("HD")), None)

                if en_file and hd_file:
                    # FCC ULS pipe-delimited, no header — use FCC field spec
                    en_cols = ["record_type","unique_system_id","uls_file_num",
                               "ebf_number","callsign","entity_type","licensee_id",
                               "entity_name","first_name","mi","last_name",
                               "suffix","phone","fax","email","street_address",
                               "city","state","zip_code","po_box","attention_line",
                               "sgin","frn","applicant_type","application_purpose",
                               "status_code","status_date","lic_category_code",
                               "linked_unique_sys_id","linked_callsign"]
                    hd_cols = ["record_type","unique_system_id","uls_file_num",
                               "ebf_number","callsign","license_status","radio_service_code",
                               "grant_date","expired_date","cancellation_date",
                               "eligibility_rule_num","applicant_type","alien","alien_status",
                               "exclusive","effective_date","dates","buildout_date",
                               "buildout_grant","op_effective_date","assignor_other",
                               "aggregation_ind","amateur_club","trust","linked_call","vanity_relationship",
                               "previous_callsign","previous_operator","trustee","trustee_name",
                               "trustee_callsign","owner_first","owner_last","owner_middle",
                               "owner_suffix","contact_first","contact_last","contact_middle",
                               "contact_suffix","contact_phone","contact_fax","contact_email"]

                    with z.open(en_file) as f:
                        df_en = pd.read_csv(f, sep="|", names=en_cols[:len(en_cols)],
                                           dtype=str, on_bad_lines="skip")
                    with z.open(hd_file) as f:
                        df_hd = pd.read_csv(f, sep="|", names=hd_cols[:len(hd_cols)],
                                           dtype=str, on_bad_lines="skip")

                    # Filter: NC active amateur licenses
                    nc_en = df_en[df_en["state"] == "NC"].copy()
                    active_hd = df_hd[df_hd["license_status"] == "A"][["unique_system_id","license_status"]]
                    nc_active = nc_en.merge(active_hd, on="unique_system_id", how="inner")

                    nc_out = os.path.join(out_dir, "nc_ham_licenses_active.csv")
                    nc_active[["callsign","entity_name","city","state","zip_code"]].to_csv(
                        nc_out, index=False
                    )
                    print(f"  ✓ NC active ham licenses → {nc_out}  ({len(nc_active):,} licenses)")
                    return nc_active
        else:
            print(f"  ⚠ FCC returned {resp.status_code}")
    except Exception as e:
        print(f"  ⚠ Download failed: {e}")

    print("\n  MANUAL DOWNLOAD:")
    print("  1. Go to: https://www.fcc.gov/uls/transactions/daily-weekly-files")
    print("  2. Download: 'Amateur Complete (License Data)' → l_amat.zip")
    print(f"  3. Save to: {out_zip}")
    print("  4. Re-run this script to parse and filter to NC\n")

    print("  SCORING TODO (add to lennen_education.ipynb):")
    print("""
    # After running fetch and getting nc_ham_licenses_active.csv:
    import pandas as pd
    ham = pd.read_csv('data/raw/fcc_ham/nc_ham_licenses_active.csv')

    # Map zip → county using HUD USPS crosswalk (or Census ZCTA crosswalk)
    # HUD crosswalk: https://www.huduser.gov/portal/datasets/usps_crosswalk.html
    crosswalk = pd.read_excel('data/raw/zip_to_county_crosswalk.xlsx',
                              dtype={'zip': str, 'county': str})
    ham['zip5'] = ham['zip_code'].str[:5]
    ham = ham.merge(crosswalk[['zip','county']], left_on='zip5', right_on='zip', how='left')

    # Count active licenses per county
    ham_counts = ham.groupby('county').size().reset_index(name='ham_license_count')
    ham_counts = ham_counts.merge(county_pop[['fips','population']], left_on='county', right_on='fips')
    ham_counts['ham_density'] = ham_counts['ham_license_count'] / ham_counts['population'] * 10000
    ham_counts['ham_norm'] = normalize(ham_counts['ham_density'])
    """)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  2. AGRICULTURAL OCCUPATION — ACS C24010
# ══════════════════════════════════════════════════════════════════════════════

def fetch_agriculture():
    """
    ACS table C24010: Occupation by Sex for the Civilian Employed Population
    The farming/fishing/forestry occupation group (variable C24010_030E for males
    + C24010_066E for females) gives the count of agricultural workers per tract.

    Why it matters: Farming households have food production capacity, land, tools,
    and physical labour skills. NC agricultural counties (Johnston, Duplin, Wayne,
    Sampson) are substantially underscored by the Alone TV proxy alone.

    API endpoint:
      https://api.census.gov/data/2022/acs/acs5
      ?get=NAME,C24010_001E,C24010_030E,C24010_066E
      &for=tract:*&in=state:37
      &key={API_KEY}

    Scoring (TODO — add to lennen_education.ipynb):
      ag_workers = C24010_030E + C24010_066E   (male + female farming workers)
      ag_total   = C24010_001E                  (total employed)
      ag_pct     = ag_workers / ag_total
      ag_score   = normalize(ag_pct)
      Direction: POSITIVE (higher ag pct = better food/land resource survival)
      Weight within E: 0.25
    """
    out_path = os.path.join(RAW_DIR, "acs_agriculture_occupation.csv")
    print("Fetching ACS C24010 agricultural occupation data...")

    if CENSUS_API_KEY == "YOUR_API_KEY_HERE":
        print("  ⚠ No Census API key set.")
        print("    Get a free key at: https://api.census.gov/data/key_signup.html")
        print("    Then set environment variable: export CENSUS_API_KEY=your_key")
        print("    Or edit CENSUS_API_KEY in this script.")

    variables = "NAME,C24010_001E,C24010_030E,C24010_066E"
    url = (
        f"https://api.census.gov/data/2022/acs/acs5"
        f"?get={variables}"
        f"&for=tract:*&in=state:{NC_STATE_FIPS}"
        f"&key={CENSUS_API_KEY}"
    )

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            df["GEOID"] = df["state"] + df["county"] + df["tract"]
            for col in ["C24010_001E", "C24010_030E", "C24010_066E"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["ag_workers_total"] = df["C24010_030E"] + df["C24010_066E"]
            df["ag_pct"] = df["ag_workers_total"] / df["C24010_001E"].replace(0, float("nan"))
            df_out = df[["GEOID","NAME","C24010_001E","ag_workers_total","ag_pct"]].copy()
            df_out.columns = ["GEOID","tract_name","total_employed","ag_workers","ag_pct"]
            df_out.to_csv(out_path, index=False)
            print(f"  ✓ ACS C24010 NC tracts saved → {out_path}  ({len(df_out)} tracts)")
            print(f"  NC mean ag occupation rate: {df_out['ag_pct'].mean():.2%}")
            return df_out
        else:
            print(f"  ⚠ Census API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  ⚠ Census API failed: {e}")

    print("\n  MANUAL DOWNLOAD:")
    print("  1. Go to: https://data.census.gov/table?q=C24010")
    print("  2. Filter: Geography → Census Tract → North Carolina → All tracts")
    print("  3. Download CSV")
    print(f"  4. Save as: {out_path}\n")

    print("  SCORING TODO (add to lennen_education.ipynb):")
    print("""
    ag = pd.read_csv('data/raw/acs_agriculture_occupation.csv')
    ag['ag_pct'] = ag['ag_pct'].fillna(0)
    ag['ag_norm'] = normalize(ag['ag_pct'])   # positive direction

    # Merge with existing education score:
    # score_E = (
    #     tv_alone_norm * 0.35 +   # existing Alone/Survivor proxy
    #     edu_attain_norm * 0.20 + # existing ACS S1501
    #     ag_norm * 0.25 +         # NEW: agricultural occupation
    #     ham_norm * 0.20          # NEW: ham radio (once fetched)
    # )
    """)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  3. PHYSICAL INACTIVITY — CDC PLACES (already in repo)
# ══════════════════════════════════════════════════════════════════════════════

def extract_physical_inactivity():
    """
    CDC PLACES 2023 already exists in the repo (cdc_places_obesity.csv).
    The 'LPA' measure (Leisure-Time Physical Inactivity) is in the same file
    but was not extracted in the original Curtis H notebook.

    This function checks for the file, extracts LPA, and writes a clean CSV.

    Scoring (TODO — add to curtis_health.ipynb alongside obesity_rate):
      inactivity_rate = LPA estimate per tract
      inactivity_norm = normalize(inactivity_rate)
      Score direction: NEGATIVE — flip before adding to score_H
        (higher inactivity = worse physical readiness)

      Updated score_H:
        score_H = 1 - (
            obesity_norm     * 0.30 +
            disability_norm  * 0.30 +
            nursing_norm     * 0.20 +
            inactivity_norm  * 0.20   # NEW
        )
    """
    cdc_path = os.path.join(RAW_DIR, "cdc_places_obesity.csv")
    out_path = os.path.join(RAW_DIR, "cdc_places_inactivity.csv")
    print("Extracting physical inactivity (LPA) from CDC PLACES file...")

    if not os.path.exists(cdc_path):
        print(f"  ⚠ CDC PLACES file not found at {cdc_path}")
        print("  Download from: https://chronicdata.cdc.gov/500-Cities-Places/")
        print("  Filter: StateAbbr=NC, MeasureId=LPA")

        # Attempt live download from CDC PLACES API
        api_url = (
            "https://chronicdata.cdc.gov/resource/cwsq-ngmh.json"
            "?StateAbbr=NC&MeasureId=LPA&$limit=5000"
        )
        try:
            resp = requests.get(api_url, timeout=30)
            if resp.status_code == 200:
                df = pd.DataFrame(resp.json())
                df.to_csv(out_path, index=False)
                print(f"  ✓ LPA data fetched from CDC API → {out_path}  ({len(df)} tracts)")
                return df
        except Exception as e:
            print(f"  ⚠ CDC API failed: {e}")
        return None

    # File exists — extract LPA measure
    try:
        df = pd.read_csv(cdc_path, dtype=str)
        measure_col = next(
            (c for c in df.columns if "measure" in c.lower()), None
        )
        if measure_col and "LPA" in df[measure_col].values:
            lpa = df[df[measure_col] == "LPA"].copy()
        else:
            # Try filtering by column name pattern
            lpa = df.copy()

        lpa.to_csv(out_path, index=False)
        print(f"  ✓ LPA rows extracted → {out_path}  ({len(lpa)} rows)")
        return lpa
    except Exception as e:
        print(f"  ⚠ Extraction failed: {e}")

    print("\n  SCORING TODO (add to curtis_health.ipynb):")
    print("""
    # Load the LPA file (or filter from cdc_places_obesity.csv):
    lpa = pd.read_csv('data/raw/cdc_places_inactivity.csv')
    # Typical CDC PLACES columns: LocationID (=GEOID), Data_Value (=rate)
    lpa = lpa.rename(columns={'LocationID': 'GEOID', 'Data_Value': 'inactivity_rate'})
    lpa['inactivity_rate'] = pd.to_numeric(lpa['inactivity_rate'], errors='coerce')

    # Merge with df_H and add to score:
    df_H = df_H.merge(lpa[['GEOID','inactivity_rate']], on='GEOID', how='left')
    df_H['inactivity_rate'] = df_H['inactivity_rate'].fillna(df_H['inactivity_rate'].median())
    df_H['inactivity_norm'] = normalize(df_H['inactivity_rate'])

    # Updated score_H (all factors negative → flip):
    df_H['score_H'] = 1 - (
        df_H['obesity_norm']     * 0.30 +
        df_H['disability_norm']  * 0.30 +
        df_H['nursing_norm']     * 0.20 +
        df_H['inactivity_norm']  * 0.20   # NEW
    )
    """)
    return None


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Category E/H extension data")
    parser.add_argument(
        "--factor", default="all",
        choices=["all", "ham", "agriculture", "inactivity"],
    )
    args = parser.parse_args()

    if args.factor in ("all", "ham"):
        fetch_ham_radio()
    if args.factor in ("all", "agriculture"):
        fetch_agriculture()
    if args.factor in ("all", "inactivity"):
        extract_physical_inactivity()

    print("\nDone. Follow SCORING TODO sections above for each teammate's notebook.")
