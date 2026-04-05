"""
fetch_category_c_extensions.py
Fetch scripts for three new Category C (Social & Community) sub-factors:
  1. Hunting license density     — NCWRC
  2. Religious congregation density — ARDA / Census
  3. Volunteer fire department coverage — NC Office of State Fire Marshal

Run this script to download raw files to data/raw/.
Scoring (normalize → score_C update) is marked TODO pending teammate review.

Usage:
    python data/fetch/fetch_category_c_extensions.py [--factor all|hunting|vfd|congregation]
"""

import os
import argparse
import time
import zipfile
import io
import requests
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

NC_FIPS = "37"


# ══════════════════════════════════════════════════════════════════════════════
#  1. HUNTING LICENSES — NC Wildlife Resources Commission
# ══════════════════════════════════════════════════════════════════════════════

def fetch_hunting_licenses():
    """
    NC Wildlife Resources Commission publishes county-level hunting license
    sales data annually. The public-facing data portal is:
      https://www.ncwildlife.org/licensing/license-info

    Direct CSV download is not available programmatically — NCWRC requires
    a public records request for machine-readable county data.

    This function:
      (a) Attempts to pull from the NCWRC ArcGIS open data portal if available
      (b) Falls back to writing a template CSV with correct structure for
          manual entry from the NCWRC annual report PDF

    Manual source:
      NCWRC Annual Report — Table: "License Sales by County"
      URL: https://www.ncwildlife.org/about-us/annual-reports
      Column needed: hunting licenses issued per county per year (most recent)

    Scoring (TODO — add to kiana_social.ipynb):
      license_density = licenses_sold / county_population * 1000
      Normalize to [0,1] across NC counties.
      Direction: POSITIVE (higher density → higher score_C contribution)
      Weight within C: 0.15 (replaces part of gun_sales weight)
    """
    out_path = os.path.join(RAW_DIR, "nc_hunting_licenses.csv")

    print("Fetching hunting license data...")

    # Attempt NCWRC ArcGIS REST API (may not be available — update URL if portal changes)
    arcgis_url = (
        "https://services.arcgis.com/gfMFBfMsQcvNRxyz/arcgis/rest/services/"
        "Hunting_License_Sales/FeatureServer/0/query"
        "?where=1%3D1&outFields=*&f=json"
    )
    try:
        resp = requests.get(arcgis_url, timeout=10)
        if resp.status_code == 200 and "features" in resp.json():
            features = resp.json()["features"]
            df = pd.DataFrame([f["attributes"] for f in features])
            df.to_csv(out_path, index=False)
            print(f"  ✓ Saved from ArcGIS → {out_path}")
            return df
    except Exception:
        pass

    # Fallback: write template for manual completion
    nc_counties = [
        "Alamance","Alexander","Alleghany","Anson","Ashe","Avery","Beaufort",
        "Bertie","Bladen","Brunswick","Buncombe","Burke","Cabarrus","Caldwell",
        "Camden","Carteret","Caswell","Catawba","Chatham","Cherokee","Chowan",
        "Clay","Cleveland","Columbus","Craven","Cumberland","Currituck","Dare",
        "Davidson","Davie","Duplin","Durham","Edgecombe","Forsyth","Franklin",
        "Gaston","Gates","Graham","Granville","Greene","Guilford","Halifax",
        "Harnett","Haywood","Henderson","Hertford","Hoke","Hyde","Iredell",
        "Jackson","Johnston","Jones","Lee","Lenoir","Lincoln","Macon","Madison",
        "Martin","McDowell","Mecklenburg","Mitchell","Montgomery","Moore","Nash",
        "New Hanover","Northampton","Onslow","Orange","Pamlico","Pasquotank",
        "Pender","Perquimans","Person","Pitt","Polk","Randolph","Richmond",
        "Robeson","Rockingham","Rowan","Rutherford","Sampson","Scotland",
        "Stanly","Stokes","Surry","Swain","Transylvania","Tyrrell","Union",
        "Vance","Wake","Warren","Washington","Watauga","Wayne","Wilkes",
        "Wilson","Yadkin","Yancey"
    ]
    template = pd.DataFrame({
        "county_name": nc_counties,
        "fips": [f"37{str(i+1).zfill(3)}" for i in range(len(nc_counties))],
        "hunting_licenses_sold": None,   # FILL FROM NCWRC ANNUAL REPORT
        "year": 2023,
        "source": "NCWRC Annual Report — manual entry required",
        "notes": "Download from https://www.ncwildlife.org/about-us/annual-reports",
    })
    template.to_csv(out_path, index=False)
    print(f"  ⚠ ArcGIS unavailable — template written to {out_path}")
    print(f"    ACTION: Fill 'hunting_licenses_sold' from NCWRC Annual Report PDF")
    print(f"    URL:    https://www.ncwildlife.org/about-us/annual-reports")
    return template


# ══════════════════════════════════════════════════════════════════════════════
#  2. RELIGIOUS CONGREGATION DENSITY — ARDA / USDA ERS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_congregation_density():
    """
    Association of Religion Data Archives (ARDA) publishes county-level
    congregation counts from the US Religion Census.
    Most recent: 2020 Religious Congregations & Membership Study (RCMS)

    Source:
      https://www.thearda.com/us-religion/census/congregational-membership
      → Download: "2020 US Religion Census: Religious Congregations & Membership Study"
      → File: County-level CSV with congregation counts by denomination

    Alternative: USDA Economic Research Service rural atlas includes
    a "social capital" index that partially captures congregation density.
      https://www.ers.usda.gov/data-products/rural-atlas/

    Scoring (TODO — add to kiana_social.ipynb):
      congregation_density = total_congregations / county_population * 10000
      Normalize to [0,1] across NC counties.
      Direction: POSITIVE (higher density → higher score_C)
      Weight within C: 0.10
    """
    out_path = os.path.join(RAW_DIR, "nc_congregation_density.csv")
    print("Fetching congregation density data...")

    # USDA ERS Rural Atlas — Social Capital index (proxy, publicly downloadable)
    ers_url = (
        "https://www.ers.usda.gov/webdocs/DataFiles/48754/"
        "RuralAtlasData24.zip?v=7681"
    )
    try:
        resp = requests.get(ers_url, timeout=30)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                # Social capital sheet is typically "SocialCapital.csv" or similar
                names = z.namelist()
                social_file = next((n for n in names if "Social" in n and n.endswith(".csv")), None)
                if social_file:
                    with z.open(social_file) as f:
                        df = pd.read_csv(f, dtype={"FIPS": str})
                    nc_df = df[df["FIPS"].str.startswith(NC_FIPS)].copy()
                    nc_df.to_csv(out_path, index=False)
                    print(f"  ✓ USDA ERS Rural Atlas social capital saved → {out_path}")
                    print(f"    NOTE: This is a proxy. For precise congregation counts,")
                    print(f"    manually download ARDA RCMS 2020 from thearda.com")
                    return nc_df
    except Exception as e:
        print(f"  ⚠ USDA ERS download failed: {e}")

    print(f"  ACTION: Manually download from https://www.thearda.com/us-religion/census/")
    print(f"          Select: 2020 RCMS → County file → Filter to NC (FIPS starts 37)")
    print(f"          Save as: {out_path}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  3. VOLUNTEER FIRE DEPARTMENT COVERAGE — NC State Fire Marshal
# ══════════════════════════════════════════════════════════════════════════════

def fetch_vfd_coverage():
    """
    NC Office of State Fire Marshal maintains a registry of fire departments.
    NFIRS (National Fire Incident Reporting System) data includes station
    locations and department type (career / volunteer / combination).

    Sources:
      Primary: NC OSFM — https://www.ncdoi.gov/fire-and-rescue/
        → Fire Department Directory (county-level)
      Supplementary: USFA NFIRS public data
        → https://www.usfa.fema.gov/data/nfirs/

    Scoring (TODO — add to kiana_social.ipynb):
      Step 1: Count volunteer + combination departments per county
      Step 2: vfd_density = vfd_count / county_area_sq_miles
      Step 3: Normalize to [0,1] across NC counties.
      Direction: POSITIVE (more VFD coverage → higher score_C)
      Weight within C: 0.10

    Note: VFD density correlates with rural community organisation —
    counties with dense VFD networks have pre-existing emergency response
    infrastructure that functions as a command node in grid-down scenarios.
    """
    out_path = os.path.join(RAW_DIR, "nc_vfd_coverage.csv")
    print("Fetching VFD coverage data...")

    # USFA Fire Department Census — machine-readable download
    usfa_url = "https://apps.usfa.fema.gov/census-download/main/content/download/FDID_NC.xlsx"
    try:
        resp = requests.get(usfa_url, timeout=30)
        if resp.status_code == 200:
            df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
            # Keep only volunteer and combination departments in NC
            if "State" in df.columns:
                df = df[df["State"] == "NC"].copy()
            vol_types = ["Volunteer", "Mostly Volunteer", "Combination"]
            if "Organization Type" in df.columns:
                vfd = df[df["Organization Type"].isin(vol_types)].copy()
            else:
                vfd = df.copy()
            vfd.to_csv(out_path, index=False)
            print(f"  ✓ USFA VFD registry saved → {out_path}  ({len(vfd)} departments)")
            return vfd
    except Exception as e:
        print(f"  ⚠ USFA download failed: {e}")

    # Fallback: NC OSFM GIS layer (requires browser — print instructions)
    print(f"  ACTION: Download from NC OSFM Fire Department Directory:")
    print(f"          https://www.ncdoi.gov/fire-and-rescue/resources-and-statistics/")
    print(f"          → 'Fire Department Directory' → Export as CSV")
    print(f"          Save as: {out_path}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING STUBS (TODO — add to kiana_social.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

SCORING_TODO = """
# ─────────────────────────────────────────────────────────
# TODO: Add to kiana_social.ipynb after fetching above data
# ─────────────────────────────────────────────────────────

def score_hunting_licenses(df_hunting, df_pop):
    df = df_hunting.merge(df_pop[['county_fips','population']], on='county_fips')
    df['license_density'] = df['hunting_licenses_sold'] / df['population'] * 1000
    df['hunting_norm'] = normalize(df['license_density'])   # positive
    return df[['county_fips', 'hunting_norm']]

def score_congregation_density(df_cong, df_pop):
    df = df_cong.merge(df_pop[['county_fips','population']], on='county_fips')
    df['cong_density'] = df['total_congregations'] / df['population'] * 10000
    df['congregation_norm'] = normalize(df['cong_density'])  # positive
    return df[['county_fips', 'congregation_norm']]

def score_vfd_coverage(df_vfd, df_area):
    counts = df_vfd.groupby('county_fips').size().reset_index(name='vfd_count')
    df = counts.merge(df_area[['county_fips','area_sq_miles']], on='county_fips')
    df['vfd_density'] = df['vfd_count'] / df['area_sq_miles']
    df['vfd_norm'] = normalize(df['vfd_density'])           # positive
    return df[['county_fips', 'vfd_norm']]

# Updated score_C formula (replace existing in kiana_social.ipynb):
# score_C = (
#     gun_norm       * 0.25 +
#     veteran_norm   * 0.20 +
#     military_norm  * 0.15 +
#     hunting_norm   * 0.15 +   # NEW
#     vfd_norm       * 0.10 +   # NEW
#     cong_norm      * 0.10 +   # NEW
#     (1 - elderly_norm) * 0.05  # flipped (dependency burden)
# )
"""


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Category C extension data")
    parser.add_argument(
        "--factor", default="all",
        choices=["all", "hunting", "congregation", "vfd"],
        help="Which factor to fetch",
    )
    args = parser.parse_args()

    if args.factor in ("all", "hunting"):
        fetch_hunting_licenses()
    if args.factor in ("all", "congregation"):
        fetch_congregation_density()
    if args.factor in ("all", "vfd"):
        fetch_vfd_coverage()

    print("\n" + "=" * 60)
    print("SCORING TODO (add to kiana_social.ipynb):")
    print(SCORING_TODO)
