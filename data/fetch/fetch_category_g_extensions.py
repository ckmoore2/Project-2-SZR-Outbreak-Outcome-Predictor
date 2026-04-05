"""
fetch_category_g_extensions.py
Fetch scripts for three new Category G (Geographic & Environmental) sub-factors:
  1. Waterway barrier / access score — USGS NHD
  2. National / state forest proximity — USDA Forest Service
  3. Road network chokepoint density — NCDOT bridge inventory

Run this script to download raw files to data/raw/.
Scoring (normalize → score_G update) is marked TODO pending Rebecca's review.

Usage:
    python data/fetch/fetch_category_g_extensions.py [--factor all|waterways|forest|bridges]

Dependencies:
    pip install requests geopandas shapely fiona
"""

import os
import io
import json
import zipfile
import argparse
import requests
import pandas as pd

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("WARNING: geopandas not installed. Spatial operations will be skipped.")
    print("         Install with: pip install geopandas")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

NC_STATE_FIPS = "37"
NC_BBOX = "-84.3219,33.8421,-75.4001,36.5881"   # minx,miny,maxx,maxy


# ══════════════════════════════════════════════════════════════════════════════
#  1. WATERWAYS — USGS National Hydrography Dataset (NHD)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_waterways():
    """
    USGS National Hydrography Dataset (NHD) provides flowline and waterbody
    geometries at 1:24,000 scale for all of NC.

    Source:
      USGS NHD Download Service:
      https://www.usgs.gov/national-hydrography/access-national-hydrography-products

    Two components needed:
      A. NHD Flowlines (rivers, streams) — for barrier analysis
      B. NHD Waterbodies (lakes, reservoirs) — for freshwater resource scoring

    API endpoint (The National Map — TNM):
      https://tnmaccess.nationalmap.gov/api/v1/products?
        datasets=National%20Hydrography%20Dataset%20(NHD)%20Best%20Resolution
        &bbox=-84.32,33.84,-75.40,36.59
        &outputFormat=JSON

    Scoring (TODO — add to rebecca_geo_weather.ipynb):
      Step 1: For each census tract, compute total major waterway length
              intersecting or within 2km of tract boundary (km of flowline)
      Step 2: Compute waterbody area within 5km of tract centroid (sq km)
      Step 3: waterway_score = 0.6 × norm(flowline_length) + 0.4 × norm(waterbody_area)
      Direction: POSITIVE (more waterway = better barrier + resource)
      Weight within G: 0.30
    """
    out_path = os.path.join(RAW_DIR, "nc_waterways_nhd.geojson")
    print("Fetching NHD waterways for NC...")

    # TNM API for NHD products — returns download URLs
    tnm_api = (
        "https://tnmaccess.nationalmap.gov/api/v1/products"
        "?datasets=National%20Hydrography%20Dataset%20%28NHD%29%20Best%20Resolution"
        f"&bbox={NC_BBOX}"
        "&outputFormat=JSON"
        "&max=5"
    )

    try:
        resp = requests.get(tnm_api, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items", [])
            print(f"  Found {len(items)} NHD products for NC bounding box.")
            for item in items[:2]:
                print(f"    {item.get('title', 'Unknown')} — {item.get('downloadURL', 'no URL')}")
            meta_path = os.path.join(RAW_DIR, "nc_nhd_product_list.json")
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Product list saved → {meta_path}")
            print(f"  ACTION: Download the listed GDB or ShapeFile from the URLs above,")
            print(f"          extract the NHDFlowline and NHDWaterbody layers,")
            print(f"          and save as: {out_path}")
        else:
            print(f"  TNM API returned {resp.status_code}")
    except Exception as e:
        print(f"  TNM API failed: {e}")

    # Direct download instruction fallback
    print("\n  MANUAL DOWNLOAD INSTRUCTIONS:")
    print("  1. Go to: https://apps.nationalmap.gov/downloader/")
    print("  2. Search: National Hydrography Dataset (NHD) Best Resolution")
    print("  3. Select: North Carolina (State)")
    print("  4. Download the GDB file and extract NHDFlowline + NHDWaterbody")
    print(f"  5. Save processed GeoJSON as: {out_path}")
    print()

    # Scoring stub
    print("  SCORING TODO (add to rebecca_geo_weather.ipynb):")
    print("""
    # After loading nhd_flowlines and nhd_waterbodies as GeoDataFrames:
    import geopandas as gpd
    tracts = gpd.read_file('data/processed/nc_tracts.shp').to_crs(epsg=32617)
    flowlines = gpd.read_file('data/raw/nc_waterways_nhd.geojson').to_crs(epsg=32617)
    waterbodies = gpd.read_file('data/raw/nc_waterbodies_nhd.geojson').to_crs(epsg=32617)

    # Major rivers only (Strahler order >= 4 in NHD)
    major_rivers = flowlines[flowlines['StreamOrde'] >= 4]

    # For each tract: total major river length within 2km buffer
    tracts['buffer_2km'] = tracts.geometry.buffer(2000)
    tracts['river_length_km'] = tracts['buffer_2km'].apply(
        lambda buf: major_rivers.clip(buf).length.sum() / 1000
    )
    tracts['waterbody_area_km2'] = tracts['buffer_2km'].apply(
        lambda buf: waterbodies.clip(buf).area.sum() / 1e6
    )
    tracts['waterway_score'] = (
        normalize(tracts['river_length_km']) * 0.6 +
        normalize(tracts['waterbody_area_km2']) * 0.4
    )
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  2. NATIONAL / STATE FOREST — USDA Forest Service
# ══════════════════════════════════════════════════════════════════════════════

def fetch_forest_land():
    """
    USDA Forest Service publishes national forest boundaries as GIS layers.
    NC has two national forests: Pisgah and Nantahala (western NC).
    NC also has 41 state forests managed by NC Forest Service.

    Sources:
      National Forests: USDA FS S_USA.AdministrativeForest
        https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.AdministrativeForest.zip
      State Forests: NC Forest Service
        https://www.ncforestservice.gov/managing_your_forest/state_forests.htm
        (no machine-readable download — manual GIS export required)

    Scoring (TODO — add to rebecca_geo_weather.ipynb):
      forest_pct = forest_area_within_10km / total_area_within_10km
      forest_score = normalize(forest_pct)
      Direction: POSITIVE (more forest = better cover, foraging, chokepoints)
      Weight within G: 0.20
    """
    out_path = os.path.join(RAW_DIR, "nc_national_forests.shp.zip")
    print("Fetching USDA National Forest boundaries...")

    fs_url = "https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.AdministrativeForest.zip"
    try:
        resp = requests.get(fs_url, timeout=60, stream=True)
        if resp.status_code == 200:
            total = 0
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total += len(chunk)
            print(f"  ✓ National Forest boundaries downloaded → {out_path} ({total//1024}KB)")
            if HAS_GEOPANDAS:
                gdf = gpd.read_file(f"zip://{out_path}")
                nc_forests = gdf[gdf["REGION"] == "08"].copy()   # Region 8 = Southern
                nc_only = nc_forests[nc_forests["FORESTNAME"].str.contains(
                    "Pisgah|Nantahala|Uwharrie|Croatan", case=False, na=False
                )]
                nc_out = os.path.join(RAW_DIR, "nc_national_forests_only.geojson")
                nc_only.to_file(nc_out, driver="GeoJSON")
                print(f"  ✓ NC forests extracted → {nc_out}  ({len(nc_only)} features)")
        else:
            print(f"  ⚠ USDA FS returned {resp.status_code}")
            print(f"    Manual download: {fs_url}")
    except Exception as e:
        print(f"  ⚠ Download failed: {e}")
        print(f"    Manual download: {fs_url}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. BRIDGE CHOKEPOINTS — NCDOT Bridge Inventory
# ══════════════════════════════════════════════════════════════════════════════

def fetch_bridge_chokepoints():
    """
    NCDOT maintains the NC Bridge Inventory — all bridges on state-maintained
    roads with location, condition, and functional data.

    Higher bridge density = more crossing points = more zombie ingress routes
    = harder to defend. So direction is NEGATIVE (flip before scoring).

    Source:
      NCDOT Open Data: https://connect.ncdot.gov/resources/bridge/Pages/default.aspx
      NBI (National Bridge Inventory) — FHWA publishes state files annually:
      https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm

    The NBI NC file is a fixed-width ASCII format — parse using FHWA spec.

    Scoring (TODO — add to rebecca_geo_weather.ipynb):
      bridge_count_per_tract = count of bridges within county / county_area_sq_miles
      bridge_score_raw = normalize(bridge_count_per_tract)
      bridge_score = 1 - bridge_score_raw   # FLIP — fewer bridges = more defensible
      Direction: NEGATIVE (flip), weight within G: 0.15
    """
    out_path = os.path.join(RAW_DIR, "nc_bridges_nbi.txt")
    print("Fetching NC bridge inventory (FHWA NBI)...")

    # FHWA NBI state files — NC file
    nbi_url = "https://www.fhwa.dot.gov/bridge/nbi/2023/delimited/NC23.txt"
    try:
        resp = requests.get(nbi_url, timeout=30)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            print(f"  ✓ NBI NC bridge file saved → {out_path}")

            # Quick parse to verify structure
            df = pd.read_csv(out_path, nrows=5, encoding="latin-1")
            print(f"  Columns: {list(df.columns[:8])} ...")
            csv_out = out_path.replace(".txt", ".csv")
            df_full = pd.read_csv(out_path, encoding="latin-1", low_memory=False)
            df_full.to_csv(csv_out, index=False)
            print(f"  ✓ Parsed CSV saved → {csv_out}  ({len(df_full)} bridges)")
        else:
            print(f"  ⚠ FHWA returned {resp.status_code}")
            # Try alternate year
            alt_url = "https://www.fhwa.dot.gov/bridge/nbi/2022/delimited/NC22.txt"
            print(f"    Trying {alt_url}")
            resp2 = requests.get(alt_url, timeout=30)
            if resp2.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(resp2.content)
                print(f"  ✓ 2022 NBI file saved → {out_path}")
    except Exception as e:
        print(f"  ⚠ Download failed: {e}")

    print("\n  SCORING TODO (add to rebecca_geo_weather.ipynb):")
    print("""
    import pandas as pd
    bridges = pd.read_csv('data/raw/nc_bridges_nbi.csv', low_memory=False)

    # Key NBI columns: COUNTY_CODE_008, LAT_016, LONG_017, BRIDGE_CONDITION
    # Filter to NC only (STATE_CODE_001 == 37)
    bridges = bridges[bridges['STATE_CODE_001'] == 37].copy()

    # Aggregate by county (NBI county code is 3-digit FIPS suffix)
    bridge_counts = bridges.groupby('COUNTY_CODE_008').size().reset_index(name='bridge_count')

    # Merge with county area data and compute density
    bridge_counts['fips'] = '37' + bridge_counts['COUNTY_CODE_008'].astype(str).str.zfill(3)
    bridge_counts = bridge_counts.merge(county_areas[['fips','area_sq_miles']], on='fips')
    bridge_counts['bridge_density'] = bridge_counts['bridge_count'] / bridge_counts['area_sq_miles']

    # FLIP: fewer bridges = more defensible = higher score
    bridge_counts['bridge_score'] = 1 - normalize(bridge_counts['bridge_density'])
    """)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Category G extension data")
    parser.add_argument(
        "--factor", default="all",
        choices=["all", "waterways", "forest", "bridges"],
    )
    args = parser.parse_args()

    if args.factor in ("all", "waterways"):
        fetch_waterways()
    if args.factor in ("all", "forest"):
        fetch_forest_land()
    if args.factor in ("all", "bridges"):
        fetch_bridge_chokepoints()

    print("\nDone. Check data/raw/ for downloaded files and follow ACTION items above.")
