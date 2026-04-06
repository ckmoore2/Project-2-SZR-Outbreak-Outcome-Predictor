# Manual Fetch Instructions
## Factors requiring manual download (auto-fetch failed)

---

### 1. VFD Coverage (Category C — Social Cohesion / Volunteer Fire Dept)

**Issue:** USFA NFIRS download failed due to SSL certificate verification error
(`SSLCertVerificationError` on `apps.usfa.fema.gov`).

**Manual steps — Option A (USFA Fire Department Census):**
1. Go to: `https://apps.usfa.fema.gov/census-download/`
2. Click: "Download the complete Fire Department Census" data
3. Filter to North Carolina (State = NC)
4. Download as CSV or Excel
5. Save as: `data/raw/nc_vfd_coverage.csv`
6. Re-run: `python data/rebuild_county_data.py`

**Manual steps — Option B (NC Office of State Fire Marshal):**
1. Go to: `https://www.ncdoi.gov/fire-and-rescue/resources-and-statistics/`
2. Download: "Fire Department Directory"
3. Save as: `data/raw/nc_vfd_coverage.csv`

**Expected columns:** `county`, `department_name`, `organization_type`,
`county_fips` (or derivable from county name using `NC_COUNTIES` dict in
`rebuild_county_data.py`)

**Integration — add to `data/rebuild_county_data.py` `build_social()` function:**
```python
vfd_path = os.path.join(_HERE, "raw", "nc_vfd_coverage.csv")
if os.path.exists(vfd_path):
    vfd = pd.read_csv(vfd_path)
    vol_types = ["Volunteer", "Mostly Volunteer", "Combination"]
    vfd_vol = vfd[vfd["organization_type"].isin(vol_types)]
    vfd_county = vfd_vol.groupby("county_fips").size().reset_index(name="vfd_count")
    # Merge and compute density per 1k sq miles (or normalize directly)
    df = df.merge(vfd_county, on="county_fips", how="left")
    df["vfd_count"] = df["vfd_count"].fillna(0)
    df["vfd_norm"] = normalize(df["vfd_count"])   # positive direction
    # Add vfd_norm to score_C formula with weight ~0.15
```

---

### 2. Congregation Density (Category C — Social Cohesion)

**Issue:** ARDA (Association of Religion Data Archives) requires account
registration for bulk data access. USDA ERS direct CSV URLs returned 404.

**Manual steps — Option A (ARDA RCMS 2020):**
1. Register (free) at: `https://www.thearda.com/`
2. Navigate to: Data > U.S. Religion > Census > Congregational Membership
3. Download: "2020 Religious Congregations & Membership Study (RCMS)"
   — select the County-level file
4. Filter to NC counties (FIPS starts with `37`)
5. Save as: `data/raw/nc_congregation_density.csv`

**Manual steps — Option B (USDA ERS Rural Atlas — Social Capital proxy):**
1. Go to: `https://www.ers.usda.gov/data-products/rural-atlas/`
2. Download: `RuralAtlasData24.zip`
3. Extract the Excel file; open the `SocialCapital` sheet
4. Filter to NC counties (State FIPS = 37)
5. Save relevant columns as: `data/raw/nc_congregation_density.csv`

**Expected columns:** `county_fips`, `total_congregations`
(or `social_capital_index` if using USDA proxy)

**Integration — add to `data/rebuild_county_data.py` `build_social()` function:**
```python
cong_path = os.path.join(_HERE, "raw", "nc_congregation_density.csv")
if os.path.exists(cong_path):
    cong = pd.read_csv(cong_path, dtype={"county_fips": str})
    cong["county_fips"] = cong["county_fips"].astype(str).str.zfill(5)
    cong = cong.merge(pop_df[["county_fips", "population"]], on="county_fips", how="left")
    cong["cong_density"] = cong["total_congregations"] / cong["population"] * 10000
    cong_county = cong[["county_fips", "cong_density"]]
    df = df.merge(cong_county, on="county_fips", how="left")
    df["cong_density"] = df["cong_density"].fillna(df["cong_density"].median())
    df["congregation_norm"] = normalize(df["cong_density"])  # positive direction
    # Add congregation_norm to score_C formula with weight ~0.15
```

---

### Re-run order after completing manual downloads

```bash
python data/rebuild_county_data.py      # picks up new raw files
python data/integrate_real_hsi.py       # recomputes HSI distributions
python data/generate_data.py            # regenerates synthetic dataset
python model/train.py                   # retrains all ablation experiments
```

---

### Expected impact

Both VFD coverage and congregation density are positive social cohesion
signals — counties with more volunteer emergency capacity and community
organizations are expected to have higher `social_score` (`score_C`).
Incorporating these factors should sharpen differentiation between rural
counties that currently receive similar `score_C` values from veteran rates
and gun proxies alone.
