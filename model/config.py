"""
config.py — Central configuration for the NC SZR Outbreak Model.

Changes from v1:
  - Sigmoid κ modifier replaces linear (threshold behaviour at high/low HSI)
  - Tract-level β variation enabled via density + mobility
  - New HSI sub-factor registry (E, C, G extensions)
  - Weight sensitivity sweep ranges defined here
"""

# ── Zombie scenario parameters ────────────────────────────────────────────────
SCENARIOS = {
    "wwz": {
        "label": "World War Z",
        "beta":  3.60e-3,
        "kappa": 2.88e-3,
        "alpha": 0.80,
        "color": "#d4820a",
    },
    "tlou": {
        "label": "The Last of Us",
        "beta":  6.00e-3,
        "kappa": 3.90e-3,
        "alpha": 0.65,
        "color": "#8db829",
    },
    "twd": {
        "label": "The Walking Dead",
        "beta":  2.80e-3,
        "kappa": 2.52e-3,
        "alpha": 0.90,
        "color": "#4a8c3f",
    },
    "28days": {
        "label": "28 Days Later",
        "beta":  9.00e-3,
        "kappa": 4.50e-3,
        "alpha": 0.50,
        "color": "#cc2200",
    },
    "rabies": {
        "label": "Rabies (baseline)",
        "beta":  0.50e-3,
        "kappa": 4.75e-4,
        "alpha": 0.95,
        "color": "#5a5040",
    },
}

# ── HSI category weights ───────────────────────────────────────────────────────
# v1 weights (original team assignment)
HSI_WEIGHTS_V1 = {
    "H": 0.15,   # Health & Fitness        — Curtis
    "M": 0.15,   # Mobility & Escape        — Curtis
    "I": 0.20,   # Infrastructure           — Curtis
    "E": 0.10,   # Education & Awareness    — Lennen
    "C": 0.25,   # Social & Community       — Kiana
    "G": 0.15,   # Geographic & Environment — Rebecca
}

# v2 weights — after adding new sub-factors; C slightly redistributed to E and G
# to reflect that hunting/agriculture/ham-radio boost E and G meaningfully
HSI_WEIGHTS_V2 = {
    "H": 0.15,
    "M": 0.15,
    "I": 0.20,
    "E": 0.13,   # +0.03 — occupation/agriculture proxy now more robust
    "C": 0.22,   # -0.03 — gun+veteran still dominant but slightly diluted
    "G": 0.15,
}

# Active weight set (swap to V2 once new E sub-factors are scored)
HSI_WEIGHTS = HSI_WEIGHTS_V1

# Named version — maps full score column names to weights.
# Used by data/generate_data.py and data/integrate_real_hsi.py so the
# weights are defined once here and not duplicated in every data script.
HSI_WEIGHTS_NAMED = {
    "health_score":         HSI_WEIGHTS_V1["H"],
    "mobility_score":       HSI_WEIGHTS_V1["M"],
    "infrastructure_score": HSI_WEIGHTS_V1["I"],
    "education_score":      HSI_WEIGHTS_V1["E"],
    "social_score":         HSI_WEIGHTS_V1["C"],
    "geo_score":            HSI_WEIGHTS_V1["G"],
}

# ── κ modifier: sigmoid vs linear ────────────────────────────────────────────
#
# V1 (linear):  κ_eff = κ_base × (1 + 0.5 × HSI)
#   → HSI=0.0: κ_eff = 1.00 × κ_base
#   → HSI=1.0: κ_eff = 1.50 × κ_base
#   Problem: assumes perfectly proportional benefit — no threshold effects.
#
# V2 (sigmoid): κ_eff = κ_base × sigmoid_modifier(HSI)
#   → HSI=0.0: κ_eff ≈ 0.73 × κ_base  (disorganised communities kill poorly)
#   → HSI=0.5: κ_eff = 1.00 × κ_base  (neutral)
#   → HSI=1.0: κ_eff ≈ 1.73 × κ_base  (organised communities much more effective)
#   Advantage: captures the real-world threshold where a sufficiently organised
#   community becomes dramatically more effective, not just marginally better.
#
KAPPA_MODIFIER = "sigmoid"   # "linear" | "sigmoid"
SIGMOID_SCALE  = 1.50        # total range = κ_base × [1-scale/2, 1+scale/2] approx
SIGMOID_STEEPNESS = 6.0      # controls how sharp the inflection is at HSI=0.5

# ── Tract-level β variation ───────────────────────────────────────────────────
#
# V1: β is uniform across all tracts.
#   Problem: Wake County's density and Buncombe's terrain produce the same
#   encounter rate, which undersells the M and G sections entirely.
#
# V2: β_eff(tract) = β_base × density_modifier(tract) × encounter_modifier(tract)
#   density_modifier  = 1 + BETA_DENSITY_WEIGHT × norm_pop_density
#   encounter_modifier = 1 - BETA_MOBILITY_WEIGHT × mobility_score
#
#   High density → more S·Z encounters → higher effective β
#   High mobility score → people flee encounters → lower effective β
#
BETA_VARY_BY_TRACT  = True
BETA_DENSITY_WEIGHT = 0.40   # max density effect: +40% on β in densest tracts
BETA_MOBILITY_WEIGHT = 0.25  # max mobility effect: -25% on β in most mobile tracts

# ── Simulation settings ───────────────────────────────────────────────────────
SIM_DAYS          = 365
SIM_STEPS_PER_DAY = 4
CONTAINMENT_THRESHOLD = 0.10   # S/N > 10% at end = contained

# ── New HSI sub-factors registry ──────────────────────────────────────────────
# Each entry describes the factor, its target category, data source,
# direction (positive = higher is better survival, negative = flip),
# and status (ready | fetch_needed | todo)
#
NEW_SUBFACTORS = {

    # ── Category C extensions (Kiana) ──────────────────────────────────────
    "hunting_license_density": {
        "category": "C",
        "label": "Hunting License Density",
        "source": "NC Wildlife Resources Commission (NCWRC) — public county data",
        "fetch_script": "data/fetch/fetch_hunting_licenses.py",
        "direction": "positive",   # more hunters → more armed/skilled community
        "status": "fetch_needed",
        "weight_within_C": 0.15,   # replaces part of gun_sales weight
        "rationale": (
            "Hunters have weapons proficiency, terrain knowledge, and "
            "self-sufficiency skills that raw gun ownership doesn't capture. "
            "A licensed hunter is more effective at zombie removal than an "
            "untrained gun owner."
        ),
    },
    "congregation_density": {
        "category": "C",
        "label": "Religious Congregation Density",
        "source": "USDA Economic Research Service / Association of Religion Data Archives",
        "fetch_script": "data/fetch/fetch_congregation_density.py",
        "direction": "positive",   # coordination hubs for mutual aid
        "status": "fetch_needed",
        "weight_within_C": 0.10,
        "rationale": (
            "FEMA documents consistently show religious congregations function "
            "as community coordination hubs in disasters. High congregation "
            "density correlates with organised mutual aid and communication "
            "networks that outlast infrastructure collapse."
        ),
    },
    "volunteer_fire_density": {
        "category": "C",
        "label": "Volunteer Fire Department Coverage",
        "source": "NC Office of State Fire Marshal — station location data",
        "fetch_script": "data/fetch/fetch_vfd_coverage.py",
        "direction": "positive",
        "status": "fetch_needed",
        "weight_within_C": 0.10,
        "rationale": (
            "Rural VFDs indicate communities already organised around emergency "
            "response with equipment, radio communication, and trained personnel. "
            "In a grid-down scenario, VFD infrastructure becomes a command node."
        ),
    },

    # ── Category G extensions (Rebecca) ─────────────────────────────────────
    "waterway_barrier_score": {
        "category": "G",
        "label": "Waterway Barrier / Access Score",
        "source": "USGS National Hydrography Dataset (NHD) — NC subset",
        "fetch_script": "data/fetch/fetch_waterways.py",
        "direction": "positive",
        "status": "fetch_needed",
        "weight_within_G": 0.30,
        "rationale": (
            "Rivers and lakes serve as both barriers to zombie movement AND "
            "freshwater survival resources. Tracts with riverine boundaries "
            "(Cape Fear, Neuse, Catawba) have natural chokepoints. Crossing "
            "points (bridges) are defensible bottlenecks that concentrate "
            "zombie flow."
        ),
    },
    "forest_public_land_pct": {
        "category": "G",
        "label": "National / State Forest Proximity",
        "source": "USDA Forest Service — National Forest boundaries (GIS)",
        "fetch_script": "data/fetch/fetch_waterways.py",   # bundled in same script
        "direction": "positive",
        "status": "fetch_needed",
        "weight_within_G": 0.20,
        "rationale": (
            "Forested public land provides cover, food foraging, and natural "
            "chokepoints. Pisgah and Nantahala national forests give western "
            "NC tracts an underrated geographic advantage not captured by "
            "NOAA weather station interpolation alone."
        ),
    },
    "bridge_chokepoint_density": {
        "category": "G",
        "label": "Road Network Chokepoint Density",
        "source": "NCDOT Bridge Inventory — public dataset",
        "fetch_script": "data/fetch/fetch_waterways.py",   # bundled
        "direction": "negative",  # fewer bridges = more defensible (flip)
        "status": "fetch_needed",
        "weight_within_G": 0.15,
        "rationale": (
            "Counties with few river crossing points are inherently more "
            "defensible. Dense bridge networks create multiple zombie ingress "
            "routes. Fewer crossings = more concentrated and defensible "
            "perimeter. Direction is NEGATIVE — flip before scoring."
        ),
    },

    # ── Category E extensions (Lennen) ───────────────────────────────────────
    "agricultural_occupation_pct": {
        "category": "E",
        "label": "Agricultural Occupation Rate",
        "source": "ACS C24010 — Occupation by Sex (farming, fishing, forestry)",
        "fetch_script": "data/fetch/fetch_agriculture.py",
        "direction": "positive",
        "status": "fetch_needed",
        "weight_within_E": 0.25,
        "rationale": (
            "Farming households have food production capacity, land, tools, "
            "and physical labour skills that outlast any urban preparation. "
            "NC agricultural counties (Johnston, Duplin, Wayne, Sampson) are "
            "substantially underscored by the current TV-show proxy alone. "
            "ACS C24010 col 'C24010_030E' captures farming/fishing/forestry."
        ),
    },
    "ham_radio_license_density": {
        "category": "E",
        "label": "Amateur Radio Operator Density",
        "source": "FCC Universal Licensing System — public license database",
        "fetch_script": "data/fetch/fetch_ham_radio.py",
        "direction": "positive",
        "status": "fetch_needed",
        "weight_within_E": 0.20,
        "rationale": (
            "FCC amateur radio licenses are public record. Ham operators "
            "provide emergency communication independence when cell and internet "
            "infrastructure fails — a genuine survival factor in prolonged "
            "grid-down scenarios. No current E sub-factor captures this."
        ),
    },
    "physical_inactivity_rate": {
        "category": "H",
        "label": "Physical Inactivity Rate",
        "source": "CDC PLACES 2023 — measure: 'LPA' (leisure-time physical inactivity)",
        "fetch_script": None,   # already in CDC PLACES — add to curtis_health.ipynb
        "direction": "negative",  # higher inactivity = lower H score (flip)
        "status": "ready",       # data already available, just not extracted
        "weight_within_H": 0.25,
        "rationale": (
            "A county can have moderate obesity but very high physical inactivity, "
            "or vice versa. LPA from CDC PLACES is a separate measure from obesity "
            "and captures sedentary population share more directly. Add alongside "
            "obesity_rate in curtis_health.ipynb."
        ),
    },
}

# ── Sensitivity sweep configuration ──────────────────────────────────────────
# Used by sensitivity_analysis.py to sweep category weights
SENSITIVITY_CONFIG = {
    "n_steps": 11,           # number of steps per weight axis (0.0 to 1.0 in 0.1 increments)
    "scenarios_to_test": ["28days", "tlou", "wwz"],  # run sweep for these scenarios
    "output_path": "outputs/sensitivity/",
    "sweep_pairs": [         # pairs to sweep against each other (hold others fixed at V1)
        ("C", "H"),          # most interesting: community strength vs physical fitness
        ("C", "G"),          # military density vs geographic isolation
        ("M", "I"),          # mobility vs infrastructure
        ("E", "C"),          # education/skills vs community defense
    ],
}

# ── Output paths ──────────────────────────────────────────────────────────────
PATHS = {
    "raw_data":       "data/raw/",
    "processed_data": "data/processed/",
    "fetch_scripts":  "data/fetch/",
    "outputs":        "outputs/",
    "sensitivity":    "outputs/sensitivity/",
    "models":         "outputs/",
}
