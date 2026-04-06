"""
Validates that all scenario beta/zeta/alpha values fall within
the slider min/max bounds defined in FEATURE_META.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

FEATURE_META = {
    "beta":  {"min": 0.0001, "max": 1.0,  "step": 0.0001},
    "zeta":  {"min": 0.0001, "max": 0.5,  "step": 0.0001},
    "alpha": {"min": 0.001,  "max": 1.0,  "step": 0.001},
}

SCENARIOS_TO_CHECK = {
    "Walking Dead":  {"beta": 0.28,  "zeta": 0.25,  "alpha": 0.01},
    "Last of Us":    {"beta": 0.50,  "zeta": 0.20,  "alpha": 0.01},
    "World War Z":   {"beta": 0.45,  "zeta": 0.35,  "alpha": 0.01},
    "28 Days Later": {"beta": 0.75,  "zeta": 0.12,  "alpha": 0.01},
    "Rabies":        {"beta": 0.05,  "zeta": 0.25,  "alpha": 0.01},
    "Early Det.":    {"beta": 0.15,  "zeta": 0.18,  "alpha": 0.01},
    "Active Spread": {"beta": 0.30,  "zeta": 0.10,  "alpha": 0.01},
    "Rapid Outbrk":  {"beta": 0.55,  "zeta": 0.08,  "alpha": 0.01},
    "Total Collapse":{"beta": 0.80,  "zeta": 0.04,  "alpha": 0.01},
    "Military":      {"beta": 0.25,  "zeta": 0.35,  "alpha": 0.01},
    "Medical":       {"beta": 0.20,  "zeta": 0.22,  "alpha": 0.01},
}

PASS = FAIL = 0
print("Slider bounds validation")
print("=" * 50)
for name, params in SCENARIOS_TO_CHECK.items():
    errors = []
    for key, val in params.items():
        if key not in FEATURE_META:
            continue
        meta = FEATURE_META[key]
        if val < meta["min"] or val > meta["max"]:
            errors.append(f"{key}={val} outside [{meta['min']}, {meta['max']}]")
        # Check step alignment (value must be multiple of step)
        step = meta["step"]
        remainder = round(val % step, 10)
        if remainder > step * 0.01 and remainder < step * 0.99:
            errors.append(f"{key}={val} not aligned to step={step}")
    if errors:
        print(f"❌ {name}: {'; '.join(errors)}")
        FAIL += 1
    else:
        print(f"✅ {name}: all values within bounds")
        PASS += 1

print(f"\n{PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
