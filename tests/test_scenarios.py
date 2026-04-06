"""
tests/test_scenarios.py
Validates that each zombie show scenario produces the expected
qualitative outcome in the frequency-dependent SZR formulation.

Run: python tests/test_scenarios.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from scipy.integrate import odeint

POPULATION = 184226   # Pitt County
DAYS       = 180

SCENARIOS = {
    "Walking Dead":  {"beta": 0.28, "zeta": 0.25, "alpha": 0.01,
                      "z0": 10,  "expect": "survive"},
    "Last of Us":    {"beta": 0.50, "zeta": 0.20, "alpha": 0.01,
                      "z0": 20,  "expect": "marginal"},
    "World War Z":   {"beta": 0.45, "zeta": 0.35, "alpha": 0.01,
                      "z0": 50,  "expect": "survive"},
    "28 Days Later": {"beta": 0.75, "zeta": 0.12, "alpha": 0.01,
                      "z0": 5,   "expect": "collapse"},
    "Rabies":        {"beta": 0.05, "zeta": 0.25, "alpha": 0.01,
                      "z0": 185, "expect": "survive"},
}

def szr_ode(y, t, beta, zeta, alpha):
    S, Z, R = y
    N = S + Z + R
    if N <= 0 or Z <= 0:
        return [0.0, 0.0, 0.0]
    dS = -beta * S * Z / N
    dZ =  beta * S * Z / N - (zeta + alpha) * Z
    dR =  (zeta + alpha) * Z
    return [dS, dZ, dR]

def run(name, params):
    N  = POPULATION
    S0 = N - params["z0"]
    t  = np.linspace(0, DAYS, DAYS * 4)
    sol = odeint(szr_ode, [S0, params["z0"], 0.0], t,
                 args=(params["beta"], params["zeta"], params["alpha"]))
    S, Z = sol[:, 0], sol[:, 1]
    peak_frac    = float(np.max(Z) / N)
    survive_frac = float(S[-1] / N)
    peak_day     = float(t[np.argmax(Z)])
    contained    = survive_frac > 0.10
    return peak_frac, survive_frac, peak_day, contained

PASS = 0
FAIL = 0

print("=" * 60)
print("SZR Scenario Simulation Tests")
print(f"Population: {POPULATION:,}  |  Days: {DAYS}")
print("=" * 60)

for name, params in SCENARIOS.items():
    peak, surv, day, contained = run(name, params)
    expect = params["expect"]

    # Determine outcome category
    if surv > 0.50:
        outcome = "survive"
    elif surv > 0.10:
        outcome = "marginal"
    else:
        outcome = "collapse"

    # Validate
    # Walking Dead / World War Z / Rabies → must survive (>50%)
    # Last of Us → marginal or survive (>10%)
    # 28 Days Later → collapse (<10%)
    if expect == "survive":
        ok = surv > 0.50 and peak > 0.001
    elif expect == "marginal":
        ok = surv > 0.10
    elif expect == "collapse":
        ok = surv < 0.10 and peak > 0.10

    status = "✅ PASS" if ok else "❌ FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1

    print(f"\n{status} {name}")
    print(f"  Peak zombie frac: {peak:.1%}  |  Day: {day:.0f}")
    print(f"  Survivors: {surv:.1%}  |  Outcome: {outcome}  |  Expected: {expect}")

    # Additional checks
    if peak < 0.001:
        print(f"  ⚠  WARNING: Peak is near zero — zombies not spreading")
    if day < 1:
        print(f"  ⚠  WARNING: Peak on day 0 — likely a parameter issue")

print(f"\n{'=' * 60}")
print(f"Results: {PASS} passed, {FAIL} failed")

if FAIL > 0:
    print("\nFAILED SCENARIOS — fix SCENARIO_PRESETS in app/streamlit_app.py")
    print("Ensure β > ζ + α for collapse scenarios")
    print("Ensure β < ζ + α for survival scenarios")
    sys.exit(1)
else:
    print("\nAll scenarios produce expected qualitative outcomes.")
    print("Safe to use for demo.")
