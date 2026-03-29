"""
Generate synthetic SZR outbreak data for training the surrogate ML model.

Simulates 10,000 scenarios using the SZR differential equation system and
records peak zombie fraction, time to peak, and containment outcome.
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint


def szr_odes(y, t, beta, zeta, alpha, N):
    """SZR differential equation system (frequency-dependent transmission).

    Uses S*Z/N normalization so that beta is independent of population size
    and R0 = beta / (zeta + alpha) regardless of N.
    """
    S, Z, R = y
    dS_dt = -beta * S * Z / N
    dZ_dt = beta * S * Z / N - zeta * Z - alpha * Z
    dR_dt = zeta * Z + alpha * Z
    return [dS_dt, dZ_dt, dR_dt]


def simulate_scenario(beta, zeta, alpha, initial_population, initial_infected,
                      mobility_score, infrastructure_score, health_score,
                      t_max=180):
    """Run a single SZR simulation and return outcome metrics."""
    # Scale beta by (1 - mobility_score): higher restriction score reduces transmission
    effective_beta = beta * (1.0 - mobility_score)

    N = initial_population
    Z0 = min(initial_infected, N - 1)
    S0 = N - Z0
    R0 = 0.0

    t = np.linspace(0, t_max, t_max + 1)
    y0 = [S0, Z0, R0]

    try:
        sol = odeint(szr_odes, y0, t, args=(effective_beta, zeta, alpha, N),
                     mxstep=5000, rtol=1e-6, atol=1e-8)
    except Exception:
        return None

    S_t = sol[:, 0]
    Z_t = sol[:, 1]

    if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
        return None

    zombie_fraction = Z_t / N
    peak_zombie_fraction = float(np.max(zombie_fraction))
    time_to_peak = int(np.argmax(zombie_fraction))

    # Containment: S(t) at end stabilizes above 10% of N
    containment = 1 if S_t[-1] > 0.10 * N else 0

    return {
        "beta": beta,
        "zeta": zeta,
        "alpha": alpha,
        "initial_population": initial_population,
        "initial_infected": initial_infected,
        "mobility_score": mobility_score,
        "infrastructure_score": infrastructure_score,
        "health_score": health_score,
        "peak_zombie_fraction": peak_zombie_fraction,
        "time_to_peak": time_to_peak,
        "containment": containment,
    }


def generate_dataset(n_scenarios=10000, seed=42):
    """Sample random parameters and simulate all scenarios."""
    rng = np.random.default_rng(seed)

    beta_vals = rng.uniform(0.001, 0.9, n_scenarios)
    zeta_vals = rng.uniform(0.001, 0.5, n_scenarios)
    alpha_vals = rng.uniform(0.0001, 0.01, n_scenarios)
    pop_vals = rng.uniform(5000, 1_000_000, n_scenarios)
    infected_vals = rng.uniform(1, 50, n_scenarios)
    mobility_vals = rng.uniform(0.0, 1.0, n_scenarios)
    infrastructure_vals = rng.uniform(0.0, 1.0, n_scenarios)
    health_vals = rng.uniform(0.0, 1.0, n_scenarios)

    records = []
    for i in range(n_scenarios):
        result = simulate_scenario(
            beta=beta_vals[i],
            zeta=zeta_vals[i],
            alpha=alpha_vals[i],
            initial_population=pop_vals[i],
            initial_infected=infected_vals[i],
            mobility_score=mobility_vals[i],
            infrastructure_score=infrastructure_vals[i],
            health_score=health_vals[i],
        )
        if result is not None:
            records.append(result)

        if (i + 1) % 1000 == 0:
            print(f"  Simulated {i + 1}/{n_scenarios} scenarios...")

    df = pd.DataFrame(records)
    # Drop rows with NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def main():
    import os

    print("Generating SZR synthetic dataset...")
    df = generate_dataset(n_scenarios=10000)

    out_path = os.path.join(os.path.dirname(__file__), "szr_synthetic.csv")
    df.to_csv(out_path, index=False)

    print(f"\nDataset shape: {df.shape}")
    containment_counts = df["containment"].value_counts()
    print(f"Containment class balance:\n{containment_counts}")
    print(f"  Containment rate: {containment_counts.get(1, 0) / len(df):.2%}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
