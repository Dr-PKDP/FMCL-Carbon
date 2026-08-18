"""
Section 11.3: global sensitivity analysis of Theorem 4's architecture-
selection crossover horizon T*_min, applied to the paper's own two worked
instantiations from Section 10.3 (Table 4: consumer-device activity
recognition; healthcare, mitigated).

This uses the same point estimates already in the manuscript's Section
10.3, widened into literature-grounded uncertainty ranges, and asks: how
much does T*_min actually move around under that uncertainty, and which
input parameter drives the movement?

Theorem 4 (Lemma 3's pairwise form), Phi_FMCL approximately 0 under P1:
    T*_a = (Phi_a - Phi_FMCL) / (c_FMCL - c_a),  if c_FMCL > c_a
    T*_a = +inf (FMCL wins at every horizon),     if c_FMCL <= c_a
    T*_min = min(T*_edge, T*_cloud)

These are the point estimates the manuscript reports as illustrative
(Section 10.1-10.3); Section 10.4 replaces them with the real-hardware
figures used in carbon_accounting/, and Section 10.5 shows the population-
scale reconciliation these point estimates motivate.
"""

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

np.random.seed(42)

# The paper's own Section 10.3 / Table 4 point estimates (g CO2e), as the
# center of each distribution. Ranges reflect uncertainty documented in
# the embodied-carbon and operational-carbon literature:
#   - embodied carbon (Phi terms): +/-40%, consistent with the up to
#     roughly 1.6x mean-vs-95th-percentile spread reported for embodied
#     carbon estimates across technology nodes in the literature
#   - edge/cloud operational rate: +/-30% (professionally operated,
#     comparatively controlled conditions)
#   - FMCL operational rate: +30%/-40% asymmetric, reflecting the paper's
#     own stated Wi-Fi-versus-cellular communication-energy penalty
#     (Sections 4.1, 5.1) as the dominant source of variance in c_FMCL
#     specifically

SCENARIOS = {
    "activity_recognition": {
        "Phi_edge":  3.0e6,
        "Phi_cloud": 1.0e6,
        "c_FMCL":    600.0,
        "c_edge":    300.0,
        "c_cloud":   250.0,
        "paper_T_star_min": 2857,   # as reported in Section 10.3
    },
    "healthcare_mitigated": {
        "Phi_edge":  5.0e5,
        "Phi_cloud": 8.0e5,
        "c_FMCL":    120.0,
        "c_edge":    90.0,
        "c_cloud":   70.0,
        "paper_T_star_min": 16000,  # as reported in Section 10.3
    },
}

PARAM_NAMES = ["Phi_edge", "Phi_cloud", "c_FMCL", "c_edge", "c_cloud"]


def bounds_for(point):
    """Literature-grounded +/- ranges around each point estimate."""
    return {
        "Phi_edge":  [0.6 * point["Phi_edge"],  1.4 * point["Phi_edge"]],
        "Phi_cloud": [0.6 * point["Phi_cloud"], 1.4 * point["Phi_cloud"]],
        "c_FMCL":    [0.6 * point["c_FMCL"],    1.3 * point["c_FMCL"]],
        "c_edge":    [0.7 * point["c_edge"],    1.3 * point["c_edge"]],
        "c_cloud":   [0.7 * point["c_cloud"],   1.3 * point["c_cloud"]],
    }


def t_star_min(Phi_edge, Phi_cloud, c_FMCL, c_edge, c_cloud, Phi_FMCL=0.0):
    """Theorem 4 exactly as stated, including the paper's own convention
    that a term is treated as unbounded when c_FMCL <= c_a."""
    BIG = 1e12  # stand-in for "unbounded" so the array stays numeric
    T_edge = np.where(c_FMCL > c_edge, (Phi_edge - Phi_FMCL) / np.maximum(c_FMCL - c_edge, 1e-9), BIG)
    T_cloud = np.where(c_FMCL > c_cloud, (Phi_cloud - Phi_FMCL) / np.maximum(c_FMCL - c_cloud, 1e-9), BIG)
    return np.minimum(T_edge, T_cloud)


def run_scenario(name, point):
    b = bounds_for(point)
    problem = {
        "num_vars": 5,
        "names": PARAM_NAMES,
        "bounds": [b[p] for p in PARAM_NAMES],
    }

    # Saltelli sampling for Sobol indices (first-order and total-order)
    param_values = saltelli.sample(problem, 2048, calc_second_order=False)
    Y = t_star_min(
        param_values[:, 0], param_values[:, 1], param_values[:, 2],
        param_values[:, 3], param_values[:, 4],
    )
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    print(f"\n=== {name} ===")
    print(f"Paper's point-estimate T*_min: {point['paper_T_star_min']:,} rounds")
    print(f"Simulated median T*_min:       {np.median(Y):,.0f} rounds")
    print(f"Simulated 90% interval:        [{np.percentile(Y, 5):,.0f}, {np.percentile(Y, 95):,.0f}] rounds")
    print(f"Coefficient of variation:      {np.std(Y) / np.mean(Y):.2f}")
    print(f"{'Parameter':<12}{'S1 (first-order)':>18}{'ST (total-order)':>18}")
    for i, p in enumerate(PARAM_NAMES):
        print(f"{p:<12}{Si['S1'][i]:>18.3f}{Si['ST'][i]:>18.3f}")

    # Decision-relevant reframing: P(FMCL still favored) at illustrative
    # planning horizons a real deployment might actually commit to.
    for T_target in [500, 2000, 5000]:
        p_favored = float(np.mean(Y > T_target))
        print(f"  P(FMCL favored | planned horizon T={T_target:,}): {p_favored:.2f}")

    return Si, Y, problem


if __name__ == "__main__":
    results = {}
    for name, point in SCENARIOS.items():
        results[name] = run_scenario(name, point)
