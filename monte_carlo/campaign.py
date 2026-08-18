"""
Section 11.4: Monte Carlo simulation of a real FMCL deployment campaign,
round by round, against dedicated-edge and cloud alternatives running over
the same calendar span. This is a genuine simulation, not a closed-form
evaluation: device availability, device-level power heterogeneity, and
per-round communication mode are all stochastic, matching the paper's own
Definition 1 (Bernoulli availability) and P2 (state-dependent, exogenous
participation). Real hardware and grid data from Section 10.4 grounds
every energy and carbon coefficient.

For each of M independent trials, a campaign of T rounds is simulated:
  - N_POOL devices, each independently available each round with
    probability P_AVAIL (Definition 1). The aggregator recruits up to
    K_CAP of the available devices per round, the realistic
    infrastructure cap Section 10.5 argues for, not a literal p*N.
  - Each device's active power is drawn once per trial (log-normal around
    the Section 10.4 point estimate) to represent real device-mix
    heterogeneity (older versus newer phones), not resampled every round.
  - Each participation event draws Wi-Fi versus cellular stochastically
    (CELLULAR_FRAC), representing real connectivity variation.
  - Cumulative carbon is tracked round by round for FMCL, and over the
    same calendar span for a continuously powered dedicated-edge server
    and an elastically billed cloud service.

The simulated crossover round (from simulated trajectories, across
trials) is compared against Theorem 4's closed-form T*_min evaluated at
the expected values of the same random quantities.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "carbon_accounting"))
from real_accounting import (P_IDLE_W, WIFI_WH_PER_MB, PAYLOAD_MB, ROUND_SECONDS,
                              SERVER_ACTIVE_W, SERVER_IDLE_FRACTION, EDGE_PUE,
                              CLOUD_PUE, CLOUD_COMPUTE_WH_PER_ROUND, CLOUD_DATA_WH_PER_ROUND,
                              CLOUD_EMBODIED_SHARE_KG, SERVER_EMBODIED_KG_CONFIGURED,
                              I_GRID_POINT, CELLULAR_RATIO_LOW, CELLULAR_RATIO_HIGH)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# Campaign parameters (a realistic single-deployment scale, distinct from
# Table 5's population-scale N; see Section 10.5 for why K is capped
# independently of N rather than set to p*N)
# ---------------------------------------------------------------------
N_POOL = 2000
P_AVAIL = 0.15            # Definition 1: per-round device availability probability
K_CAP = 300               # realistic per-round aggregator cap (Section 10.5)
T_ROUNDS = 2000
CADENCE_HOURS = 1.0       # hourly rounds, so calendar span equals T_ROUNDS hours
CELLULAR_FRAC = 0.3       # stochastic per participation event
DEVICE_POWER_CV = 0.25    # device-to-device heterogeneity in active power
M_TRIALS = 300

RNG_SEED = 42


def run_trial(rng):
    # Each device's active power is drawn once for this trial (fixed device mix)
    device_active_w = rng.lognormal(mean=np.log(4.0), sigma=DEVICE_POWER_CV, size=N_POOL)
    marginal_w = np.maximum(device_active_w - P_IDLE_W, 0.1)

    cellular_ratio = rng.uniform(CELLULAR_RATIO_LOW, CELLULAR_RATIO_HIGH)  # fixed per trial
    wifi_wh = WIFI_WH_PER_MB * PAYLOAD_MB

    cum_fmcl = np.zeros(T_ROUNDS)
    total = 0.0
    for t in range(T_ROUNDS):
        available = rng.random(N_POOL) < P_AVAIL
        avail_idx = np.nonzero(available)[0]
        if len(avail_idx) > K_CAP:
            chosen = rng.choice(avail_idx, size=K_CAP, replace=False)
        else:
            chosen = avail_idx

        if len(chosen) > 0:
            train_wh = marginal_w[chosen] * (ROUND_SECONDS / 3600.0)
            is_cellular = rng.random(len(chosen)) < CELLULAR_FRAC
            comm_wh = np.where(is_cellular, wifi_wh * cellular_ratio, wifi_wh)
            round_wh = np.sum(train_wh + comm_wh)
        else:
            round_wh = 0.0

        total += round_wh / 1000.0 * I_GRID_POINT  # g CO2e
        cum_fmcl[t] = total

    # Edge: continuous draw over the same calendar span (T_ROUNDS * CADENCE_HOURS)
    calendar_hours = np.arange(1, T_ROUNDS + 1) * CADENCE_HOURS
    active_wh_per_round = SERVER_ACTIVE_W * (ROUND_SECONDS / 3600.0)
    idle_wh_per_hour = SERVER_ACTIVE_W * SERVER_IDLE_FRACTION
    cum_edge = (np.arange(1, T_ROUNDS + 1) * active_wh_per_round +
                calendar_hours * idle_wh_per_hour) * EDGE_PUE / 1000.0 * I_GRID_POINT
    phi_edge_g = SERVER_EMBODIED_KG_CONFIGURED * 1000.0
    cum_edge_total = cum_edge + phi_edge_g

    # Cloud: per-round elastic cost, no idle penalty
    cloud_wh_per_round = (CLOUD_COMPUTE_WH_PER_ROUND + CLOUD_DATA_WH_PER_ROUND) * CLOUD_PUE
    cum_cloud = np.arange(1, T_ROUNDS + 1) * cloud_wh_per_round / 1000.0 * I_GRID_POINT
    phi_cloud_g = CLOUD_EMBODIED_SHARE_KG * 1000.0
    cum_cloud_total = cum_cloud + phi_cloud_g

    cum_fmcl_total = cum_fmcl  # Phi_FMCL approximately 0 under P1

    return cum_fmcl_total, cum_edge_total, cum_cloud_total


def find_crossover(cum_fmcl, cum_other):
    """First round index (1-based) where FMCL's cumulative carbon exceeds
    the other architecture's; None if it never does within the campaign."""
    exceeds = cum_fmcl > cum_other
    if not np.any(exceeds):
        return None
    return int(np.argmax(exceeds)) + 1


if __name__ == "__main__":
    rng_master = np.random.default_rng(RNG_SEED)
    crossovers_edge = []
    crossovers_cloud = []
    all_fmcl_trajectories = []

    for trial in range(M_TRIALS):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        cum_fmcl, cum_edge, cum_cloud = run_trial(rng)
        all_fmcl_trajectories.append(cum_fmcl)
        crossovers_edge.append(find_crossover(cum_fmcl, cum_edge))
        crossovers_cloud.append(find_crossover(cum_fmcl, cum_cloud))

    all_fmcl_trajectories = np.array(all_fmcl_trajectories)

    n_no_cross_edge = sum(1 for c in crossovers_edge if c is None)
    n_no_cross_cloud = sum(1 for c in crossovers_cloud if c is None)
    finite_edge = [c for c in crossovers_edge if c is not None]
    finite_cloud = [c for c in crossovers_cloud if c is not None]

    print(f"=== Monte Carlo campaign simulation: {M_TRIALS} trials, {T_ROUNDS} rounds each ===")
    print(f"N_pool={N_POOL}, p_avail={P_AVAIL}, K_cap={K_CAP}, cellular_frac={CELLULAR_FRAC}")
    print()
    print(f"Trials where FMCL never exceeds EDGE's cumulative carbon within {T_ROUNDS} rounds: "
          f"{n_no_cross_edge}/{M_TRIALS} ({100*n_no_cross_edge/M_TRIALS:.1f}%)")
    if finite_edge:
        print(f"  Empirical crossover-vs-edge round: median={np.median(finite_edge):.0f}, "
              f"90% CI=[{np.percentile(finite_edge,5):.0f}, {np.percentile(finite_edge,95):.0f}]")
    print()
    print(f"Trials where FMCL never exceeds CLOUD's cumulative carbon within {T_ROUNDS} rounds: "
          f"{n_no_cross_cloud}/{M_TRIALS} ({100*n_no_cross_cloud/M_TRIALS:.1f}%)")
    if finite_cloud:
        print(f"  Empirical crossover-vs-cloud round: median={np.median(finite_cloud):.0f}, "
              f"90% CI=[{np.percentile(finite_cloud,5):.0f}, {np.percentile(finite_cloud,95):.0f}]")

    # Analytical Theorem 4 prediction at expected values
    expected_K = min(N_POOL * P_AVAIL, K_CAP)
    from real_accounting import fmcl_per_round, edge_per_round, cloud_per_round, phi_edge, phi_cloud
    c_f = fmcl_per_round(int(expected_K), cellular_fraction=CELLULAR_FRAC,
                          cellular_ratio=(CELLULAR_RATIO_LOW + CELLULAR_RATIO_HIGH) / 2)
    c_e = edge_per_round(inter_round_hours=CADENCE_HOURS)
    c_c = cloud_per_round()
    phi_e, phi_c = phi_edge(), phi_cloud()
    T_edge_star = phi_e / (c_f - c_e) if c_f > c_e else float('inf')
    T_cloud_star = phi_c / (c_f - c_c) if c_f > c_c else float('inf')
    print()
    print(f"Analytical Theorem 4 prediction at E[K]={expected_K:.0f}: "
          f"T*_edge={T_edge_star:.0f}, T*_cloud={T_cloud_star:.1f}")

    np.savez(os.path.join(HERE, "mc_results.npz"),
             all_fmcl=all_fmcl_trajectories,
             crossovers_edge=np.array([c if c is not None else -1 for c in crossovers_edge]),
             crossovers_cloud=np.array([c if c is not None else -1 for c in crossovers_cloud]))
