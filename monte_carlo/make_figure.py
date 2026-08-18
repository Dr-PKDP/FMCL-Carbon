"""
Generates Fig. 6: simulated cumulative carbon for FMCL against edge and
cloud (left panel), and the simulated crossover-round distribution
against Theorem 4's closed-form prediction (right panel).

Run campaign.py first to produce mc_results.npz.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "carbon_accounting"))
from real_accounting import (SERVER_ACTIVE_W, SERVER_IDLE_FRACTION, EDGE_PUE, CLOUD_PUE,
                              CLOUD_COMPUTE_WH_PER_ROUND, CLOUD_DATA_WH_PER_ROUND,
                              CLOUD_EMBODIED_SHARE_KG, SERVER_EMBODIED_KG_CONFIGURED,
                              I_GRID_POINT, ROUND_SECONDS)

sys.path.insert(0, HERE)
from campaign import T_ROUNDS, CADENCE_HOURS

data = np.load(os.path.join(HERE, "mc_results.npz"))
all_fmcl = data['all_fmcl']            # (M_trials, T_ROUNDS)
crossovers_cloud = data['crossovers_cloud']
finite_cloud = crossovers_cloud[crossovers_cloud > 0]

rounds = np.arange(1, T_ROUNDS + 1)

# Reconstruct edge and cloud deterministic trajectories (same for every trial)
calendar_hours = rounds * CADENCE_HOURS
active_wh_per_round = SERVER_ACTIVE_W * (ROUND_SECONDS / 3600.0)
idle_wh_per_hour = SERVER_ACTIVE_W * SERVER_IDLE_FRACTION
cum_edge = (rounds * active_wh_per_round + calendar_hours * idle_wh_per_hour) * EDGE_PUE / 1000.0 * I_GRID_POINT
cum_edge += SERVER_EMBODIED_KG_CONFIGURED * 1000.0

cloud_wh_per_round = (CLOUD_COMPUTE_WH_PER_ROUND + CLOUD_DATA_WH_PER_ROUND) * CLOUD_PUE
cum_cloud = rounds * cloud_wh_per_round / 1000.0 * I_GRID_POINT
cum_cloud += CLOUD_EMBODIED_SHARE_KG * 1000.0

fmcl_median = np.median(all_fmcl, axis=0)
fmcl_p5 = np.percentile(all_fmcl, 5, axis=0)
fmcl_p95 = np.percentile(all_fmcl, 95, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.fill_between(rounds, fmcl_p5, fmcl_p95, color="#2E86AB", alpha=0.25, label="FMCL (90% band, 300 trials)")
ax.plot(rounds, fmcl_median, color="#2E86AB", linewidth=2, label="FMCL (median trajectory)")
ax.plot(rounds, cum_edge, color="#C1272D", linewidth=2, linestyle='--', label="Dedicated edge")
ax.plot(rounds, cum_cloud, color="#6A4C93", linewidth=2, label="Cloud")
ax.axvline(np.median(finite_cloud), color="#6A4C93", linestyle=':', alpha=0.7)
ax.axvline(161.6, color="gray", linestyle=':', alpha=0.7)
ax.set_xlabel("Training round")
ax.set_ylabel("Cumulative lifecycle carbon (g CO\u2082e)")
ax.set_yscale("log")
ax.set_xlim(0, 500)
ax.set_title("Simulated campaign: cumulative carbon\n(edge never catches up within 2,000 rounds)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")

ax2 = axes[1]
ax2.hist(finite_cloud, bins=30, color="#6A4C93", alpha=0.75, label="Empirical crossover round\n(300 simulated trials)")
ax2.axvline(161.6, color="black", linestyle="--", linewidth=2,
            label="Theorem 4 analytical prediction\n(161.6 rounds, at E[K]=300)")
ax2.axvline(np.median(finite_cloud), color="gray", linestyle=":", linewidth=2,
            label=f"Simulated median ({np.median(finite_cloud):.0f} rounds)")
ax2.set_xlabel("Round at which FMCL's cumulative carbon\nfirst exceeds cloud's")
ax2.set_ylabel("Trials")
ax2.set_title("Theory vs. simulation:\nclosed form lands inside the empirical distribution", fontsize=10)
ax2.legend(fontsize=8)

fig.suptitle("Monte Carlo campaign simulation: stochastic availability, device heterogeneity,\n"
             "and communication mode, against real hardware data (300 trials, 2,000 rounds each)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(HERE, "fig6_monte_carlo.png"), dpi=150)
print("saved:", os.path.join(HERE, "fig6_monte_carlo.png"))
