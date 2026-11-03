"""
Generates Fig. 6: simulated cumulative carbon for FMCL against edge and
cloud (left panel), and the simulated crossover-round distribution
against Theorem 3's closed-form prediction (right panel).

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
from campaign import T_ROUNDS, CADENCE_HOURS, theorem3_prediction, N_POOL, K_CAP
_EK, _PW, _TE, T_PRED = theorem3_prediction()

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


plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "text.color": "black", "axes.labelcolor": "black",
                     "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "axes.edgecolor": "black", "legend.labelcolor": "black", "svg.fonttype": "none",
                     "savefig.dpi": 300})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

ax = axes[0]
ax.plot(rounds, fmcl_median, color="#2E86AB", linewidth=2, label="FMCL (median, 300 trials)")
ax.plot(rounds, cum_edge, color="#C1272D", linewidth=2, linestyle='--', label="Dedicated edge")
ax.plot(rounds, cum_cloud, color="#6A4C93", linewidth=2, label="Cloud")
ax.axvline(T_PRED, color="black", linestyle="--", linewidth=1.6, label="Theorem 3 crossover")
ax.axvline(np.median(finite_cloud), color="#ff7f0e", linestyle=":", linewidth=2.2, label="Simulated median crossover")
ax.set_xlabel("Training round")
ax.set_ylabel("Cumulative carbon (g CO$_2$e)")
ax.set_yscale("log")
ax.set_xlim(0, 500)
ax.set_title("Cumulative carbon")
h, l = ax.get_legend_handles_labels()   # order: FMCL, Dedicated edge, Cloud, Theorem 3 crossover, Simulated median crossover
blank = plt.Line2D([], [], color="none")
ax.legend(handles=[h[0], h[1], h[4], h[2], h[3], blank], labels=[l[0], l[1], l[4], l[2], l[3], ""],
          fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False,
          handlelength=2.2, columnspacing=2.2)

band_rel = (fmcl_p95 - fmcl_p5) / fmcl_median
print(f"FMCL 5th-95th band: {band_rel.min()*100:.1f}%-{band_rel.max()*100:.1f}% of median across rounds")

ax2 = axes[1]
ax2.hist(finite_cloud, bins=30, color="#6A4C93", alpha=0.85, edgecolor="white", linewidth=0.4, label="Simulated crossover rounds (300 trials)")
ax2.axvline(T_PRED, color="black", linestyle="--", linewidth=2,
            label=f"Theorem 3 prediction ({T_PRED:.1f}, $E[K]$ = {_EK:.0f})")
ax2.axvline(np.median(finite_cloud), color="#ff7f0e", linestyle=":", linewidth=2.4,
            label=f"Simulated median ({np.median(finite_cloud):.0f} rounds)")
ax2.set_xlabel("Round at which FMCL first exceeds cloud")
ax2.set_ylabel("Trials")
ax2.set_title("Crossover round")
ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=False, handlelength=2.2)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig6_monte_carlo.svg"), bbox_inches="tight")
print("saved:", os.path.join(HERE, "fig6_monte_carlo.svg"))
