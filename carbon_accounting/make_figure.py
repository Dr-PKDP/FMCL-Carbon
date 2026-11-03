"""
Generates Fig. 3: per-round operational carbon vs. K (left panel) and
total-order Sobol sensitivity of the crossover horizon to each real-data
parameter (right panel).

Run real_data_sobol.py first if you want the right-panel bars regenerated
from a fresh sample; the values below are the ones reported in the
manuscript and match a saved run with the seed fixed in that script.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_accounting import fmcl_per_round, edge_per_round, cloud_per_round

HERE = os.path.dirname(os.path.abspath(__file__))


plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "text.color": "black", "axes.labelcolor": "black",
                     "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "axes.edgecolor": "black", "legend.labelcolor": "black", "svg.fonttype": "none",
                     "savefig.dpi": 300})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"width_ratios": [1, 0.95]})

# Panel 1: per-round FMCL rate vs K, at fixed cellular fraction, against edge and cloud
ax = axes[0]
K_range = np.linspace(10, 600, 80)
c_fmcl_wifi = [fmcl_per_round(int(k), cellular_fraction=0.0) for k in K_range]
c_fmcl_cell = [fmcl_per_round(int(k), cellular_fraction=1.0) for k in K_range]
c_edge_1h = edge_per_round(inter_round_hours=1.0)
c_edge_24h = edge_per_round(inter_round_hours=24.0)
c_cloud_val = cloud_per_round()
ax.plot(K_range, c_fmcl_wifi, '-', color="#2E86AB", linewidth=2.5, label=r"$c_{FMCL}$ (Wi-Fi)")
ax.plot(K_range, c_fmcl_cell, color="#2E86AB", linewidth=2.5, linestyle=':', label=r"$c_{FMCL}$ (100% cellular)")
ax.axhline(c_edge_1h, color="#C1272D", linewidth=2, linestyle='--', label=r"$c_{edge}$ (hourly rounds)")
ax.axhline(c_edge_24h, color="#ff7f0e", linewidth=2, linestyle='-.', label=r"$c_{edge}$ (daily rounds)")
ax.axhline(c_cloud_val, color="#6A4C93", linewidth=2.5, label=r"$c_{cloud}$")
ax.set_xlabel("$K$: devices per round")
ax.set_ylabel("Per-round carbon (g CO$_2$e)")
ax.set_ylim(1, 1e4)
ax.set_yscale("log")
ax.axvline(303, color="black", linewidth=0.8, linestyle="-.")
ax.text(296, 620, "$K \\approx 303$:\n$c_{FMCL} = c_{cloud}$\n(30% cellular)", fontsize=8.5, va="center", ha="right")
ax.set_title("Per-round carbon against $K$")
ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, handlelength=2.6)

# Panel 2: Sobol total-order indices, K included and dominant
# Values from real_data_sobol.py (seed=11, 2048 base samples, 1000 bootstrap resamples)
ax2 = axes[1]
names = ["Server embodied\ncarbon (LCA)", "Grid carbon\nintensity", "Cellular\nfraction",
         "Cellular/Wi-Fi\nenergy ratio", "Training\ncadence", "Edge\nPUE", "$K$: devices\nper round"]
ST = [0.000, 0.002, 0.009, 0.003, 0.000, 0.000, 0.998]
colors_bar = ["#2E86AB"] * 7
colors_bar[2] = "#F18F01"
colors_bar[3] = "#F18F01"
colors_bar[6] = "#C1272D"
ax2.barh(names, ST, color=colors_bar)
for y, v in enumerate(ST):     # tiny bars are invisible, so the index is printed beside each one
    ax2.text(v + 0.012, y, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color="black")
ax2.set_xlabel("Total-order Sobol index $S_T$")
ax2.set_title("Sensitivity of crossover horizon")
ax2.set_xlim(0, 1.12)
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.invert_yaxis()

fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.30, wspace=0.8)
fig.savefig(os.path.join(HERE, "fig3_real_data.svg"), bbox_inches="tight")
print("saved:", os.path.join(HERE, "fig3_real_data.svg"))
