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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: per-round FMCL rate vs K, at fixed cellular fraction, against edge and cloud
ax = axes[0]
K_range = np.linspace(10, 600, 80)
c_fmcl_wifi = [fmcl_per_round(int(k), cellular_fraction=0.0) for k in K_range]
c_fmcl_cell = [fmcl_per_round(int(k), cellular_fraction=1.0) for k in K_range]
c_edge_1h = edge_per_round(inter_round_hours=1.0)
c_edge_24h = edge_per_round(inter_round_hours=24.0)
c_cloud_val = cloud_per_round()
ax.plot(K_range, c_fmcl_wifi, '-', color="#2E86AB", linewidth=2.5, label="c_FMCL (Wi-Fi)")
ax.plot(K_range, c_fmcl_cell, color="#2E86AB", linewidth=2.5, linestyle=':', label="c_FMCL (100% cellular)")
ax.axhline(c_edge_1h, color="#C1272D", linestyle='--', label="c_edge (hourly rounds)")
ax.axhline(c_edge_24h, color="#C1272D", linestyle='--', alpha=0.4, label="c_edge (daily rounds)")
ax.axhline(c_cloud_val, color="#6A4C93", linewidth=2.5, label="c_cloud")
ax.set_xlabel("K: devices contributing per round")
ax.set_ylabel("Per-round operational carbon (g CO2e)")
ax.set_yscale("log")
ax.set_title("K, not network mode, sets whether FMCL\nis even competitive with cloud", fontsize=10)
ax.legend(fontsize=7, loc="upper left")

# Panel 2: Sobol total-order indices, K included and dominant
# Values from real_data_sobol.py (seed=11, 2048 base samples, 1000 bootstrap resamples)
ax2 = axes[1]
names = ["Server embodied\ncarbon (LCA)", "Grid carbon\nintensity", "Cellular\nfraction",
         "Cellular/Wi-Fi\nenergy ratio", "Training\ncadence", "Edge\nPUE", "K: devices\nper round"]
ST = [0.000, 0.003, 0.293, 0.092, 0.000, 0.000, 0.839]
colors_bar = ["#cfd8e3"] * 7
colors_bar[2] = "#F18F01"
colors_bar[3] = "#F4B860"
colors_bar[6] = "#C1272D"
ax2.barh(names, ST, color=colors_bar)
ax2.set_xlabel("Total-order Sobol index $S_T$")
ax2.set_title("With K properly treated as uncertain,\nit dominates, and Theorem 3 already sets it", fontsize=10)
ax2.set_xlim(0, 1)
ax2.invert_yaxis()

fig.suptitle("Real-data-grounded architecture comparison (Fairphone 5, Dell R740, LBNL/DOE, Google Cloud PUE)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(HERE, "fig3_real_data.png"), dpi=150)
print("saved:", os.path.join(HERE, "fig3_real_data.png"))
