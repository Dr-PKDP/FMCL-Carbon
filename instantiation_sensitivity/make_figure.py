"""
Generates Fig. 5: total-order Sobol sensitivity of T*_min (top row) and
the resulting distribution against the paper's point estimate (bottom
row), for both worked instantiations of Section 10.3.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

from sensitivity import SCENARIOS, PARAM_NAMES, bounds_for, t_star_min, sobol_response

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "text.color": "black", "axes.labelcolor": "black",
                     "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "axes.edgecolor": "black", "legend.labelcolor": "black", "svg.fonttype": "none",
                     "savefig.dpi": 300})

HERE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1))

scenario_meta = [
    ("activity_recognition", "Activity recognition ($N = 10^6$)", axes[0, 0], axes[1, 0], "#2E86AB"),
    ("healthcare_mitigated", "Healthcare, mitigated ($N = 2\\times10^5$)", axes[0, 1], axes[1, 1], "#A23B72"),
]

for key, label, ax_bar, ax_hist, color in scenario_meta:
    point = SCENARIOS[key]
    b = bounds_for(point)
    problem = {"num_vars": 5, "names": PARAM_NAMES, "bounds": [b[p] for p in PARAM_NAMES]}
    param_values = saltelli.sample(problem, 2048, calc_second_order=False)
    Y = t_star_min(param_values[:, 0], param_values[:, 1], param_values[:, 2],
                    param_values[:, 3], param_values[:, 4])
    Si = sobol.analyze(problem, sobol_response(Y), calc_second_order=False, print_to_console=False)

    display_names = [r"$\Phi_{edge}$", r"$\Phi_{cloud}$", r"$c_{FMCL}$", r"$c_{edge}$", r"$c_{cloud}$"]
    ax_bar.barh(display_names, Si['ST'], color=color)
    ax_bar.set_xlabel("Total-order Sobol index $S_T$")
    ax_bar.set_title(f"{label}\nDrivers of variance in $\\log(1+T^*_{{min}})$")
    ax_bar.set_xlim(0, 1)
    ax_bar.tick_params(labelsize=8.5, colors="black")

    Yc = Y[Y < 1e6]
    ax_hist.hist(Yc, bins=np.geomspace(Yc.min(), Yc.max(), 61), color=color, alpha=0.85)
    # Consistent convention across Fig. 5 and Fig. 6:
    #   black dashed = the paper's analytical point estimate
    #   orange dotted = the simulated/empirical median
    ax_hist.axvline(point['paper_T_star_min'], color="black", linestyle="--", linewidth=2,
                     label=f"Point estimate ({point['paper_T_star_min']:,} rounds)")
    sim_median = np.median(Y)
    ax_hist.axvline(sim_median, color="#ff7f0e", linestyle=":", linewidth=2.4,
                     label=f"Simulated median ({sim_median:,.0f} rounds)")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel(r"$T^*_{min}$ (rounds, log scale)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.tick_params(labelsize=8.5, colors="black")
    ax_hist.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, frameon=False, handlelength=2.6)

fig.tight_layout(h_pad=0.8)
fig.savefig(os.path.join(HERE, "fig5_sobol_sensitivity.svg"), bbox_inches="tight")
print("saved:", os.path.join(HERE, "fig5_sobol_sensitivity.svg"))
