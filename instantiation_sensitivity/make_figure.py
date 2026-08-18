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

from sensitivity import SCENARIOS, PARAM_NAMES, bounds_for, t_star_min

# Consistent black-text styling across all manuscript figures.
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.titlecolor'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['legend.labelcolor'] = 'black'
mpl.rcParams['figure.titlesize'] = 11
mpl.rcParams['figure.titleweight'] = 'normal'

HERE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

scenario_meta = [
    ("activity_recognition", "Activity recognition\n(N=10^6)", axes[0, 0], axes[1, 0], "#2E86AB"),
    ("healthcare_mitigated", "Healthcare, mitigated\n(N=2x10^5)", axes[0, 1], axes[1, 1], "#A23B72"),
]

for key, label, ax_bar, ax_hist, color in scenario_meta:
    point = SCENARIOS[key]
    b = bounds_for(point)
    problem = {"num_vars": 5, "names": PARAM_NAMES, "bounds": [b[p] for p in PARAM_NAMES]}
    param_values = saltelli.sample(problem, 2048, calc_second_order=False)
    Y = t_star_min(param_values[:, 0], param_values[:, 1], param_values[:, 2],
                    param_values[:, 3], param_values[:, 4])
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    display_names = [r"$\Phi_{edge}$", r"$\Phi_{cloud}$", r"$c_{FMCL}$", r"$c_{edge}$", r"$c_{cloud}$"]
    ax_bar.barh(display_names, Si['ST'], color=color)
    ax_bar.set_xlabel("Total-order Sobol index $S_T$", fontsize=10, color="black")
    ax_bar.set_title(f"{label}\nWhat drives uncertainty in $T^*_{{min}}$", fontsize=10, color="black")
    ax_bar.set_xlim(0, 1)
    ax_bar.tick_params(labelsize=9.5, colors="black")

    ax_hist.hist(Y[Y < 1e6], bins=60, color=color, alpha=0.75)
    # Consistent convention across Fig. 5 and Fig. 6:
    #   black dashed = the paper's analytical point estimate
    #   grey dotted  = the simulated/empirical median
    ax_hist.axvline(point['paper_T_star_min'], color="black", linestyle="--", linewidth=2,
                     label=f"Paper's point estimate\n({point['paper_T_star_min']:,} rounds)")
    sim_median = np.median(Y[Y < 1e6])
    ax_hist.axvline(sim_median, color="gray", linestyle=":", linewidth=2,
                     label=f"Simulated median\n({sim_median:,.0f} rounds)")
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel(r"$T^*_{min}$ (rounds, log scale)", fontsize=10, color="black")
    ax_hist.set_ylabel("Frequency", fontsize=10, color="black")
    ax_hist.tick_params(labelsize=9.5, colors="black")
    ax_hist.legend(fontsize=8, labelcolor="black")

fig.suptitle("Propagating parameter uncertainty through Theorem 4's crossover criterion\n"
             "(illustrative point estimates from Section 10.3, not the measured figures of Section 10.4)",
             fontsize=11.5, color="black")
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(os.path.join(HERE, "fig5_sobol_sensitivity.png"), dpi=150)
print("saved:", os.path.join(HERE, "fig5_sobol_sensitivity.png"))
