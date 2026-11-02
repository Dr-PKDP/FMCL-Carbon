"""
Generates Fig. 4: the numerically constructed Pareto frontier at fixed
accuracy, with the points weighted-sum scalarization reaches and the
points it misses at every weight shown separately.

Run pareto_frontier.py first, or run this script directly; it builds the
grid itself if pareto_data.npz is not present.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from pareto_frontier import build_grid, pareto_front, weighted_sum_recovered

HERE = os.path.dirname(os.path.abspath(__file__))

pts = build_grid(n_sigma=220, n_p=220)
front = pareto_front(pts)
recovered = weighted_sum_recovered(pts, front, n_weights=6000)

E, Carb = pts[:, 2], pts[:, 3]
front_reached = front & recovered
front_missed = front & ~recovered


plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "text.color": "black", "axes.labelcolor": "black",
                     "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "axes.edgecolor": "black", "legend.labelcolor": "black", "svg.fonttype": "none",
                     "savefig.dpi": 300})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

for ax, xlim, title in [(axes[0], None, "Full frontier (log-log)"),
                         (axes[1], (2.5, 15), "Zoom: low privacy loss")]:
    ax.scatter(E[~front], Carb[~front], s=4, color="#aab4c3", alpha=0.6, rasterized=True,
               label="Dominated operating points")
    ax.scatter(E[front_reached], Carb[front_reached], s=30, color="#1f6f9a",
               label=f"Reached by weighted-sum (n={front_reached.sum()})", zorder=3)
    ax.scatter(E[front_missed], Carb[front_missed], s=40, color="#C1272D", marker="D", edgecolor="black", linewidth=0.4,
               label=f"Missed at every weight (n={front_missed.sum()})", zorder=3)
    ax.set_yscale("log")
    if xlim is None:
        ax.set_xscale("log")
    else:
        ax.set_xlim(*xlim)
    ax.set_xlabel(r"Privacy loss $\mathcal{E}(T_{acc},\sigma,p)$")
    ax.set_ylabel(r"Carbon $\mathbb{C}(T_{acc},p)$ (log scale)")
    ax.set_title(title)
    if xlim is not None:
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5, markerscale=1.5, columnspacing=1.0, handletextpad=0.3)

fig.tight_layout(rect=[0, 0.09, 1, 1])
fig.savefig(os.path.join(HERE, "fig4_pareto_frontier.svg"), bbox_inches="tight")
print("saved:", os.path.join(HERE, "fig4_pareto_frontier.svg"))
print("frontier size:", front.sum(), " missed:", front_missed.sum())
