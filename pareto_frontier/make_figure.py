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

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

for ax, xlim, title in [(axes[0], None, "Full frontier (log-log)"),
                         (axes[1], (2.5, 15), "Zoomed: low-privacy-loss region")]:
    ax.scatter(E[~front], Carb[~front], s=4, color="#cfd8e3", alpha=0.5,
               label="Dominated operating points")
    ax.scatter(E[front_reached], Carb[front_reached], s=30, color="#2E86AB",
               label=f"Reached by weighted-sum (n={front_reached.sum()})", zorder=3)
    ax.scatter(E[front_missed], Carb[front_missed], s=40, color="#C1272D", marker="D",
               label=f"Missed at every weight (n={front_missed.sum()})", zorder=3)
    ax.set_yscale("log")
    if xlim is None:
        ax.set_xscale("log")
    else:
        ax.set_xlim(*xlim)
    ax.set_xlabel(r"Privacy loss $\mathcal{E}(T_{acc},\sigma,p)$")
    ax.set_ylabel(r"Carbon $\mathbb{C}(T_{acc},p)$  (log scale)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")

fig.suptitle("Proposition 3 in practice: the Pareto frontier at fixed accuracy, "
             f"{100*front_missed.sum()/front.sum():.0f}% of frontier points unreachable "
             "by weighted-sum scalarization at any weight", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(HERE, "fig4_pareto_frontier.png"), dpi=150)
print("saved:", os.path.join(HERE, "fig4_pareto_frontier.png"))
print("frontier size:", front.sum(), " missed:", front_missed.sum())
