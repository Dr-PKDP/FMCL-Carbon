"""
floor_scaling.py
================
Runs the floor-scaling experiment (Section 11.3) and saves the results
that make_figures.py uses to draw Figure 5.

What it tests
-------------
Theorem 1 / Lemma 1 predict that the stationarity floor scales as 1/p
(and more precisely as 1/(Np) when DP noise dominates).  This script
measures the empirical floor at seven participation rates, fits a linear
model to floor vs 1/p, and reports the Pearson correlation.

How long it takes
-----------------
Approximately 3–8 minutes on a modern laptop (56 training runs of
200 rounds each).  A progress line is printed for each participation rate.

Output
------
floor_data.npy  —  saved in the same folder as this script.
                   Loaded by make_figures.py to draw Figure 5.

Usage
-----
    python floor_scaling.py
"""

import numpy as np
from simulation import make_federated_data, run

# ---------------------------------------------------------------------------
# Settings  (match the paper's Section 11.1)
# ---------------------------------------------------------------------------
N_CLIENTS  = 200
K_CLASSES  = 4
DIM        = 10
N_PER      = 50
ALPHA      = 0.3     # Dirichlet concentration

SIGMA      = 0.02    # DP noise multiplier  (small but enough to dominate floor)
CLIP       = 5.0     # gradient clip bound  C
T_ROUNDS   = 200     # training rounds per run
TAIL       = 30      # rounds used to estimate the floor (last TAIL rounds)
N_SEEDS    = 15       # independent replications per participation rate

PS = np.array([1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1])

# ---------------------------------------------------------------------------
# Generate the shared dataset once
# ---------------------------------------------------------------------------
print("Generating federated dataset …")
clients, K, dim = make_federated_data(
    N=N_CLIENTS, K=K_CLASSES, dim=DIM,
    n_per=N_PER, alpha=ALPHA, seed=0
)
print(f"  N={len(clients)} clients, K={K} classes, dim={dim}\n")

# ---------------------------------------------------------------------------
# Measure empirical floor at each participation rate
# ---------------------------------------------------------------------------
measured = []

for p in PS:
    runs = [
        run(clients, K, dim, p=p, T=T_ROUNDS,
            sigma=SIGMA, clip=CLIP, seed=s)
        for s in range(N_SEEDS)
    ]
    floor = np.mean([r[-TAIL:].mean() for r in runs])
    measured.append(floor)
    print(f"p = {p:.2f}   floor = {floor:.4e}   1/(Np) = {1/(len(clients)*p):.4e}")

measured = np.array(measured)

# ---------------------------------------------------------------------------
# Report the Pearson correlation
# ---------------------------------------------------------------------------
r = np.corrcoef(measured, 1.0 / PS)[0, 1]
print(f"\nPearson corr(floor, 1/p) = {r:.4f}")

# ---------------------------------------------------------------------------
# Save for make_figures.py
# ---------------------------------------------------------------------------
import os
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "floor_data.npy")
np.save(out_path, {
    "ps":       PS,
    "measured": measured,
    "sigma":    SIGMA,
    "N":        len(clients),
    "pearson_r": r,
}, allow_pickle=True)
print(f"\nSaved  →  {out_path}")