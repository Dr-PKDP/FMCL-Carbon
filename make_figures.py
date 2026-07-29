"""
make_figures.py
===============
Generates Figures 4 and 5 from Section 11 of:

  "Federated Mobile Crowd Learning: Convergence, Privacy, and
   Lifecycle Carbon Guarantees"

Figure 4  —  Stationarity gap vs. training round at four participation rates.
             Confirms the floor predicted by Theorem 1 rises as p falls.

Figure 5  —  Measured stationarity floor vs. 1/p with a linear fit.
             Confirms the floor scales as 1/p (Lemma 1), r ≈ 0.97.

Usage
-----
Run floor_scaling.py FIRST (it generates floor_data.npy).
Then run this script:

    python floor_scaling.py     # takes 3-8 minutes
    python make_figures.py      # takes 2-5 minutes

Output files
------------
figure4_convergence.png
figure5_floor_scaling.png

Both are saved in the same folder as this script at 200 dpi.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt

from simulation import make_federated_data, run

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared dataset  (same settings as floor_scaling.py)
# ---------------------------------------------------------------------------
print("Generating federated dataset …")
clients, K, dim = make_federated_data(
    N=200, K=4, dim=10, n_per=50, alpha=0.3, seed=0
)
print(f"  N={len(clients)} clients, K={K} classes, dim={dim}\n")


# ===========================================================================
# Figure 4 — Convergence curves at p ∈ {1.0, 0.5, 0.2, 0.1}
# ===========================================================================
print("Running Figure 4 experiment (4 participation rates × 5 seeds × 150 rounds) …")

PS4    = [1.0, 0.5, 0.2, 0.1]
SEEDS4 = 5
T4     = 150

curves = {}
for p in PS4:
    cs = [run(clients, K, dim, p=p, T=T4, sigma=0.02, seed=s) for s in range(SEEDS4)]
    curves[p] = np.mean(cs, axis=0)
    print(f"  p = {p:.1f}   late-round floor = {curves[p][-20:].mean():.4e}")

# --- plot ---
fig, ax = plt.subplots(figsize=(6.2, 4.2))
colors = plt.cm.viridis(np.linspace(0, 0.85, len(PS4)))

for p, c in zip(PS4, colors):
    ax.semilogy(curves[p], color=c, lw=1.8, label=f"p = {p}")

ax.set_xlabel("Training round $t$")
ax.set_ylabel(r"$\|\nabla F(w^t)\|^2$")
ax.set_title("Stationarity gap vs. training round\n"
             "(log scale, mean of 5 seeds)")
ax.legend(title="participation rate", frameon=False, fontsize=9)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()

out4 = os.path.join(THIS_DIR, "figure4_convergence.png")
fig.savefig(out4, dpi=200)
plt.close(fig)
print(f"\nSaved  →  {out4}")


# ===========================================================================
# Figure 5 — Floor vs. 1/p  (requires floor_data.npy from floor_scaling.py)
# ===========================================================================
floor_path = os.path.join(THIS_DIR, "floor_data.npy")

if not os.path.exists(floor_path):
    print("\n*** floor_data.npy not found. ***")
    print("    Run  python floor_scaling.py  first, then re-run this script.")
else:
    fd = np.load(floor_path, allow_pickle=True).item()
    ps2    = fd["ps"]
    meas   = fd["measured"]
    r_val  = fd.get("pearson_r", np.corrcoef(meas, 1.0 / ps2)[0, 1])

    # Linear fit through (1/p, floor)
    inv_p  = 1.0 / ps2
    A      = np.vstack([inv_p, np.ones_like(inv_p)]).T
    slope, intercept = np.linalg.lstsq(A, meas, rcond=None)[0]
    xx     = np.linspace(inv_p.min(), inv_p.max(), 200)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(inv_p, meas, "o-",
            color="#c0392b", lw=1.6, ms=6, label="measured floor")
    ax.plot(xx, slope * xx + intercept, "--",
            color="#2c3e50", lw=1.3,
            label=f"linear fit  (r = {r_val:.2f})")

    ax.set_xlabel(r"$1/p$  (inverse participation rate)")
    ax.set_ylabel(r"stationarity floor  $\overline{\|\nabla F\|^2}$")
    ax.set_title(r"Floor scales as $1/p$  (Lemma 1)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out5 = os.path.join(THIS_DIR, "figure5_floor_scaling.png")
    fig.savefig(out5, dpi=200)
    plt.close(fig)
    print(f"Saved  →  {out5}")

print("\nDone.")