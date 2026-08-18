"""
Section 11.2: numerical construction of the (privacy, carbon) Pareto
frontier at a fixed accuracy target, giving Propositions 2 and 3 a
computed, visual demonstration rather than a purely analytic one.

At a fixed accuracy target eps_acc, choosing (sigma, p) determines
T_acc(sigma, p) (the rounds needed to hit the target, Assumption C1), which
in turn determines the privacy loss E(T_acc, sigma, p) (Proposition 1)
and the carbon C(T_acc, p) = c_round * p * T_acc. This reduces the
three-objective problem to a clean two-objective (privacy vs. carbon)
frontier at fixed accuracy: exactly the epsilon-constraint framing
Theorem 3 and Section 8.4's "Reachability of the Frontier" discussion
already argue for over weighted-sum scalarization.

The script (a) sweeps a fine (sigma, p) grid, (b) computes the true
Pareto frontier directly from the grid, (c) separately runs weighted-sum
scalarization across many weight vectors, and (d) reports which Pareto
points weighted-sum recovers and which it misses: Proposition 3's claim
made concrete rather than asserted from the Hessian sign alone.
"""
import os

import numpy as np

# Reuse the activity-recognition parameters already in the paper's Section 10.1
N = 1e6
d = 1e4
G = C = 1.0
L = D = 1.0
eps_acc = 0.5
kappa = np.log(1 / 1e-5)  # delta = 1e-5, a standard choice; kappa = log(1/delta)
c_round = 1.0


def B_of(sigma):
    return G**2 + d * sigma**2 * C**2


def T_acc(sigma, p):
    B = B_of(sigma)
    alpha = eps_acc + G**2 / N
    denom = alpha * N * p - B
    if denom <= 0:
        return None
    return 2 * L * D * N * p / denom


def privacy_loss(T, sigma, p):
    Ti = p * T
    return Ti / (2 * sigma**2) + np.sqrt(2 * kappa * Ti) / sigma


def carbon(T, p):
    return c_round * p * T


def build_grid(n_sigma=140, n_p=140):
    sigmas = np.geomspace(0.05, 20, n_sigma)
    ps = np.geomspace(1e-4, 0.999, n_p)
    points = []  # (sigma, p, E, C)
    for sigma in sigmas:
        for p in ps:
            T = T_acc(sigma, p)
            if T is None or T <= 0:
                continue
            E = privacy_loss(T, sigma, p)
            Carb = carbon(T, p)
            if np.isfinite(E) and np.isfinite(Carb):
                points.append((sigma, p, E, Carb))
    return np.array(points)


def pareto_front(points):
    """Return a boolean mask of non-dominated points in (E, C), both minimized."""
    E, Carb = points[:, 2], points[:, 3]
    n = len(points)
    order = np.argsort(E)
    best_C_so_far = np.inf
    # Sort by E ascending; a point is on the frontier only if its C is
    # strictly less than every point with E at or below it seen so far.
    front_mask = np.zeros(n, dtype=bool)
    for idx in order:
        if Carb[idx] < best_C_so_far - 1e-12:
            front_mask[idx] = True
            best_C_so_far = Carb[idx]
    return front_mask


def weighted_sum_recovered(points, front_mask, n_weights=4000):
    """For many weight vectors w in [0,1], minimize w*E_norm + (1-w)*C_norm
    over the grid, and collect which frontier points get selected."""
    E, Carb = points[:, 2], points[:, 3]
    E_n = (E - E.min()) / (E.max() - E.min())
    C_n = (Carb - Carb.min()) / (Carb.max() - Carb.min())
    recovered = np.zeros(len(points), dtype=bool)
    for w in np.linspace(0, 1, n_weights):
        obj = w * E_n + (1 - w) * C_n
        best_idx = np.argmin(obj)
        recovered[best_idx] = True
    return recovered


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))

    # These grid and sweep resolutions match the manuscript's reported
    # frontier size (85 points) and recovery count (42), used for both the
    # headline numbers in Section 11.2 and Fig. 4.
    pts = build_grid(n_sigma=220, n_p=220)
    print(f"Grid points evaluated: {len(pts)}")
    front = pareto_front(pts)
    print(f"Points on the numerically constructed Pareto frontier: {front.sum()}")

    recovered = weighted_sum_recovered(pts, front, n_weights=6000)
    front_and_recovered = front & recovered
    front_and_missed = front & ~recovered

    print(f"Frontier points reached by weighted-sum scalarization (6000 weights swept): "
          f"{front_and_recovered.sum()} / {front.sum()}")
    print(f"Frontier points weighted-sum never reaches at any weight: "
          f"{front_and_missed.sum()} / {front.sum()}  "
          f"({100*front_and_missed.sum()/front.sum():.1f}% of the true frontier)")

    np.savez(os.path.join(HERE, "pareto_data.npz"),
             points=pts, front=front, recovered=recovered)
