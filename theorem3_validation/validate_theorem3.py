"""
Section 11.1: numerical validation of Theorem 3 across many parameter regimes.

Theorem 3 claims: for fixed sigma and accuracy target eps_acc, the carbon-
minimizing participation rate is p* = 2*p_min in closed form, where
p_min = B / (alpha*N),  B = G^2 + d*sigma^2*C^2,  alpha = eps_acc + G^2/N.

The paper's own text (Section 8.4) reports this checked to four
significant figures against a grid search at one parameter point. This
validates it across a broad, randomized sweep of regimes (varying N,
eps_acc, sigma, d) and reports the worst-case deviation between the
closed form and a direct numerical optimizer, turning a single spot
check into a systematic result.
"""
import numpy as np
from scipy.optimize import minimize_scalar

rng = np.random.default_rng(7)
N_TRIALS = 2000

L, D, G, C = 1.0, 1.0, 1.0, 1.0  # normalized per the paper's own worked instantiations (Section 10.1)


def B_of(d, sigma):
    return G**2 + d * sigma**2 * C**2


def alpha_of(eps_acc, N):
    return eps_acc + G**2 / N


def p_min_of(d, sigma, eps_acc, N):
    return B_of(d, sigma) / (alpha_of(eps_acc, N) * N)


def carbon(p, d, sigma, eps_acc, N, c_round=1.0):
    """C(p) = c_round * p * T_acc(p); returns +inf outside the feasible region.
    T_acc(p) = 2LD / (alpha - B/(Np)) = 2LD*N*p / (alpha*N*p - B); the
    paper's own proof of Theorem 3 states both forms, and the second is
    used directly here."""
    B = B_of(d, sigma)
    alpha = alpha_of(eps_acc, N)
    denom = alpha * N * p - B
    if denom <= 0:
        return np.inf
    T_acc = 2 * L * D * N * p / denom
    return c_round * p * T_acc


def run_sweep():
    max_rel_err = 0.0
    infeasible_at_2pmin = 0
    worst_case = None
    log_records = []

    for _ in range(N_TRIALS):
        N = 10 ** rng.uniform(3, 7)               # population: 1e3 to 1e7
        d = 10 ** rng.uniform(1, 5)                # model dimension: 10 to 1e5
        sigma = 10 ** rng.uniform(-1, 1)           # noise multiplier: 0.1 to 10
        eps_acc = 10 ** rng.uniform(-1, 0.5)        # accuracy target: 0.1 to about 3.2

        p_min = p_min_of(d, sigma, eps_acc, N)
        p_star_closed_form = 2 * p_min
        if not (0 < p_star_closed_form < 1):
            continue  # outside a physically valid participation rate; skip

        # Numerically search for the carbon-minimizing p on (p_min, 1)
        lo = p_min * (1 + 1e-9)
        hi = min(1.0, p_min * 50)  # search window generous relative to p_min
        res = minimize_scalar(lambda p: carbon(p, d, sigma, eps_acc, N),
                               bounds=(lo, hi), method='bounded',
                               options={'xatol': 1e-12})
        p_star_numeric = res.x

        rel_err = abs(p_star_numeric - p_star_closed_form) / p_star_closed_form
        log_records.append((N, d, sigma, eps_acc, p_star_closed_form, p_star_numeric, rel_err))
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst_case = (N, d, sigma, eps_acc, p_star_closed_form, p_star_numeric, rel_err)

    return log_records, max_rel_err, worst_case


if __name__ == "__main__":
    records, max_rel_err, worst = run_sweep()
    print(f"Trials with a valid (0,1) participation window: {len(records)} / {N_TRIALS}")
    print(f"Maximum relative error between closed-form p* and numerical argmin: {max_rel_err:.2e}")
    print(f"Worst-case trial: N={worst[0]:.3g}, d={worst[1]:.3g}, sigma={worst[2]:.3g}, "
          f"eps_acc={worst[3]:.3g}")
    print(f"  closed-form p* = {worst[4]:.6f}, numeric argmin = {worst[5]:.6f}, "
          f"rel. error = {worst[6]:.2e}")

    print("\nRatio check (closed-form p* / p_min) across all trials:")
    ratio_vals = []
    for N, d, sigma, eps_acc, p_star_cf, p_star_num, rel_err in records:
        pm = p_min_of(d, sigma, eps_acc, N)
        ratio_vals.append(p_star_cf / pm)
    ratio_vals = np.array(ratio_vals)
    print(f"  mean={ratio_vals.mean():.10f}, std={ratio_vals.std():.2e}, "
          f"min={ratio_vals.min():.10f}, max={ratio_vals.max():.10f}")
