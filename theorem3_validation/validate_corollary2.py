"""Numerical check of Corollary 2 (carbon-optimal participation under time-varying grid intensity).

For each random draw: a window of W slots with fleet-weighted intensities I_t, a fixed noise
multiplier sigma, accuracy target eps_acc. Rounds are placed in the cheapest slots, so campaign
carbon is F(p) = p * G(T_acc(p)), G(n) = sum of the n smallest intensities (linear interpolation
for fractional n). Checks: (i) argmin_p F >= 2 p_min; (ii) equality for a flat profile;
(iii) the first-order condition G(T) = p |T'(p)| I_(T) holds at interior minimizers.
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq

rng = np.random.default_rng(7)

def problem(N, d, sigma, eps, G2=1.0, C2=1.0, LD=1.0):
    B = G2 + d * sigma**2 * C2
    alpha = eps + G2 / N
    pmin = B / (alpha * N)
    T = lambda p: 2 * LD / (alpha - B / (N * p))
    dT = lambda p: -2 * LD * (B / (N * p**2)) / (alpha - B / (N * p))**2   # T'(p) < 0
    return pmin, T, dT

def make_G(I):
    s = np.sort(I)
    cs = np.concatenate([[0.0], np.cumsum(s)])
    def G(n):
        n = min(max(n, 0.0), len(s))
        k = int(np.floor(n)); f = n - k
        return cs[k] + (f * s[k] if k < len(s) else 0.0)
    def marg(n):                      # I_(ceil n): marginal intensity of the last slot used
        k = min(max(int(np.ceil(n)) - 1, 0), len(s) - 1)
        return s[k]
    return G, marg

def diurnal(W, lo=250.0, hi=550.0, phase=0.0, noise=0.0):
    t = np.arange(W)
    base = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(2 * np.pi * (t - phase) / 24)
    return base if noise == 0 else base * (1 + noise * rng.standard_normal(W))

def run(n_draws=2000):
    res = []
    for _ in range(n_draws):
        N = 10**rng.uniform(3, 7); d = 10**rng.uniform(1, 5)
        sigma = 10**rng.uniform(-1, 1); eps = rng.uniform(0.1, 3.2)
        pmin, T, dT = problem(N, d, sigma, eps)
        if not (2 * pmin < 1.0): continue
        W = int(rng.choice([72, 168, 336]))
        I = diurnal(W, phase=rng.uniform(0, 24), noise=rng.uniform(0, 0.15))
        G, marg = make_G(I)
        # feasible p: Tacc(p) <= W, p in (pmin, 1]; lower end set by the deadline
        p_lo = brentq(lambda p: T(p) - W, pmin * (1 + 1e-9), 1.0) if T(1.0) <= W else None
        if p_lo is None: continue
        lo = max(p_lo, pmin * (1 + 1e-6))
        F = lambda p: p * G(T(p))
        r = minimize_scalar(F, bounds=(lo, 1.0), method='bounded', options={'xatol': 1e-12})
        pstar = r.x
        # flat profile control
        Gf, _ = make_G(np.full(W, 400.0))
        rf = minimize_scalar(lambda p: p * Gf(T(p)), bounds=(max(lo, pmin*(1+1e-6)), 1.0), method='bounded', options={'xatol': 1e-12})
        res.append(dict(N=N, W=W, pmin=pmin, lo=lo, pstar=pstar, flat=rf.x,
                        ratio=pstar / (2 * pmin), Tstar=T(pstar),
                        foc=(G(T(pstar)) - pstar * abs(dT(pstar)) * marg(T(pstar))) / max(G(T(pstar)), 1e-12),
                        interior=(pstar > lo * 1.0001 and pstar < 0.9999)))
    return res

def worked_case():
    """40 rounds in a 72-hour window under a 250-550 g CO2e/kWh diurnal profile (Sections 10.5, 11.1)."""
    I = diurnal(72); G, _ = make_G(I)
    mean_sched, mean_win = G(40) / 40, I.mean()
    q = 1 - mean_sched / mean_win
    print('scheduled mean %.1f vs window mean %.1f g CO2e/kWh (saving %.1f%%)' % (mean_sched, mean_win, 100 * q))
    try:
        import os, sys, math
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'carbon_accounting'))
        import real_accounting as ra
        kappa = math.log(1e6)
        for name, eps_p, d in (('activity recognition', 8.0, 1e4), ('healthcare', 4.0, 1e2)):
            lam = (math.sqrt(2 * (kappa + eps_p)) - math.sqrt(2 * kappa)) ** 2
            K = (2 / 0.5) * (1 + d / lam)
            cf, cc, phi = ra.fmcl_per_round(K, cellular_fraction=0.3), ra.cloud_per_round(), ra.phi_cloud()
            print('%s: T*_cloud %.1f -> %.1f rounds (FMCL schedules, cloud does not)' % (name, phi / (cf - cc), phi / (cf * (1 - q) - cc)))
    except ImportError:
        print('carbon_accounting not on path; skipping threshold check')

if __name__ == '__main__':
    worked_case()
    r = run()
    ratio = np.array([x['ratio'] for x in r]); 
    # draws whose deadline floor lies above 2 pmin are constrained by the window, not by the corollary
    free = [x for x in r if x['lo'] <= 2 * x['pmin'] * (1 - 1e-9)]
    rf = np.array([x['ratio'] for x in free])
    print('draws with valid window:', len(r), '| deadline floor below 2pmin:', len(free))
    print('min p*/(2pmin) over free draws: %.6f' % rf.min())
    print('flat-profile max |p*/(2pmin)-1| : %.2e' % max(abs(x['flat'] / (2 * x['pmin']) - 1) for x in free))
    print('share with p*>2pmin(1+1e-4): %.3f' % np.mean(rf > 1 + 1e-4))
    print('median / 90th pct / max of p*/(2pmin): %.4f %.4f %.4f' % tuple(np.percentile(rf, [50, 90, 100])))
    inter = [x for x in free if x['interior']]
    print('interior minimizers:', len(inter), '| max |FOC residual|/G: %.2e' % max(abs(x['foc']) for x in inter) if inter else 'none')
