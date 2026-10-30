"""
Recruitment floor implied by the accuracy and privacy targets (Section 10.5).

A participating device must stay within its privacy budget at its first
participation, so T_priv(sigma_max) >= 1 whenever p * T_acc < 1. With
T_priv(sigma) = lam * sigma^2 and lam = (sqrt(2(kappa + eps_p)) - sqrt(2 kappa))^2,
Lemma 1's noise ceiling gives K >= (2 / eps_acc) * (G^2 (1 - p) + d C^2 / lam),
independent of N. Theorem 4 is then evaluated at that K.
"""
import math
import real_accounting as ra

DELTA = 1e-6
KAPPA = math.log(1.0 / DELTA)
G = C = 1.0
EPS_ACC = 0.5
T_ACC = 40
CELLULAR_FRAC = 0.3

CASES = {"activity recognition": dict(eps_p=8.0, d=1e4, N=1e6),
         "healthcare":           dict(eps_p=4.0, d=1e2, N=2e5)}

if __name__ == "__main__":
    c_cloud = ra.cloud_per_round()
    for name, c in CASES.items():
        lam = (math.sqrt(2 * (KAPPA + c["eps_p"])) - math.sqrt(2 * KAPPA)) ** 2
        K = (2.0 / EPS_ACC) * (G ** 2 + c["d"] * C ** 2 / lam)
        c_f = ra.fmcl_per_round(K, cellular_fraction=CELLULAR_FRAC)
        T_star = ra.phi_cloud() / (c_f - c_cloud) if c_f > c_cloud else float("inf")
        K_cap = 500
        d_max = lam * (EPS_ACC * K_cap / 2.0 - G ** 2) / C ** 2
        print(f"{name}: lambda={lam:.4f}  K_floor={K:,.0f}  p={K / c['N']:.4f}  pT_acc={K / c['N'] * T_ACC:.2f}")
        print(f"  c_FMCL={c_f:.0f} g/round  T*_cloud={T_star:.2f} rounds (T_acc={T_ACC})  d_max at K_cap={K_cap}: {d_max:.0f}")
