"""Robustness of the Section 10.4 Sobol indices to the treatment of configurations without a finite crossover.

Re-uses the response T_min_of from real_data_sobol.py and varies (i) the clipping threshold of
log(1 + min(T, clip)), (ii) the response itself (indicator of a finite crossover), and reports
(iii) Spearman correlations among finite crossovers.
"""
import numpy as np
from scipy.stats import spearmanr
from SALib.sample import saltelli
from SALib.analyze import sobol
from real_data_sobol import problem, T_min_of

np.random.seed(11)
X = saltelli.sample(problem, 2048, calc_second_order=False)
Y = np.array([T_min_of(*row) for row in X])
names = problem['names']

def indices(resp):
    Si = sobol.analyze(problem, resp, calc_second_order=False, print_to_console=False, num_resamples=200)
    return dict(zip(names, Si['ST']))

print('Total-order index of K_devices vs clipping threshold:')
for clip in (1e4, 1e5, 1e6, 1e7, 1e8):
    print('  clip %.0e: ST(K) = %.3f' % (clip, indices(np.log1p(np.minimum(Y, clip)))['K_devices']))
print('Indicator of a finite crossover: ST(K) = %.3f' % indices((Y < 1e8).astype(float))['K_devices'])
fin = Y < 1e8
print('Finite crossovers: %d of %d' % (fin.sum(), len(Y)))
for j, n in enumerate(names):
    rho = spearmanr(X[fin, j], Y[fin]).correlation
    print('  Spearman rho(%s, T*) = %.2f' % (n, rho))
