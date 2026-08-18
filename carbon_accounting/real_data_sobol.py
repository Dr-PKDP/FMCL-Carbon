"""
Global sensitivity analysis of Theorem 4's crossover, propagating measured
and reported ranges (not assumed +/- percentages) through the model:

  - Server embodied carbon: 910 - 4,288 kg CO2e [DellR740 measured range,
    storage-removed to fully-configured]
  - Grid carbon intensity: 250 - 550 g CO2e/kWh [real regional spread]
  - Cellular participation fraction: 0 - 100% [deployment-dependent]
  - Cellular/Wi-Fi energy ratio: 33 - 104x [RadioTail, measured at realistic
    FL payload sizes, not the commonly cited but payload-mismatched 5-10x]
  - Training cadence: 1 - 24 hours between rounds [realistic FL deployment
    range, charging-window-scheduled through continuous]
  - Edge PUE: 1.5 - 2.0 [commonly cited on-premises server-room range]
  - K, devices contributing per round: 20 - 500 [realistic range for
    cross-device FL coordination, consistent with production FL practice]

Output: distribution of T*_min (rounds), and which parameter's measured or
realistic range drives that distribution most, reported as total-order
Sobol indices with 95% bootstrap confidence intervals (1,000 resamples),
matching the figures reported in the manuscript's Section 10.4.
"""
import os
import sys

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_accounting import (P_ACTIVE_W, P_IDLE_W, ROUND_SECONDS, WIFI_WH_PER_MB, PAYLOAD_MB,
                              SERVER_ACTIVE_W, SERVER_IDLE_FRACTION, CLOUD_PUE,
                              CLOUD_COMPUTE_WH_PER_ROUND, CLOUD_DATA_WH_PER_ROUND,
                              CLOUD_EMBODIED_SHARE_KG)

np.random.seed(11)

problem = {
    'num_vars': 7,
    'names': ['server_embodied_kg', 'grid_intensity', 'cellular_frac', 'cellular_ratio', 'cadence_hours', 'edge_pue', 'K_devices'],
    'bounds': [
        [910.0, 4288.0],
        [250.0, 550.0],
        [0.0, 1.0],
        [33.0, 104.0],
        [1.0, 24.0],
        [1.5, 2.0],
        [20.0, 500.0],
    ]
}


def T_min_of(server_embodied_kg, grid_intensity, cellular_frac, cellular_ratio, cadence_hours, edge_pue, K_devices, K=None):
    K = int(round(K_devices))
    marginal_train_wh = (P_ACTIVE_W - P_IDLE_W) * (ROUND_SECONDS / 3600.0)
    wifi_wh = WIFI_WH_PER_MB * PAYLOAD_MB
    comm_wh = wifi_wh * (1 - cellular_frac) + wifi_wh * cellular_ratio * cellular_frac
    c_fmcl = K * (marginal_train_wh + comm_wh) / 1000.0 * grid_intensity

    active_wh = SERVER_ACTIVE_W * (ROUND_SECONDS / 3600.0)
    idle_wh = SERVER_ACTIVE_W * SERVER_IDLE_FRACTION * cadence_hours
    c_edge = (active_wh + idle_wh) * edge_pue / 1000.0 * grid_intensity

    c_cloud = (CLOUD_COMPUTE_WH_PER_ROUND + CLOUD_DATA_WH_PER_ROUND) * CLOUD_PUE / 1000.0 * grid_intensity

    phi_edge = server_embodied_kg * 1000.0
    phi_cloud = CLOUD_EMBODIED_SHARE_KG * 1000.0

    BIG = 1e9
    T_edge = phi_edge / (c_fmcl - c_edge) if c_fmcl > c_edge else BIG
    T_cloud = phi_cloud / (c_fmcl - c_cloud) if c_fmcl > c_cloud else BIG
    return min(T_edge, T_cloud)


if __name__ == "__main__":
    param_values = saltelli.sample(problem, 2048, calc_second_order=False)
    Y = np.array([T_min_of(*row) for row in param_values])

    finite = Y[Y < 1e8]
    infinite_frac = np.mean(Y >= 1e8)

    print("Fraction of real-data parameter space where FMCL dominates at every horizon "
          f"(no finite crossover): {infinite_frac:.1%}")
    print("K, the number of devices contributing per round, is swept over 20-500,")
    print("a realistic range for cross-device FL coordination, rather than fixed at one value.")
    if len(finite) > 0:
        print(f"Among the remaining {1-infinite_frac:.1%} (finite crossover exists):")
        print(f"  median T*_min = {np.median(finite):.1f} rounds")
        print(f"  90% interval  = [{np.percentile(finite,5):.1f}, {np.percentile(finite,95):.1f}] rounds")

    print("\nTotal-order Sobol indices, with 95% bootstrap confidence intervals over")
    print("1,000 resamples, on log(1+T*_min), since the output is dominated by the")
    print("binary finite/infinite split:")
    Y_log = np.log1p(np.minimum(Y, 1e6))
    Si_log = sobol.analyze(problem, Y_log, calc_second_order=False, print_to_console=False,
                            num_resamples=1000)
    for i, name in enumerate(problem['names']):
        print(f"  {name:20s}  ST={Si_log['ST'][i]:.3f}  ST_conf={Si_log['ST_conf'][i]:.3f}  "
              f"(95% CI [{Si_log['ST'][i]-Si_log['ST_conf'][i]:.3f}, {Si_log['ST'][i]+Si_log['ST_conf'][i]:.3f}])")
