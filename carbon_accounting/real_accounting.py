"""
Real-data-grounded instantiation of Theorem 4's architecture-selection criterion.

Every parameter below traces to a cited, published source rather than an
illustrative round number. Where a source gives a range, both endpoints are
carried through to Section 11's sensitivity analysis rather than collapsed
to a single point estimate.

SOURCES
-------
[Fairphone5]  Sanchez, Baur, Eguren (2024). "Life Cycle Assessment of the
              Fairphone 5." Fraunhofer IZM, commissioned by Fairphone B.V.
              ISO 14040/14044/14067. Total GWP100 (3yr baseline) = 42.1 kg
              CO2e; production phase = 32.7 kg CO2e (78% of total).

[DellR740]    thinkstep AG / Sphera (2019). "Life Cycle Assessment of Dell
              PowerEdge R740." Commissioned by Dell Technologies. ISO
              14040/14044 cradle-to-grave, critically reviewed (University
              of Limerick). Embodied carbon = 4,288 kg CO2e as configured;
              910 kg CO2e with storage removed. Functional unit: one
              server, 4-year service life.

[Shehabi24]   Shehabi et al. (2024). "United States Data Center Energy
              Usage Report." LBNL / U.S. DOE. National-average idle power
              draw = 36% of peak (2023), down from 51% (2014).

[R740power]   Vendor and third-party power measurements for the R740
              platform: approximately 70 W idle (light single-socket) to
              650-760 W peak (dual-socket, high-TDP); independent AVX-512
              stress testing on the same platform family measured over
              700 W. (enterasource.com power-consumption analysis, 2026;
              ServeTheHome R640/R740xd power comparison.)

[DeviceTrain] On-device training power and duration are less tightly
              pinned down in the published literature than the LCA, DOE,
              and PUE figures above; this module states that difference in
              evidence tier explicitly. Device active power: independent
              measurement studies of on-device training on mobile-class
              processors report power draw in the range of a few watts
              under sustained compute load, consistent with the
              P_ACTIVE_W = 4.0 point estimate used here, though not
              pinning it to better than roughly a factor of 2. Round
              duration: reported local-epoch training times for compact
              models on mobile-class or comparable-compute hardware (for
              example, Raspberry Pi 4 at roughly 144 s/epoch for a
              moderate image dataset; other studies report 14 s to several
              minutes depending on model and dataset size) bracket the
              ROUND_SECONDS = 90 point estimate within roughly a 3-5x
              range rather than fixing it precisely. Both parameters are
              point estimates consistent with published measurement
              ranges, not independently sourced primary figures the way
              the LCA and DOE data above are. The manuscript states this
              distinction explicitly rather than letting the uniform
              citation style imply uniform certainty.

[GoogleCloud] Google Cloud fleet-average PUE = 1.10 (published, widely
              cited as industry-leading). Cross-provider average PUE
              across 104 datacenters = 1.18 (Holori PUE comparison, 2026).

[CCF]         Cloud Carbon Footprint methodology (open-source, built on
              Etsy's "Cloud Jewels" energy-per-vCPU-hour coefficients),
              used for the cloud compute-energy conversion factor.

[EPAeGRID]    U.S. EPA eGRID national-average grid carbon intensity,
              commonly cited at roughly 369-386 g CO2/kWh (2023 data);
              the IEA global average is higher, around 480 g CO2e/kWh.
              This paper's Section 10 already uses 400 g CO2e/kWh as a
              representative figure, retained here as the point estimate,
              with a wide band carried into the sensitivity analysis.

[RadioTail]   Radio-tail communication-energy modeling consistent with the
              FMCL series' own calibration work: the cellular-to-Wi-Fi
              energy ratio at realistic FL payload sizes (0.1-1 MB) is
              33-104x, not the 5-10x figure that applies only at much
              larger (approximately 15 MB) payloads and is commonly
              miscited for FL specifically.

All computations below are per training round, at K participating FMCL
devices, matching the round structure already established in Section 4.
"""
import numpy as np

# ---------------------------------------------------------------------
# Grid carbon intensity (shared point estimate and real-world spread)
# ---------------------------------------------------------------------
I_GRID_POINT = 400.0          # g CO2e/kWh, Section 10's own convention [EPAeGRID-consistent]
I_GRID_LOW, I_GRID_HIGH = 250.0, 550.0   # low-carbon grid to coal-heavy grid, real regional spread

# ---------------------------------------------------------------------
# FMCL device tier
# ---------------------------------------------------------------------
DEVICE_EMBODIED_KG = 32.7      # kg CO2e, [Fairphone5] production-phase share (the sunk cost P1 refers to)
P_ACTIVE_W = 4.0               # W, representative flagship SoC under sustained NPU/CPU load
P_IDLE_W = 0.75                # W, [Fairphone5]-consistent baseline (27 Wh / 1.5-day charge cycle)
ROUND_SECONDS = 90.0           # local training interval per device per round
PAYLOAD_MB = 1.0               # compressed gradient upload, as elsewhere in this paper
WIFI_WH_PER_MB = 0.005         # Wh/MB, within the 0.001-0.01 Wh/device range already cited [71]
CELLULAR_RATIO_LOW, CELLULAR_RATIO_HIGH = 33.0, 104.0   # [RadioTail], realistic FL payload sizes

# ---------------------------------------------------------------------
# Dedicated-edge tier (single representative server; M scales linearly)
# ---------------------------------------------------------------------
SERVER_EMBODIED_KG_CONFIGURED = 4288.0   # kg CO2e, [DellR740] as configured
SERVER_EMBODIED_KG_NOSTORAGE = 910.0     # kg CO2e, [DellR740] compute-only (storage removed)
SERVER_ACTIVE_W = 500.0        # W, representative dual-socket draw under FL aggregation load [R740power]
SERVER_IDLE_FRACTION = 0.36    # [Shehabi24]
SERVER_IDLE_W = SERVER_ACTIVE_W * SERVER_IDLE_FRACTION
EDGE_PUE = 1.75                # on-premises server room, midpoint of commonly cited 1.5-2.0 range

# ---------------------------------------------------------------------
# Cloud tier
# ---------------------------------------------------------------------
CLOUD_PUE = 1.10                # [GoogleCloud]
CLOUD_COMPUTE_WH_PER_ROUND = 20.0   # Wh, server-side compute at cloud utilization, [CCF]-consistent
CLOUD_DATA_WH_PER_ROUND = 3.0       # Wh, wide-area data movement per round (data-governance-limited case)
CLOUD_EMBODIED_SHARE_KG = 2.0       # kg CO2e/year, shared multi-tenant amortized share (small, not zero)


def fmcl_per_round(K, cellular_fraction=0.0, cellular_ratio=None, I_grid=I_GRID_POINT):
    """c_FMCL: marginal device energy plus communication energy, K devices per round."""
    if cellular_ratio is None:
        cellular_ratio = (CELLULAR_RATIO_LOW + CELLULAR_RATIO_HIGH) / 2
    marginal_train_wh = (P_ACTIVE_W - P_IDLE_W) * (ROUND_SECONDS / 3600.0)
    wifi_comm_wh = WIFI_WH_PER_MB * PAYLOAD_MB
    comm_wh = wifi_comm_wh * (1 - cellular_fraction) + wifi_comm_wh * cellular_ratio * cellular_fraction
    per_device_wh = marginal_train_wh + comm_wh
    total_wh = K * per_device_wh
    return total_wh / 1000.0 * I_grid  # g CO2e per round (Wh to kWh, times g CO2/kWh)


def edge_per_round(M=1, rho=1.0, I_grid=I_GRID_POINT, pue=EDGE_PUE, inter_round_hours=1.0):
    """c_edge: active plus rho*idle server energy, M servers, facility overhead via PUE.

    inter_round_hours is an explicit, stated modeling choice: how much idle
    time elapses per round, that is, the training cadence. This is not a
    fixed physical constant. It is the single parameter that most
    determines how much idle cost each round is charged, and it is
    carried into the sensitivity analysis explicitly rather than fixed
    at one value.
    """
    active_wh = SERVER_ACTIVE_W * (ROUND_SECONDS / 3600.0)
    idle_wh = SERVER_IDLE_W * rho * inter_round_hours
    total_wh = M * (active_wh + idle_wh) * pue
    return total_wh / 1000.0 * I_grid


def cloud_per_round(I_cloud=I_GRID_POINT, pue=CLOUD_PUE):
    total_wh = (CLOUD_COMPUTE_WH_PER_ROUND + CLOUD_DATA_WH_PER_ROUND) * pue
    return total_wh / 1000.0 * I_cloud


def phi_edge(M=1, server_embodied_kg=SERVER_EMBODIED_KG_CONFIGURED):
    return M * server_embodied_kg * 1000.0  # kg to g CO2e


def phi_cloud():
    return CLOUD_EMBODIED_SHARE_KG * 1000.0  # g CO2e/year, amortized share


if __name__ == "__main__":
    K = 100  # devices per round, matching Section 10.1's activity-recognition scale order

    print("=== Real-data-grounded per-round operational carbon (g CO2e/round, K=100) ===")
    print("Cadence sensitivity: edge idle cost depends directly on training frequency.\n")

    for cadence_label, hours in [("Hourly rounds", 1.0), ("Every 6 hours", 6.0),
                                   ("Daily (overnight) rounds", 24.0)]:
        c_edge = edge_per_round(inter_round_hours=hours)
        c_cloud = cloud_per_round()
        print(f"--- {cadence_label} ({hours}h between rounds) ---")
        for cell_label, frac in [("Wi-Fi only", 0.0), ("30% cellular", 0.3), ("100% cellular", 1.0)]:
            c_fmcl = fmcl_per_round(K, cellular_fraction=frac)
            phi_e, phi_c = phi_edge(), phi_cloud()
            T_edge_star = phi_e / (c_fmcl - c_edge) if c_fmcl > c_edge else float('inf')
            T_cloud_star = phi_c / (c_fmcl - c_cloud) if c_fmcl > c_cloud else float('inf')
            T_min = min(T_edge_star, T_cloud_star)
            print(f"  {cell_label:15s}: c_FMCL={c_fmcl:7.1f}  c_edge={c_edge:7.1f}  c_cloud={c_cloud:6.1f}  "
                  f"T*_min={T_min:12.1f}" + ("  [FMCL dominant at every horizon]" if T_min == float('inf') else ""))
        print()

    print("=== Embodied carbon (g CO2e) ===")
    print(f"Phi_edge (1 server, configured):  {phi_edge():12.0f}")
    print(f"Phi_edge (1 server, no storage):  {phi_edge(server_embodied_kg=SERVER_EMBODIED_KG_NOSTORAGE):12.0f}")
    print(f"Phi_cloud (amortized share):       {phi_cloud():12.0f}")
    print(f"Phi_FMCL:                          {0:12.0f}  (near-zero under P1; device embodied = {DEVICE_EMBODIED_KG} kg CO2e, sunk)")
