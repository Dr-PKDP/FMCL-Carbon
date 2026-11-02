"""Fig. 2: total lifecycle carbon versus training horizon for the illustrative three-architecture instance.

C_a(T) = Phi_a + c_a * T (Eq. 13), with
    FMCL : Phi = 0,        c = 500 g CO2e/round
    edge : Phi = 2.5e6 g,  c = 200 g CO2e/round
    cloud: Phi = 1.2e6 g,  c = 320 g CO2e/round
Output: fig2_crossover.svg (black text, legend outside the axes).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCH = {"FMCL": (0.0, 500.0), "Dedicated edge": (2.5e6, 200.0), "Cloud": (1.2e6, 320.0)}
T = [0, 20000]

def cross(a, b):
    (pa, ca), (pb, cb) = ARCH[a], ARCH[b]
    return (pb - pa) / (ca - cb)

t_cloud = cross("FMCL", "Cloud")            # 6,666.7: FMCL falls behind cloud (T*_min)
t_edge = cross("FMCL", "Dedicated edge")    # 8,333.3
t_ce = (ARCH["Dedicated edge"][0] - ARCH["Cloud"][0]) / (ARCH["Cloud"][1] - ARCH["Dedicated edge"][1])  # 10,833.3

plt.rcParams.update({"font.size": 9, "text.color": "black", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black", "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(9.0, 2.7))
for name, (phi, c), ls in zip(ARCH, ARCH.values(), ("-", "--", ":")):
    ax.plot(T, [phi + c * t for t in T], ls, color="black", lw=1.8, label=name)
marks = ((t_cloud, "#d62728", f"FMCL/cloud crossover, T* = {t_cloud:,.0f}"),
         (t_edge, "#1f77b4", f"FMCL/edge crossover, T = {t_edge:,.0f}"),
         (t_ce, "#2ca02c", f"cloud/edge crossover, T = {t_ce:,.0f}"))
bands = ((0, t_cloud, "#d62728", "FMCL optimal"), (t_cloud, t_ce, "#2ca02c", "Cloud optimal"),
         (t_ce, 20000, "#ff7f0e", "Edge optimal"))
for lo, hi, col, lab in bands:     # lowest-carbon architecture on each interval (lower envelope)
    ax.axvspan(lo, hi, color=col, alpha=0.07, lw=0)
    ax.text((lo + hi) / 2, 1.12e7, lab, ha="center", va="center", fontsize=9, color="black")
for x, col, lab in marks:
    ax.axvline(x, color=col, lw=2.0, ls=(0, (6, 3)), label=lab)
ax.set_xlim(0, 20000); ax.set_ylim(0, 1.2e7)
ax.set_xlabel("Training horizon T (rounds)"); ax.set_ylabel("Total lifecycle carbon (g CO$_2$e)")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=2.6)
fig.tight_layout()
fig.savefig("fig2_crossover.svg", bbox_inches="tight")
print(f"FMCL/cloud {t_cloud:.1f}  FMCL/edge {t_edge:.1f}  cloud/edge {t_ce:.1f}")
