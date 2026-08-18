# Reuse-Centric Federated Learning: supplementary code

Code accompanying "Reuse-Centric Federated Learning: A Lifecycle-Carbon
Criterion for Distributed Architecture Selection." This repository
contains the scripts used to produce the numerical results and figures
in Sections 10 and 11 of the paper: the real-hardware carbon accounting,
the Pareto-frontier construction, the sensitivity analyses, the
closed-form validation of Theorem 3, and the Monte Carlo campaign
simulation.

Each script is runnable on its own and prints the same figures reported
in the manuscript. None of them require a GPU or take more than a few
minutes to run on a laptop.

## Repository structure

```
carbon_accounting/          Section 10.4, Fig. 3
  real_accounting.py           source module: per-round carbon for FMCL,
                                dedicated edge, and cloud, from cited
                                hardware and grid data
  real_data_sobol.py            Sobol sensitivity over the measured
                                parameter ranges, with K swept as an
                                uncertain quantity
  make_figure.py                regenerates Fig. 3

pareto_frontier/            Section 11.2, Fig. 4
  pareto_frontier.py             numerical Pareto frontier at fixed
                                accuracy, and the weighted-sum recovery
                                test behind Proposition 3
  make_figure.py                regenerates Fig. 4

instantiation_sensitivity/  Section 11.3, Fig. 5, Table 7
  sensitivity.py                 Sobol sensitivity of the architecture-
                                selection crossover for both worked
                                instantiations of Table 4
  make_figure.py                regenerates Fig. 5

theorem3_validation/        Section 11.1
  validate_theorem3.py           numerical validation of the closed-form
                                carbon-optimal participation rate against
                                a direct optimizer, across 2,000 randomized
                                parameter regimes

monte_carlo/                Section 11.4, Fig. 6
  campaign.py                    stochastic simulation of a full
                                deployment campaign: device availability,
                                device-level power heterogeneity, and
                                communication mode all vary round by round
  make_figure.py                regenerates Fig. 6
```

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.12, numpy, scipy, matplotlib, and SALib. No other
dependencies.

## Running the analyses

Each subdirectory is self-contained; run its main script directly:

```
cd carbon_accounting
python real_accounting.py       # per-round carbon at several cadences
python real_data_sobol.py       # Sobol sensitivity, with bootstrap CIs
python make_figure.py           # writes fig3_real_data.png
```

```
cd pareto_frontier
python pareto_frontier.py       # frontier size and weighted-sum recovery
python make_figure.py           # writes fig4_pareto_frontier.png
```

```
cd instantiation_sensitivity
python sensitivity.py           # both worked instantiations
python make_figure.py           # writes fig5_sobol_sensitivity.png
```

```
cd theorem3_validation
python validate_theorem3.py     # closed-form vs. numerical optimizer
```

```
cd monte_carlo
python campaign.py              # writes mc_results.npz
python make_figure.py           # writes fig6_monte_carlo.png, reads mc_results.npz
```

`monte_carlo/make_figure.py` and `instantiation_sensitivity/make_figure.py`
import from `carbon_accounting/real_accounting.py`; run scripts from
within their own directory as shown above, and the imports resolve
automatically relative to each script's own location.

## What each script reproduces

| Script | Reproduces |
|---|---|
| `carbon_accounting/real_accounting.py` | Per-round operational carbon for FMCL, dedicated edge, and cloud at several cadences and connectivity mixes |
| `carbon_accounting/real_data_sobol.py` | 21.4% of the real-data parameter space has no finite crossover; where one exists, median 120 rounds, 90% interval [33, 1,487]; K dominates the variance (S_T = 0.84, 95% CI [0.79, 0.89]) |
| `pareto_frontier/pareto_frontier.py` | 85-point numerically constructed Pareto frontier; weighted-sum scalarization reaches 42 of them, missing 51% at every weight |
| `instantiation_sensitivity/sensitivity.py` | Sensitivity of both worked instantiations' crossover horizon; 90% intervals [1,545, 9,154] and [5,987, 272,901] rounds |
| `theorem3_validation/validate_theorem3.py` | Closed-form participation rate matches a direct numerical optimizer to a maximum relative error of 1.04e-7 across 1,523 valid trials |
| `monte_carlo/campaign.py` | Empirical crossover-vs-cloud median of 164 rounds (90% CI [114, 295]) across 300 simulated trials, against a closed-form prediction of 161.6 rounds |

## Reproducibility notes

Every script fixes its own random seed, so re-running any of them
reproduces the exact numbers above. `real_data_sobol.py` and
`sensitivity.py` use Saltelli sampling with 2,048 base samples; the
former additionally bootstraps 1,000 resamples to report confidence
intervals on the Sobol indices, matching what the manuscript reports in
Section 10.4.

The carbon accounting in `real_accounting.py` traces every constant to a
cited, published source; full citations are in the module's own
docstring. Where a cited source gives a range rather than a point
estimate, both endpoints are carried into the relevant sensitivity
analysis rather than collapsed to a single figure.

## License

MIT. See `LICENSE`.
