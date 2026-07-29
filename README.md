# FMCL Simulation Code

Reproducibility code for:

> P. K. D. Pramanik, "Federated Mobile Crowd Learning:
> Convergence, Privacy, and Lifecycle Carbon Guarantees"

## What this repository contains

| File | Purpose |
|---|---|
| `simulation.py` | Core FedProx simulation engine (imported by the other scripts) |
| `floor_scaling.py` | Runs the floor-scaling experiment; generates `floor_data.npy` |
| `make_figures.py` | Generates Figure 4 and Figure 5 from the paper |

## Requirements

Python 3.10 or later. Install dependencies with:
pip install numpy matplotlib scipy

## How to reproduce the figures

Run the scripts in order:
    python floor_scaling.py
    python make_figures.py


`floor_scaling.py` takes approximately 5–10 minutes.
`make_figures.py` takes approximately 3–5 minutes.

Output files `figure4_convergence.png` and `figure5_floor_scaling.png`
are saved in the same folder.

## Correspondence

Pijush Kanti Dutta Pramanik
pijushjld@yahoo.co.in