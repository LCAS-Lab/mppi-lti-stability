# MPPI Closed-Loop Stability — LTI Simulation Code

Simulation code for the paper:

> **Closed-Loop Stability of Model Predictive Path Integral Control for Linear Time-Invariant Systems**  
> Hyung-Jin (Daniel) Yoon and Hunmin Kim  
> *Preprint, 2026.*  
> [arXiv link — to be added upon submission]

---

## Overview

This repository contains the simulation experiments and figure generation code for Paper P1 in a three-paper series establishing formal stability guarantees for MPPI control.

**Core result:** MPPI is exponentially stable in expectation for LTI systems when the sample count satisfies M ≥ M*, where M* is explicit and computable from the DARE solution.

---

## Contents

| File | Description |
|------|-------------|
| `mppi_stability_experiments_torch.py` | Full experiment suite (CUDA-accelerated). Runs all 5 experiments, prints console diagnostics, outputs LaTeX tables, and saves publication-quality figures. |
| `mppi_figures.py` | CPU/NumPy-only figure generation. Identical outputs; no PyTorch required. Use this if you don't have a GPU. |

---

## Experiments

| # | Name | What it shows |
|---|------|---------------|
| 1 | Bound Verification | Empirical E[‖x_k‖] vs. certificate envelope c·ρ^k·‖x₀‖ + γ√tr(Σ_w) |
| 2 | Decay Rate vs. M | Empirical decay factor ρ̂(M) as M increases; crossing of certificate bound at M ≥ M* |
| 3 | Phase Portrait | Trajectory bundles for M=50 (uncertified) vs. M=500 (certified) |
| 4 | M* Comparison | Analytical M*=153 vs. empirical M̂*=30; σ_w-independence confirmed |
| 5 | ESS Diagnostic | Normalized ESS/M ≈ 0.003 is normal for MPPI; convergence holds regardless |

---

## System: Double Integrator

```
A = [[1, 1], [0, 1]]    B = [[0], [1]]    Q = I₂    R = 0.1
```

DARE solution gives ρ(A_cl) = 0.362, certificate decay rate ρ = 0.943, M* = 153 (η = 0.05).

---

## Requirements

**GPU version** (`mppi_stability_experiments_torch.py`):
```
pip install torch numpy scipy matplotlib
```
Runs on CUDA, Apple MPS, or CPU automatically.

**CPU version** (`mppi_figures.py`):
```
pip install numpy scipy matplotlib
```

---

## Usage

```bash
# GPU-accelerated (recommended for full experiment suite)
python mppi_stability_experiments_torch.py --outdir figures/

# CPU-only (figure generation)
python mppi_figures.py
```

Figures are saved as both `.pdf` (for LaTeX/Overleaf) and `.png` to the output directory.

---

## Citation

If you use this code, please cite:

```bibtex
@article{yoon2026mppi_lti,
  title   = {Closed-Loop Stability of Model Predictive Path Integral Control
             for Linear Time-Invariant Systems},
  author  = {Yoon, Hyung-Jin and Kim, Hunmin},
  journal = {arXiv preprint},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX}
}

@misc{yoon2026mppi_lti_code,
  author       = {Yoon, Hyung-Jin and Kim, Hunmin},
  title        = {{MPPI} Stability Simulation Code --- {LTI} Systems},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/stargaze221/mppi-lti-stability}},
  note         = {Simulation code for ``Closed-Loop Stability of Model
                  Predictive Path Integral Control for Linear
                  Time-Invariant Systems''}
}
```

---

## Series

| Paper | Topic | Status |
|-------|-------|--------|
| **P1 ([mppi-lti-stability](https://github.com/stargaze221/mppi-lti-stability))** | LTI stability, Lyapunov perturbation | In preparation |
| P2 | Nonlinear stability, contraction + CLF | In preparation |
| P3 | Adaptive noise covariance estimation | In preparation |

---

## License

MIT License. See `LICENSE`.
