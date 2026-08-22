# MPPI Closed-Loop Stability — LTI Simulation Code

Simulation and numerical-audit code for the paper:

> **Finite-Sample Closed-Loop Stability of Model Predictive Path Integral Control for Linear Time-Invariant Systems**  
> Hyung-Jin Yoon and Hunmin Kim  
> *Preprint, 2026; revised manuscript in preparation.*

---

## Overview

This repository contains the simulation experiments and numerical certificate audit for the LTI/quadratic MPPI stability paper.

The current analysis treats finite-sample MPPI as a stochastic perturbation of an LQR reference. With a DARE terminal cost, the exact finite-horizon first action equals the infinite-horizon LQR action. The infinite-sample MPPI temperature bias is represented explicitly, and a finite-horizon high-probability practical stability certificate is obtained when the temperature-biased matrix condition

```text
Q_lambda = P - A_lambda^T P A_lambda > 0
```

holds and the finite-sample approximation error is sufficiently small.

The explicit theorem-derived sample count is a **sufficient worst-case high-probability bound**, not a practical minimum-sample threshold. For the benchmark studied in the revised manuscript, direct numerical evaluation shows that this sufficient count is extremely conservative relative to the sample counts that work empirically.

---

## Canonical files

| File | Role |
|------|------|
| `mppi_stability_experiments_torch.py` | **Canonical paper simulation.** CUDA/MPS/CPU implementation of Experiments 1–4 plus the nominal-sequence diagnostic. The main closed-loop simulations enforce the bounded nominal-sequence assumption with radial projection at `D_U = 9`. |
| `finite_sample_certificate_scan.py` | Numerical audit of the explicit finite-sample certificate, including the sharpened Gaussian denominator bound using `Gamma_lambda`. |
| `nominal_projection_sweep.py` | Diagnostic sweep used to select and validate `D_U = 9` and quantify how often the projection safeguard activates. |

Historical scripts and obsolete generated figures associated with the earlier `M*=153` interpretation were removed from the working tree. They remain available through Git history.

---

## Benchmark

Double integrator:

```text
A = [[1, 1],
     [0, 1]]
B = [[0],
     [1]]
Q = I_2
R = 0.1
```

Main MPPI parameters:

```text
planning horizon N = 10
temperature lambda = 1.0
sampling covariance Sigma_E = I
process-noise standard deviation = 0.1   # stochastic experiments
nominal-sequence radius D_U = 9
```

For this benchmark, the repaired analytical quantities include approximately

```text
rho(A_cl)       = 0.3616
||L_lambda||_2  = 0.1857
alpha_lambda    = 0.9860
q_lambda        = 0.8906
rho_lambda      = 0.9437
```

The value `rho_lambda` is the Lyapunov transient factor after the Young-inequality step. It is not an empirical stability threshold.

---

## Projection safeguard

The canonical simulation applies the MPPI control action without clipping. After the updated horizon sequence is shifted, the **next nominal sampling center only** is radially projected onto

```text
||U_bar||_2 <= 9.
```

This makes the simulation implementation consistent with the bounded nominal-sequence assumption used by the theorem while leaving the Gaussian perturbations unbounded.

A separate 300-trajectory projection sweep of length 200 found that the safeguard activated in approximately `0.0917%` of updates for `M=50` and did not activate for `M=200` or `M=1000` in the tested runs.

---

## Experiments

The canonical simulation generates the following analyses:

1. **Closed-loop response and nominal transient reference** — empirical mean state norm for `M = 50, 200, 1000`, together with the LQR Monte Carlo reference and the theorem transient term.
2. **Empirical Lyapunov decay vs. sample count** — noise-free diagnostic `rho_hat(M)` over a wide range of sample counts. This is diagnostic only and is not used as a theorem certificate.
3. **Phase portrait** — stochastic trajectory comparison for `M=50` and `M=500` against the LQR stationary covariance ellipse.
4. **ESS diagnostic** — effective sample size and state norm along a representative `M=500` trajectory.
5. **Nominal-sequence norm diagnostic** — deliberately unprojected warm-start statistics used to understand the projection radius.

No experiment defines an empirical `M*`, and no tested sample count is labeled certified or uncertified solely from observed decay.

---

## Numerical certificate audit

`finite_sample_certificate_scan.py` evaluates the full chain

```text
epsilon
  -> e_epsilon
  -> Delta_epsilon
  -> beta
  -> r_beta
  -> d_beta
  -> Zbar_beta
  -> C_beta
  -> M_star
```

in the log domain to avoid numerical underflow/overflow.

For the representative choice

```text
T = 200
delta_s = delta_x = 0.025
epsilon = 0.1
D_U = 9
```

the localization level is approximately `beta = 1.07e6`, and even the sharpened Gaussian denominator bound yields an astronomically conservative sufficient count (`log10(M_star) ~ 2.61e6`). This result should be interpreted as conservatism of the uniform sufficient certificate, not as the number of samples required by MPPI in practice.

---

## Requirements

```bash
pip install torch numpy scipy matplotlib
```

The simulation automatically uses CUDA when available, then Apple MPS, then CPU.

---

## Usage

Run the canonical full simulation and save figures into `figures/`:

```bash
python3 mppi_stability_experiments_torch.py --outdir figures
```

For a shorter development run:

```bash
python3 mppi_stability_experiments_torch.py --quick --outdir figures_quick
```

Audit the finite-sample certificate:

```bash
python3 finite_sample_certificate_scan.py
```

Reproduce the projection-radius sweep:

```bash
python3 nominal_projection_sweep.py
```

The simulation writes both PDF and PNG figures.

---

## Reproducibility note

The current working tree is aligned with the repaired theorem and projected warm-start implementation. Older scripts and figures that used the historical calibrated constant and `M*=153` interpretation are intentionally not part of the canonical working tree; Git history preserves those versions for provenance.

---

## License

MIT License. See `LICENSE`.
