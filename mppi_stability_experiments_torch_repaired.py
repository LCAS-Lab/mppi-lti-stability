"""
MPPI Closed-Loop Stability Experiments — P1 (LTI/LQR)
Repaired-theorem simulation audit
=====================================================

This version is aligned with the repaired manuscript structure:
  * the infinite-sample temperature bias is computed explicitly through
    Sigma_*, L_lambda, G_lambda, A_lambda, Q_lambda, q_lambda, rho_lambda;
  * Experiment 2 uses the true Lyapunov function V(x)=x^T P x;
  * empirical decay-rate plots are diagnostics, not theorem certificates;
  * the old calibrated M* formula and certified/uncertified labels are removed;
  * the low-M finer sweep is retained only as an empirical threshold diagnostic.

The new finite-sample numerical certificate
    D_U, epsilon, beta, Zbar_beta, C_beta, M_star
is intentionally NOT computed here yet. It should be added only after the
repaired finite-sample subsection is numerically instantiated.

Usage:
    python mppi_stability_experiments_torch_repaired.py
    python mppi_stability_experiments_torch_repaired.py --outdir figures_repaired
    python mppi_stability_experiments_torch_repaired.py --quick
"""

import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_discrete_are

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="MPPI stability experiments aligned with the repaired P1 theorem"
)
parser.add_argument(
    "--outdir", type=str, default="figures_repaired",
    help="Directory to save figures (default: ./figures_repaired)",
)
parser.add_argument(
    "--quick", action="store_true",
    help="Reduced Monte Carlo counts for a fast local smoke test",
)
args = parser.parse_args()
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def tt(x, dtype=torch.float64):
    return torch.tensor(x, dtype=dtype, device=device)


def tn(x):
    return x.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# -----------------------------------------------------------------------------
# System and MPPI parameters
# -----------------------------------------------------------------------------
A_np = np.array([[1.0, 1.0], [0.0, 1.0]])
B_np = np.array([[0.0], [1.0]])
Q_np = np.eye(2)
R_np = np.array([[0.1]])
n, m = 2, 1
N_hor = 10
lam = 1.0
sig_eps = 1.0

# DARE / LQR
P_np = solve_discrete_are(A_np, B_np, Q_np, R_np)
K_np = np.linalg.solve(
    R_np + B_np.T @ P_np @ B_np,
    B_np.T @ P_np @ A_np,
)
Acl_np = A_np - B_np @ K_np
P_eigs = np.linalg.eigvalsh(P_np)
lmin_P = P_eigs.min()
lmax_P = P_eigs.max()
Acl_eigs = np.linalg.eigvals(Acl_np)
rho_LQR = float(max(abs(Acl_eigs)))

dare_residual = np.max(np.abs(
    P_np - (
        Q_np + A_np.T @ P_np @ A_np
        - A_np.T @ P_np @ B_np
        @ np.linalg.solve(
            R_np + B_np.T @ P_np @ B_np,
            B_np.T @ P_np @ A_np,
        )
    )
))

# -----------------------------------------------------------------------------
# Stacked quadratic cost J(x,U)=U^T H U + 2 x^T F U + x^T G x
# rhs_mat_np below is F^T.
# -----------------------------------------------------------------------------
Su_np = np.zeros((n * N_hor, m * N_hor))
Sx_np = np.zeros((n * N_hor, n))
Qbar_np = np.zeros((n * N_hor, n * N_hor))
for i in range(N_hor):
    Sx_np[i*n:(i+1)*n, :] = np.linalg.matrix_power(A_np, i + 1)
    for j in range(i + 1):
        Su_np[i*n:(i+1)*n, j*m:(j+1)*m] = (
            np.linalg.matrix_power(A_np, i - j) @ B_np
        )
    Qbar_np[i*n:(i+1)*n, i*n:(i+1)*n] = (
        Q_np if i < N_hor - 1 else P_np
    )

H_np = Su_np.T @ Qbar_np @ Su_np + np.kron(np.eye(N_hor), R_np)
Ft_np = Su_np.T @ Qbar_np @ Sx_np
lmin_H = np.linalg.eigvalsh(H_np).min()

# -----------------------------------------------------------------------------
# Repaired theorem: exact infinite-sample temperature bias
# Sigma_* = (Sigma_E^{-1} + 2H/lambda)^{-1}
# L_lambda = S0 - (2/lambda) S0 Sigma_* H
# G_lambda = L_lambda H^{-1} F^T
# A_lambda = A_cl + B G_lambda
# Q_lambda = P - A_lambda^T P A_lambda
# q_lambda = 1 - lambda_min(Q_lambda)/lambda_max(P)
# rho_lambda = sqrt(q_lambda)
# -----------------------------------------------------------------------------
mN = m * N_hor
Sigma_E_np = (sig_eps ** 2) * np.eye(mN)
Sigma_star_np = np.linalg.inv(
    np.linalg.inv(Sigma_E_np) + (2.0 / lam) * H_np
)
S0_np = np.zeros((m, mN))
S0_np[:, :m] = np.eye(m)
L_lambda_np = S0_np - (2.0 / lam) * S0_np @ Sigma_star_np @ H_np
G_lambda_np = L_lambda_np @ np.linalg.solve(H_np, Ft_np)
A_lambda_np = Acl_np + B_np @ G_lambda_np
Q_lambda_np = P_np - A_lambda_np.T @ P_np @ A_lambda_np
Q_lambda_eigs = np.linalg.eigvalsh(Q_lambda_np)
q_lambda_pd = bool(Q_lambda_eigs.min() > 0.0)
alpha_lambda = float(Q_lambda_eigs.min() / lmax_P)
q_lambda = float(1.0 - alpha_lambda)
rho_lambda = float(np.sqrt(q_lambda)) if 0.0 <= q_lambda < 1.0 else np.nan

# Move to device
A = tt(A_np)
B = tt(B_np)
P = tt(P_np)
H = tt(H_np)
Ft = tt(Ft_np)

# -----------------------------------------------------------------------------
# Console helpers
# -----------------------------------------------------------------------------
W = 76


def hdr(s):
    print(f"\n{'='*W}\n  {s}\n{'='*W}")


def sec(s):
    print(f"\n{'-'*W}\n  {s}\n{'-'*W}")


def row(label, val, note=""):
    note_str = f"   [{note}]" if note else ""
    print(f"  {label:<44s} {val}{note_str}")


hdr("MPPI STABILITY EXPERIMENTS — REPAIRED-THEOREM AUDIT")
print(f"  Output dir : {os.path.abspath(OUTDIR)}")
print(
    f"  Device     : {device}"
    + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
)
print(f"  Mode       : {'QUICK smoke test' if args.quick else 'FULL rerun'}")

sec("System / DARE")
row("A", "[[1, 1], [0, 1]]  (double integrator)")
row("B", "[[0], [1]]")
row("Q", "I_2")
row("R", "0.1")
row("Horizon N", N_hor)
row("Temperature lambda", lam)
row("Sampling sigma_epsilon", sig_eps)
row("P eigenvalues", f"[{P_eigs[0]:.6f}, {P_eigs[1]:.6f}]")
row("K", f"[{K_np[0,0]:.6f}, {K_np[0,1]:.6f}]")
row("rho(A_cl)", f"{rho_LQR:.6f}")
row("DARE residual", f"{dare_residual:.3e}")
row("lambda_min(H)", f"{lmin_H:.6f}")

sec("Repaired infinite-sample bias / deterministic reference")
row("||L_lambda||_2", f"{np.linalg.norm(L_lambda_np, 2):.6f}")
row("G_lambda", np.array2string(G_lambda_np, precision=6))
row("eig(A_lambda)", np.array2string(np.linalg.eigvals(A_lambda_np), precision=6))
row("eig(Q_lambda)", np.array2string(Q_lambda_eigs, precision=6))
row("Q_lambda positive definite?", "YES" if q_lambda_pd else "NO")
row("alpha_lambda", f"{alpha_lambda:.6f}")
row("q_lambda", f"{q_lambda:.6f}")
row("rho_lambda = sqrt(q_lambda)", f"{rho_lambda:.6f}")
row("rho_LQR", f"{rho_LQR:.6f}")
print()
print("  IMPORTANT: rho_lambda is the deterministic biased-reference contraction")
print("  quantity. Finite-sample MPPI has an additive approximation term, so")
print("  empirical rho_hat(M) below is a diagnostic, not a theorem certificate.")
print("  The new finite-sample M_star is intentionally not evaluated in this script.")

# -----------------------------------------------------------------------------
# Core MPPI functions
# -----------------------------------------------------------------------------
def mppi_step(x, U_nom, M, gen):
    eps = (
        torch.randn(M, mN, dtype=torch.float64, device=device, generator=gen)
        * sig_eps
    )
    U_s = U_nom.unsqueeze(0) + eps
    rhs = Ft @ x
    HU = U_s @ H
    costs = (U_s * HU).sum(1) + 2.0 * (U_s @ rhs)

    # Standard numerically stable MPPI normalization.
    beta = costs.min()
    weights = torch.exp(-(costs - beta) / lam)
    weights = weights / weights.sum()

    delta = (weights.unsqueeze(1) * eps).sum(0)
    U_new = U_nom + delta
    u = U_new[:m]

    U_shifted = torch.zeros(mN, dtype=torch.float64, device=device)
    U_shifted[:(N_hor - 1)*m] = U_new[m:]
    ess = 1.0 / (weights**2).sum()
    return u, U_shifted, ess


def simulate_states(x0_np, M, sigma_w, T, gen):
    """Return the full state trajectory and ESS sequence."""
    x = tt(x0_np)
    states = [tn(x).copy()]
    ess_l = [float(M)]
    U_nom = torch.zeros(mN, dtype=torch.float64, device=device)

    for _ in range(T):
        u, U_nom, ess = mppi_step(x, U_nom, M, gen)
        if sigma_w > 0:
            noise = sigma_w * torch.randn(
                n, dtype=torch.float64, device=device, generator=gen
            )
        else:
            noise = torch.zeros(n, dtype=torch.float64, device=device)
        x = A @ x + B @ u + noise
        states.append(tn(x).copy())
        ess_l.append(float(ess))

    return np.asarray(states), np.asarray(ess_l)


def simulate_norms(x0_np, M, sigma_w, T, gen):
    states, ess = simulate_states(x0_np, M, sigma_w, T, gen)
    return np.linalg.norm(states, axis=1), ess


def simulate_lqr_norms(x0_np, sigma_w, T, gen):
    x = tt(x0_np)
    norms = [float(torch.linalg.norm(x))]
    K = tt(K_np)
    for _ in range(T):
        u = -(K @ x)
        if sigma_w > 0:
            noise = sigma_w * torch.randn(
                n, dtype=torch.float64, device=device, generator=gen
            )
        else:
            noise = torch.zeros(n, dtype=torch.float64, device=device)
        x = A @ x + B @ u + noise
        norms.append(float(torch.linalg.norm(x)))
    return np.asarray(norms)


def mc_mean_norm(x0_np, M, sigma_w, T, n_mc, seed=99):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = np.zeros((n_mc, T + 1))
    for i in range(n_mc):
        out[i], _ = simulate_norms(x0_np, M, sigma_w, T, gen)
    return out.mean(0), out.std(0)


def mc_mean_norm_lqr(x0_np, sigma_w, T, n_mc, seed=199):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = np.zeros((n_mc, T + 1))
    for i in range(n_mc):
        out[i] = simulate_lqr_norms(x0_np, sigma_w, T, gen)
    return out.mean(0), out.std(0)


def lyapunov_values(states):
    """V_k = x_k^T P x_k for a state trajectory of shape (T+1,n)."""
    return np.einsum("bi,ij,bj->b", states, P_np, states)


def estimate_decay_rate(M, n_traj=300, T_decay=20, seed=1):
    """
    Estimate the noise-free empirical Lyapunov decay rate.

    For each trajectory, compute q_k = V_{k+1}/V_k using the true
    Lyapunov function V(x)=x^T P x.  The trajectory statistic is the median
    over steps k=2,...,9 when available.  Aggregate those trajectory medians
    with another median, then report rho_hat=sqrt(q_hat).

    This is an empirical diagnostic only.  It is not the finite-sample
    stability certificate from the repaired theorem.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    rng_local = np.random.default_rng(seed)
    traj_q = []

    for _ in range(n_traj):
        xi = rng_local.standard_normal(n) * 3.0
        if np.linalg.norm(xi) < 0.1:
            continue

        states, _ = simulate_states(xi, M, 0.0, T_decay, gen)
        V = lyapunov_values(states)
        valid = V[:-1] > 1e-10
        q_inst = V[1:][valid] / V[:-1][valid]

        if q_inst.size <= 3:
            continue
        lo = 2
        hi = min(10, q_inst.size)
        if hi <= lo:
            continue
        traj_q.append(float(np.median(q_inst[lo:hi])))

    if not traj_q:
        return np.nan, np.nan

    q_hat = float(np.median(traj_q))
    rho_hat = float(np.sqrt(max(q_hat, 0.0)))
    return q_hat, rho_hat


EMPIRICAL_RHO_THRESHOLD = 0.99
FINE_M_SCAN = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]


def empirical_threshold(seed=3, n_traj=300):
    """Finer noise-free sweep; not the theorem's finite-sample M_star."""
    rows = []
    first = None
    for M in FINE_M_SCAN:
        qh, rh = estimate_decay_rate(M, n_traj=n_traj, T_decay=20, seed=seed)
        rows.append((M, qh, rh))
        if first is None and np.isfinite(rh) and rh < EMPIRICAL_RHO_THRESHOLD:
            first = (M, qh, rh)
    return first, rows


# -----------------------------------------------------------------------------
# Experiment setup
# -----------------------------------------------------------------------------
x0 = np.array([5.0, 5.0])
T_sim = 80 if args.quick else 200
sw0 = 0.10
N_MC = 40 if args.quick else 300
N_TRAJ = 50 if args.quick else 300
M_SCAN = [10, 20, 50, 100, 200, 500] if args.quick else [
    10, 20, 50, 100, 200, 500, 1000, 5000, 10000
]
t_arr = np.arange(T_sim + 1)

# -----------------------------------------------------------------------------
# EXP 1: Closed-loop trajectory overview
# -----------------------------------------------------------------------------
sec(f"Experiment 1: Closed-loop overview (n_mc={N_MC}, sigma_w={sw0:.2f})")
e1_mean, e1_std = {}, {}
print(f"  {'M':>7}  {'E||x_T||':>12}  {'Std||x_T||':>12}  {'time':>8}")
print(f"  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*8}")
for M in [50, 200, 1000]:
    t0 = time.perf_counter()
    mn, sd = mc_mean_norm(x0, M, sw0, T_sim, N_MC, seed=99)
    e1_mean[M] = mn
    e1_std[M] = sd
    print(
        f"  {M:>7d}  {mn[-1]:>12.5f}  {sd[-1]:>12.5f}  "
        f"{time.perf_counter()-t0:>7.2f}s"
    )

lqr_mean, lqr_std = mc_mean_norm_lqr(x0, sw0, T_sim, N_MC, seed=199)
print(f"  {'LQR':>7}  {lqr_mean[-1]:>12.5f}  {lqr_std[-1]:>12.5f}")
print("  NOTE: This experiment is descriptive. It is not a theorem-bound check.")

# -----------------------------------------------------------------------------
# EXP 2: True Lyapunov decay rate vs M
# -----------------------------------------------------------------------------
sec(f"Experiment 2: True Lyapunov decay diagnostic (n_traj={N_TRAJ}, noise-free)")
print(f"  {'M':>7}  {'q_hat':>10}  {'rho_hat':>10}  {'vs rho_lambda':>15}  {'time':>8}")
print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*15}  {'-'*8}")
q_hats, rho_hats = [], []
for M in M_SCAN:
    t0 = time.perf_counter()
    qh, rh = estimate_decay_rate(M, n_traj=N_TRAJ, T_decay=20, seed=1)
    q_hats.append(qh)
    rho_hats.append(rh)
    rel = "below" if np.isfinite(rh) and rh <= rho_lambda else "above"
    print(
        f"  {M:>7d}  {qh:>10.5f}  {rh:>10.5f}  "
        f"{rel:>15s}  {time.perf_counter()-t0:>7.2f}s"
    )
print("  rho_hat = sqrt(median trajectory-wise median(V_{k+1}/V_k))).")
print("  Comparison with rho_lambda is diagnostic only because finite-sample")
print("  approximation contributes an additive practical-stability term.")

# -----------------------------------------------------------------------------
# EXP 3: Phase portrait
# -----------------------------------------------------------------------------
sec(f"Experiment 3: Phase portrait ({'12' if args.quick else '30'} trajectories, sigma_w={sw0:.2f})")
n_pp = 12 if args.quick else 30
T_pp = 25 if args.quick else 35
rng_pp = np.random.default_rng(2)
inits = rng_pp.standard_normal((n_pp, n)) * 2.5 + np.array([2.5, 2.5])

def phase_trajectories(M, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for xi in inits:
        states, _ = simulate_states(xi, M, sw0, T_pp, gen)
        out.append(states)
    return np.asarray(out)

txy50 = phase_trajectories(50, 2)
txy500 = phase_trajectories(500, 3)
for M, tr in [(50, txy50), (500, txy500)]:
    final_norm = np.linalg.norm(tr[:, -1, :], axis=1)
    print(f"  M={M:>4d}: mean ||x_T||={final_norm.mean():.5f}, max={final_norm.max():.5f}")
print("  NOTE: no certified/uncertified label is used before the new M_star is computed.")

# -----------------------------------------------------------------------------
# EXP 4: Finer empirical low-M threshold scan
# -----------------------------------------------------------------------------
sec(f"Experiment 4: Finer empirical threshold scan (rho_hat < {EMPIRICAL_RHO_THRESHOLD:.2f})")
fine_n_traj = 60 if args.quick else 300
first_emp, fine_rows = empirical_threshold(seed=3, n_traj=fine_n_traj)
print(f"  {'M':>7}  {'q_hat':>10}  {'rho_hat':>10}  {'threshold?':>12}")
print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*12}")
for M, qh, rh in fine_rows:
    flag = "YES" if rh < EMPIRICAL_RHO_THRESHOLD else "NO"
    print(f"  {M:>7d}  {qh:>10.5f}  {rh:>10.5f}  {flag:>12s}")
if first_emp is not None:
    empirical_M, empirical_q, empirical_rho = first_emp
    print(
        f"  First grid point with rho_hat < {EMPIRICAL_RHO_THRESHOLD:.2f}: "
        f"M={empirical_M} (rho_hat={empirical_rho:.5f})"
    )
else:
    empirical_M = None
    print("  No grid point met the empirical threshold.")
print("  This is the separate finer sweep that includes M=30 in its grid.")
print("  The corrected V=x^T P x metric may shift the legacy M=30 threshold.")
print("  It is NOT the theorem's finite-sample M_star and has no sigma_w claim.")

# -----------------------------------------------------------------------------
# EXP 5: ESS diagnostic without certificate interpretation
# -----------------------------------------------------------------------------
sec(f"Experiment 5: ESS diagnostic (M=500, sigma_w={sw0:.2f})")
gen5 = torch.Generator(device=device).manual_seed(5)
norm_e5, ess_raw = simulate_norms(x0, 500, sw0, T_sim, gen5)
ess_e5 = ess_raw / 500.0
print(f"  Mean ESS/M (k>=1): {ess_e5[1:].mean():.6f}")
print(f"  Min  ESS/M (k>=1): {ess_e5[1:].min():.6f}")
print(f"  Max  ESS/M (k>=1): {ess_e5[1:].max():.6f}")
print(f"  Final ||x_T||     : {norm_e5[-1]:.6f}")
print("  ESS is reported only as a sampling-weight diagnostic; it is not used")
print("  as a surrogate for the repaired finite-sample stability certificate.")

# -----------------------------------------------------------------------------
# Consistency checks
# -----------------------------------------------------------------------------
sec("Consistency checks")
checks = []
checks.append(("DARE residual < 1e-10", dare_residual < 1e-10, dare_residual))
checks.append(("Q_lambda positive definite", q_lambda_pd, Q_lambda_eigs.min()))
checks.append(("0 < q_lambda < 1", 0.0 < q_lambda < 1.0, q_lambda))
checks.append(("rho_lambda^2 = q_lambda", abs(rho_lambda**2 - q_lambda) < 1e-12, rho_lambda**2 - q_lambda))
checks.append(("Experiment 2 finite outputs", np.all(np.isfinite(rho_hats)), np.nanmax(rho_hats)))
for name, ok, detail in checks:
    print(f"  {name:<42s} {'PASS' if ok else 'FAIL':>6s}   {detail}")

print()
print("  Pending numerical certificate (not silently substituted):")
print("    D_U, epsilon, beta, Zbar_beta, C_beta, M_star")

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
sec("Generating figures")

# Figure 1: closed-loop overview
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for M in [50, 200, 1000]:
    mn, sd = e1_mean[M], e1_std[M]
    ax.fill_between(t_arr, np.maximum(mn - sd, 0), mn + sd, alpha=0.10)
    ax.plot(t_arr, mn, lw=1.6, label=rf"MPPI $M={M}$")
ax.plot(t_arr, lqr_mean, "k--", lw=1.8, label="LQR Monte Carlo reference")
ax.set_xlabel("Time step $k$")
ax.set_ylabel(r"$\mathbb{E}[\|x_k\|_2]$")
ax.set_title(
    "Experiment 1: Closed-Loop Monte Carlo Overview\n"
    rf"($\sigma_w={sw0:.2f}$, $n_{{\rm mc}}={N_MC}$, $x_0=(5,5)^\top$)"
)
ax.legend(loc="upper right")
ax.set_xlim(0, T_sim)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig1_closed_loop_overview.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "fig1_closed_loop_overview.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Fig 1 saved")

# Figure 2: Lyapunov decay diagnostic
fig, ax = plt.subplots(figsize=(5.9, 4.2))
ax.semilogx(M_SCAN, rho_hats, "o-", lw=2.0, ms=6, label=r"Empirical $\hat\rho(M)$")
ax.axhline(rho_lambda, ls="--", lw=1.6, label=rf"Biased-reference $\rho_\lambda={rho_lambda:.3f}$")
ax.axhline(rho_LQR, ls="-.", lw=1.5, label=rf"LQR spectral radius $={rho_LQR:.3f}$")
ax.axhline(1.0, ls=":", lw=1.3, label=r"$\hat\rho=1$")
ax.set_xlabel("Sample count $M$ (log scale)")
ax.set_ylabel(r"Empirical Lyapunov decay rate $\hat\rho(M)$")
ax.set_title(
    "Experiment 2: True Lyapunov Decay Diagnostic\n"
    r"($V(x)=x^\top P x$, noise-free; diagnostic, not certificate)"
)
ax.legend(loc="best", fontsize=8.5)
ax.grid(True, which="both", alpha=0.20)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig2_lyapunov_decay_rate.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "fig2_lyapunov_decay_rate.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Fig 2 saved")

# Figure 3: phase portrait
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharey=True)
for ax, trajs, M in [(axes[0], txy50, 50), (axes[1], txy500, 500)]:
    for tr in trajs:
        ax.plot(tr[:, 0], tr[:, 1], lw=0.8, alpha=0.35)
        ax.plot(tr[0, 0], tr[0, 1], "o", ms=3, alpha=0.5)
        ax.plot(tr[-1, 0], tr[-1, 1], "s", ms=3, alpha=0.7)
    ax.plot(0, 0, "k*", ms=10, label="Origin")
    ax.set_xlabel("$x_1$ (position)")
    ax.set_title(rf"$M={M}$")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
axes[0].set_ylabel("$x_2$ (velocity)")
fig.suptitle(
    rf"Experiment 3: Phase Portrait ($\sigma_w={sw0:.2f}$, no certificate labels)"
)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig3_phase_portrait.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "fig3_phase_portrait.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Fig 3 saved")

# Figure 4: fine low-M sweep
fine_M = [r[0] for r in fine_rows]
fine_rho = [r[2] for r in fine_rows]
fig, ax = plt.subplots(figsize=(5.8, 4.0))
ax.plot(fine_M, fine_rho, "o-", lw=1.8, ms=6, label=r"Empirical $\hat\rho$")
ax.axhline(
    EMPIRICAL_RHO_THRESHOLD, ls=":", lw=1.5,
    label=rf"Empirical threshold $\hat\rho<{EMPIRICAL_RHO_THRESHOLD:.2f}$",
)
if empirical_M is not None:
    ax.axvline(empirical_M, ls="--", lw=1.4, label=rf"First grid hit $M={empirical_M}$")
ax.set_xlabel("Sample count $M$")
ax.set_ylabel(r"Empirical $\hat\rho(M)$")
ax.set_title("Experiment 4: Finer Low-$M$ Empirical Sweep\n(not theoretical $M^*$)")
ax.legend(loc="best", fontsize=8.5)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig4_empirical_threshold_scan.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "fig4_empirical_threshold_scan.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Fig 4 saved")

# Figure 5: ESS diagnostic
fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.4), sharex=True)
t5 = np.arange(norm_e5.size)
axes[0].plot(t5, ess_e5, lw=1.2, label=r"$\mathrm{ESS}_k/M$")
axes[0].fill_between(t5, 0, ess_e5, alpha=0.18)
axes[0].set_ylabel("Normalized ESS")
axes[0].set_title("Experiment 5: ESS Diagnostic (descriptive only)")
axes[0].grid(alpha=0.20)
axes[0].legend(loc="upper right")
axes[1].plot(t5, norm_e5, lw=1.4, label=r"$\|x_k\|_2$")
axes[1].fill_between(t5, 0, norm_e5, alpha=0.10)
axes[1].set_xlabel("Time step $k$")
axes[1].set_ylabel(r"$\|x_k\|_2$")
axes[1].grid(alpha=0.20)
axes[1].legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig5_ess_diagnostic.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "fig5_ess_diagnostic.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Fig 5 saved")

print(f"\n  All repaired-audit figures saved to: {os.path.abspath(OUTDIR)}/")
hdr("DONE")
