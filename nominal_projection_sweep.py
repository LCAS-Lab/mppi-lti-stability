"""Diagnostic sweep for the bounded nominal-sequence assumption in P1.

Purpose
-------
Compare radial projection radii D_U in {8, 9, 10} against the unprojected
warm-start implementation.  Projection is applied only to the shifted nominal
sequence used as the next MPPI sampling center; the current applied action is
not clipped.

This script is a certificate-design diagnostic.  It does not itself compute
M_star.
"""
import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_discrete_are

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="figures_repaired")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def tt(x):
    return torch.tensor(x, dtype=torch.float64, device=device)

# Same benchmark and MPPI parameters as mppi_stability_experiments_torch_repaired.py
A_np = np.array([[1., 1.], [0., 1.]])
B_np = np.array([[0.], [1.]])
Q_np = np.eye(2)
R_np = np.array([[0.1]])
n, m = 2, 1
N_hor = 10
lam = 1.0
sig_eps = 1.0
sigma_w = 0.10
mN = m * N_hor
x0 = np.array([5., 5.])

P_np = solve_discrete_are(A_np, B_np, Q_np, R_np)
Su_np = np.zeros((n*N_hor, m*N_hor))
Sx_np = np.zeros((n*N_hor, n))
Qbar_np = np.zeros((n*N_hor, n*N_hor))
for i in range(N_hor):
    Sx_np[i*n:(i+1)*n] = np.linalg.matrix_power(A_np, i+1)
    for j in range(i+1):
        Su_np[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A_np, i-j) @ B_np
    Qbar_np[i*n:(i+1)*n, i*n:(i+1)*n] = Q_np if i < N_hor-1 else P_np
H_np = Su_np.T @ Qbar_np @ Su_np + np.kron(np.eye(N_hor), R_np)
Ft_np = Su_np.T @ Qbar_np @ Sx_np

A, B, H, Ft = tt(A_np), tt(B_np), tt(H_np), tt(Ft_np)


def mppi_step(x, U_nom, M, gen, D_U=None):
    """One MPPI step; optionally project only the next shifted sampling center."""
    eps = torch.randn(M, mN, dtype=torch.float64, device=device, generator=gen) * sig_eps
    U_s = U_nom.unsqueeze(0) + eps
    rhs = Ft @ x
    HU = U_s @ H
    costs = (U_s * HU).sum(1) + 2 * (U_s @ rhs)
    b = costs.min()
    w = torch.exp(-(costs-b)/lam)
    w = w / w.sum()
    U_new = U_nom + (w.unsqueeze(1) * eps).sum(0)
    u = U_new[:m]

    U_shift = torch.zeros(mN, dtype=torch.float64, device=device)
    U_shift[:(N_hor-1)*m] = U_new[m:]
    pre_norm = float(torch.linalg.norm(U_shift))
    activated = False
    if D_U is not None and pre_norm > D_U:
        U_shift = U_shift * (D_U / pre_norm)
        activated = True
    post_norm = float(torch.linalg.norm(U_shift))
    return u, U_shift, pre_norm, post_norm, activated


def run_case(M, D_U, ntraj, T, seed):
    # Resetting the generator for each D_U gives common random numbers across cases.
    gen = torch.Generator(device=device).manual_seed(seed)
    terminal = np.empty(ntraj)
    n_updates = 0
    n_active = 0
    max_pre = 0.0
    max_post = 0.0
    for i in range(ntraj):
        x = tt(x0)
        U_nom = torch.zeros(mN, dtype=torch.float64, device=device)
        for _ in range(T):
            u, U_nom, pre, post, active = mppi_step(x, U_nom, M, gen, D_U)
            max_pre = max(max_pre, pre)
            max_post = max(max_post, post)
            n_updates += 1
            n_active += int(active)
            noise = sigma_w * torch.randn(n, dtype=torch.float64, device=device, generator=gen)
            x = A @ x + B @ u + noise
        terminal[i] = float(torch.linalg.norm(x))
    return {
        "mean_terminal": float(terminal.mean()),
        "std_terminal": float(terminal.std()),
        "activation_rate": n_active / n_updates,
        "n_active": n_active,
        "n_updates": n_updates,
        "max_pre": max_pre,
        "max_post": max_post,
    }


T = 80 if args.quick else 200
N_TRAJ = 40 if args.quick else 300
M_VALUES = [50, 200, 1000]
D_VALUES = [8.0, 9.0, 10.0]
SEEDS = {50: 8050, 200: 8200, 1000: 9000}

print("\n" + "="*88)
print("  NOMINAL-SEQUENCE PROJECTION SWEEP")
print("="*88)
print(f"  Device: {device}; mode: {'QUICK' if args.quick else 'FULL'}")
print("  Projection is applied only to the shifted nominal sequence for the next step.")
print("  The current MPPI action u_k is not clipped.\n")

results = {}
for M in M_VALUES:
    print(f"M={M}")
    base = run_case(M, None, N_TRAJ, T, SEEDS[M])
    results[(M, None)] = base
    print(f"  unprojected: E||x_T||={base['mean_terminal']:.5f} +/- {base['std_terminal']:.5f}, "
          f"max shifted norm={base['max_pre']:.5f}")
    for D in D_VALUES:
        r = run_case(M, D, N_TRAJ, T, SEEDS[M])
        results[(M, D)] = r
        delta = r['mean_terminal'] - base['mean_terminal']
        print(f"  D_U={D:4.1f}: activation={100*r['activation_rate']:.4f}% "
              f"({r['n_active']}/{r['n_updates']}), max pre={r['max_pre']:.5f}, "
              f"max post={r['max_post']:.5f}, E||x_T||={r['mean_terminal']:.5f} "
              f"(delta={delta:+.5f})")
    print()

# Hard numerical sanity check for the deterministic implementation bound.
tol = 1e-10
for M in M_VALUES:
    for D in D_VALUES:
        assert results[(M, D)]["max_post"] <= D + tol
print("  Projection-bound check: PASS for all projected cases.")

# Figure 1: activation rate versus D_U
fig, ax = plt.subplots(figsize=(6.0, 4.2))
for M in M_VALUES:
    y = [100*results[(M, D)]["activation_rate"] for D in D_VALUES]
    ax.plot(D_VALUES, y, "o-", label=rf"$M={M}$")
ax.set_xlabel(r"Projection radius $D_U$")
ax.set_ylabel("Projection activation rate (%)")
ax.set_title("Nominal-sequence projection frequency")
ax.grid(alpha=.2)
ax.legend()
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(args.outdir, f"fig6_projection_activation.{ext}"),
                bbox_inches="tight", dpi=150 if ext == "png" else None)
plt.close(fig)

# Figure 2: terminal performance relative to the unprojected implementation
fig, ax = plt.subplots(figsize=(6.0, 4.2))
for M in M_VALUES:
    base = results[(M, None)]["mean_terminal"]
    y = [results[(M, D)]["mean_terminal"] - base for D in D_VALUES]
    ax.plot(D_VALUES, y, "o-", label=rf"$M={M}$")
ax.axhline(0, ls=":")
ax.set_xlabel(r"Projection radius $D_U$")
ax.set_ylabel(r"Change in mean terminal norm")
ax.set_title("Effect of projection on closed-loop performance")
ax.grid(alpha=.2)
ax.legend()
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(args.outdir, f"fig6_projection_performance.{ext}"),
                bbox_inches="tight", dpi=150 if ext == "png" else None)
plt.close(fig)

print(f"  Figures saved to {os.path.abspath(args.outdir)}")
