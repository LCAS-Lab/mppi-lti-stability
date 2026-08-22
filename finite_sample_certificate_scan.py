"""Numerical audit of the P1 finite-sample certificate.

This script implements the current manuscript formulas without changing the
stated theorem. All denominator and sample-count calculations are performed
in the log domain because the sufficient bounds can be extremely conservative.

The current manuscript lower bound retains the exact Gaussian quadratic
correction through
    Gamma_lambda = H/lambda - 2 H Sigma_* H/lambda^2
                 = (lambda H^{-1} + 2 Sigma_E)^{-1}.
For historical comparison only, the script also reports the older relaxation
obtained by dropping the positive correction and using ||H||/lambda. That
legacy block is NOT the current manuscript certificate.
"""
import argparse
import math
import numpy as np
from scipy.linalg import solve_discrete_are

parser = argparse.ArgumentParser()
parser.add_argument("--T", type=int, default=200)
parser.add_argument("--delta", type=float, default=0.05)
parser.add_argument("--delta-s-frac", type=float, default=0.5,
                    help="fraction of total delta allocated to sampling failures")
parser.add_argument("--D-U", dest="D_U", type=float, default=9.0)
parser.add_argument("--sigma-w", type=float, default=0.10)
parser.add_argument("--eps", nargs="+", type=float,
                    default=[0.05, 0.10, 0.25, 0.50, 1.00])
args = parser.parse_args()

if args.T < 1:
    raise ValueError("T must be at least 1")
if not (0.0 < args.delta < 1.0):
    raise ValueError("delta must lie in (0,1)")
if not (0.0 < args.delta_s_frac < 1.0):
    raise ValueError("delta-s-frac must lie in (0,1)")
if args.D_U < 0.0:
    raise ValueError("D_U must be nonnegative")
if any(e <= 0.0 for e in args.eps):
    raise ValueError("all epsilon values must be positive")

# Same benchmark as the main repaired simulation.
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])
Q = np.eye(2)
R = np.array([[0.1]])
n, m = 2, 1
N = 10
lam = 1.0
sigma_eps = 1.0
Sigma_eps = np.array([[sigma_eps**2]])
Sigma_E = np.kron(np.eye(N), Sigma_eps)
x0 = np.array([5.0, 5.0])
Sigma_w = args.sigma_w**2 * np.eye(n)

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
Acl = A - B @ K
P_eigs = np.linalg.eigvalsh(P)
lmin_P = float(P_eigs.min())
lmax_P = float(P_eigs.max())

# Stacked finite-horizon quadratic cost.
Su = np.zeros((n*N, m*N))
Sx = np.zeros((n*N, n))
Qbar = np.zeros((n*N, n*N))
for i in range(N):
    Sx[i*n:(i+1)*n] = np.linalg.matrix_power(A, i+1)
    for j in range(i+1):
        Su[i*n:(i+1)*n, j*m:(j+1)*m] = np.linalg.matrix_power(A, i-j) @ B
    Qbar[i*n:(i+1)*n, i*n:(i+1)*n] = Q if i < N-1 else P
H = Su.T @ Qbar @ Su + np.kron(np.eye(N), R)
Ft = Su.T @ Qbar @ Sx

Sigma_star = np.linalg.inv(np.linalg.inv(Sigma_E) + (2.0/lam)*H)
S0 = np.zeros((m, m*N))
S0[:, :m] = np.eye(m)
L_lambda = S0 - (2.0/lam) * S0 @ Sigma_star @ H
G_lambda = L_lambda @ np.linalg.solve(H, Ft)
A_lambda = Acl + B @ G_lambda
Q_lambda = P - A_lambda.T @ P @ A_lambda
alpha_lambda = float(np.linalg.eigvalsh(Q_lambda).min())
q_lambda = float(1.0 - alpha_lambda/(2.0*lmax_P))
rho_lambda = math.sqrt(q_lambda)

L_norm = float(np.linalg.norm(L_lambda, 2))
Ce = float(np.linalg.norm(B.T @ P @ B, 2)
           + 2.0*np.linalg.norm(A_lambda.T @ P @ B, 2)**2/alpha_lambda)
Cw0 = float(np.trace(P @ Sigma_w))
V0 = float(x0 @ P @ x0)
H_norm = float(np.linalg.norm(H, 2))
HinvFt_norm = float(np.linalg.norm(np.linalg.solve(H, Ft), 2))
sigma_max = float(np.sqrt(np.max(np.diag(Sigma_eps))))
tr_Sigma_eps = float(np.trace(Sigma_eps))

# det(I + 2 Sigma_E^{1/2} H Sigma_E^{1/2}/lambda)^{-1/2}
Sigma_E_half = np.linalg.cholesky(Sigma_E)
Dmat = np.eye(m*N) + (2.0/lam) * Sigma_E_half @ H @ Sigma_E_half.T
sign, logdet = np.linalg.slogdet(Dmat)
if sign <= 0:
    raise RuntimeError("determinant factor is not positive")
log_det_prefactor = -0.5*logdet

# Exact quadratic matrix in the Gaussian denominator exponent.
Gamma_lambda = H/lam - (2.0/lam**2) * H @ Sigma_star @ H
Gamma_lambda = 0.5*(Gamma_lambda + Gamma_lambda.T)
Gamma_alt = np.linalg.inv(lam*np.linalg.inv(H) + 2.0*Sigma_E)
Gamma_eigs = np.linalg.eigvalsh(Gamma_lambda)
Gamma_norm = float(np.linalg.norm(Gamma_lambda, 2))
Gamma_identity_residual = float(np.max(np.abs(Gamma_lambda-Gamma_alt)))


def logaddexp_many(vals):
    a = max(vals)
    return a + math.log(sum(math.exp(v-a) for v in vals))


def certificate_row(epsilon, quad_norm):
    delta_s = args.delta * args.delta_s_frac
    delta_x = args.delta - delta_s
    eta = delta_s / args.T

    e_epsilon = L_norm*args.D_U + epsilon
    Delta = Cw0 + Ce*e_epsilon**2
    beta_exit = (
        q_lambda*(1.0-q_lambda**args.T)*V0 + args.T*Delta
    ) / (delta_x*(1.0-q_lambda))
    beta = max(V0, beta_exit)

    r_beta = math.sqrt(beta/lmin_P)
    d_beta = args.D_U + HinvFt_norm*r_beta
    log_Zbar = log_det_prefactor - quad_norm*d_beta**2

    ell_T = math.log(4.0*m*args.T/delta_s)
    log_M_bernstein = math.log(ell_T)
    log_M_den = math.log(2.0*math.log(4.0*args.T/delta_s)) - 2.0*log_Zbar

    # C_beta = 16 sigma_max sqrt(m)/Z + sqrt(2 tr Sigma_eps)/Z^2
    log_C1 = math.log(16.0*sigma_max*math.sqrt(m)) - log_Zbar
    log_C2 = 0.5*math.log(2.0*tr_Sigma_eps) - 2.0*log_Zbar
    log_Cbeta = logaddexp_many([log_C1, log_C2])
    log_M_accuracy = 2.0*log_Cbeta - 2.0*math.log(epsilon) + math.log(ell_T)
    log_Mstar = max(log_M_bernstein, log_M_den, log_M_accuracy)

    residual_radius = math.sqrt(Delta/((1.0-q_lambda)*lmin_P))
    return {
        "epsilon": epsilon,
        "eta": eta,
        "e": e_epsilon,
        "Delta": Delta,
        "beta": beta,
        "r_beta": r_beta,
        "d_beta": d_beta,
        "log_Z": log_Zbar,
        "log10_Z": log_Zbar/math.log(10.0),
        "log10_M_den": log_M_den/math.log(10.0),
        "log10_M_acc": log_M_accuracy/math.log(10.0),
        "log10_Mstar": log_Mstar/math.log(10.0),
        "residual_radius": residual_radius,
    }


def print_block(title, quad_norm):
    print("\n" + title)
    print("-"*118)
    print(f"  exponent matrix norm used = {quad_norm:.9g}")
    print("  eps       e_eps       Delta        beta        d_beta      log10(Zbar)   log10(Mden)   log10(Macc)   log10(M*)")
    rows = []
    for eps in args.eps:
        r = certificate_row(eps, quad_norm)
        rows.append(r)
        print(f"  {eps:5.3f}  {r['e']:10.4f}  {r['Delta']:10.4f}  {r['beta']:11.4g}  {r['d_beta']:11.4f}"
              f"  {r['log10_Z']:12.3f}  {r['log10_M_den']:12.3f}  {r['log10_M_acc']:12.3f}  {r['log10_Mstar']:11.3f}")
    return rows


print("\n" + "="*118)
print("  P1 FINITE-SAMPLE CERTIFICATE NUMERICAL AUDIT")
print("="*118)
print(f"  T={args.T}, delta={args.delta:g}, delta_s={args.delta*args.delta_s_frac:g}, "
      f"delta_x={args.delta*(1-args.delta_s_frac):g}, D_U={args.D_U:g}, sigma_w={args.sigma_w:g}")
print(f"  V(x0)={V0:.6f}, ||L_lambda||={L_norm:.6f}, C_e,lambda={Ce:.6f}, C_w^(0)={Cw0:.6f}")
print(f"  q_lambda={q_lambda:.6f}, rho_lambda={rho_lambda:.6f}, ||H||={H_norm:.6f}, "
      f"||H^(-1)F^T||={HinvFt_norm:.6f}")
print(f"  log(det prefactor)={log_det_prefactor:.6f}")
print(f"  Gamma_lambda eig range=[{Gamma_eigs.min():.6f}, {Gamma_eigs.max():.6f}], "
      f"||Gamma_lambda||={Gamma_norm:.6f}")
print(f"  Gamma identity residual={Gamma_identity_residual:.3e}")

rows_manuscript = print_block(
    "CURRENT MANUSCRIPT BOUND: exact Gamma_lambda quadratic exponent", Gamma_norm)
rows_legacy = print_block(
    "LEGACY DIAGNOSTIC ONLY: drop positive correction and use ||H||/lambda", H_norm/lam)

print("\nInterpretation")
print("-"*118)
print("  1) The first block is the certificate stated in the repaired manuscript.")
print("  2) The second block is retained only to quantify the conservatism removed by the Gamma_lambda sharpening.")
print("  3) Very large log10(M*) values indicate conservatism of the sufficient uniform bound, not an empirical instability threshold.")
print("  4) Do not exponentiate log10(M*) when it is large; report it in logarithmic form during the audit.")
