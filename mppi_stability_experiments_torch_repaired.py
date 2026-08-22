"""P1 MPPI/LTI simulation audit aligned with the repaired theorem.

The finite-sample numerical certificate (D_U, epsilon, beta, Zbar_beta,
C_beta, M_star) is intentionally left pending until a deterministic bound on
nominal sequences and the target accuracy/confidence split are fixed.
"""
import argparse, os, time, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="figures_repaired")
parser.add_argument("--quick", action="store_true")
args = parser.parse_args(); os.makedirs(args.outdir, exist_ok=True)
OUTDIR = args.outdir
if torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
def tt(x): return torch.tensor(x, dtype=torch.float64, device=device)
def tn(x): return x.detach().cpu().numpy()

plt.rcParams.update({"font.family":"serif","font.size":11,"axes.spines.top":False,"axes.spines.right":False})

# System / LQR
A_np=np.array([[1.,1.],[0.,1.]])
B_np=np.array([[0.],[1.]])
Q_np=np.eye(2); R_np=np.array([[0.1]])
n,m=2,1; N_hor=10; lam=1.; sig_eps=1.; sigma_w_ref=0.10
P_np=solve_discrete_are(A_np,B_np,Q_np,R_np)
K_np=np.linalg.solve(R_np+B_np.T@P_np@B_np,B_np.T@P_np@A_np)
Acl_np=A_np-B_np@K_np
P_eigs=np.linalg.eigvalsh(P_np); lmin_P=P_eigs.min(); lmax_P=P_eigs.max()
c_const=float(np.sqrt(lmax_P/lmin_P))
rho_LQR=float(max(abs(np.linalg.eigvals(Acl_np))))
Sigma_w_ref_np=sigma_w_ref**2*np.eye(n)
Cw0_ref=float(np.trace(P_np@Sigma_w_ref_np))
Sigma_lqr_ss_np=solve_discrete_lyapunov(Acl_np,Sigma_w_ref_np)
dare_residual=np.max(np.abs(P_np-(Q_np+A_np.T@P_np@A_np-A_np.T@P_np@B_np@np.linalg.solve(R_np+B_np.T@P_np@B_np,B_np.T@P_np@A_np))))

# Stacked quadratic cost J=U'HU+2x'FU+x'Gx
Su_np=np.zeros((n*N_hor,m*N_hor)); Sx_np=np.zeros((n*N_hor,n)); Qbar_np=np.zeros((n*N_hor,n*N_hor))
for i in range(N_hor):
    Sx_np[i*n:(i+1)*n]=np.linalg.matrix_power(A_np,i+1)
    for j in range(i+1): Su_np[i*n:(i+1)*n,j*m:(j+1)*m]=np.linalg.matrix_power(A_np,i-j)@B_np
    Qbar_np[i*n:(i+1)*n,i*n:(i+1)*n]=Q_np if i<N_hor-1 else P_np
H_np=Su_np.T@Qbar_np@Su_np+np.kron(np.eye(N_hor),R_np)
Ft_np=Su_np.T@Qbar_np@Sx_np
mN=m*N_hor; Sigma_E_np=sig_eps**2*np.eye(mN)
Sigma_star_np=np.linalg.inv(np.linalg.inv(Sigma_E_np)+(2./lam)*H_np)
S0_np=np.zeros((m,mN)); S0_np[:,:m]=np.eye(m)
L_lambda_np=S0_np-(2./lam)*S0_np@Sigma_star_np@H_np
G_lambda_np=L_lambda_np@np.linalg.solve(H_np,Ft_np)
A_lambda_np=Acl_np+B_np@G_lambda_np
Q_lambda_np=P_np-A_lambda_np.T@P_np@A_lambda_np
Q_lambda_eigs=np.linalg.eigvalsh(Q_lambda_np)
alpha_lambda=float(Q_lambda_eigs.min())
q_lambda=float(1-alpha_lambda/(2*lmax_P))
rho_lambda=float(np.sqrt(q_lambda)) if 0<=q_lambda<1 else np.nan

A=tt(A_np); B=tt(B_np); H=tt(H_np); Ft=tt(Ft_np)

def hdr(s): print("\n"+"="*76+"\n  "+s+"\n"+"="*76)
def sec(s): print("\n"+"-"*76+"\n  "+s+"\n"+"-"*76)
def row(k,v): print(f"  {k:<45s}{v}")
hdr("MPPI STABILITY EXPERIMENTS — REPAIRED-THEOREM AUDIT")
row("Device",str(device)); row("Mode","QUICK" if args.quick else "FULL")
sec("Analytical parameters")
for k,v in [
    ("P eigenvalues",f"[{P_eigs[0]:.6f}, {P_eigs[1]:.6f}]"),
    ("K",f"[{K_np[0,0]:.6f}, {K_np[0,1]:.6f}]"),
    ("rho(A_cl)",f"{rho_LQR:.6f}"),("DARE residual",f"{dare_residual:.3e}"),
    ("c",f"{c_const:.6f}"),("C_w^(0), sigma_w=0.10",f"{Cw0_ref:.6f}"),
    ("||L_lambda||_2",f"{np.linalg.norm(L_lambda_np,2):.6f}"),
    ("G_lambda",np.array2string(G_lambda_np,precision=6)),
    ("eig(Q_lambda)",np.array2string(Q_lambda_eigs,precision=6)),
    ("alpha_lambda=lambda_min(Q_lambda)",f"{alpha_lambda:.6f}"),
    ("q_lambda=1-alpha/(2 lambda_max(P))",f"{q_lambda:.6f}"),
    ("rho_lambda=sqrt(q_lambda)",f"{rho_lambda:.6f}")]: row(k,v)
print("  NOTE: rho_lambda is the theorem transient factor after Young's inequality.")
print("  Finite-sample rho_hat(M) below is diagnostic, not a certificate.")

# MPPI / simulation
def mppi_step(x,U_nom,M,gen):
    eps=torch.randn(M,mN,dtype=torch.float64,device=device,generator=gen)*sig_eps
    U_s=U_nom.unsqueeze(0)+eps; rhs=Ft@x; HU=U_s@H
    costs=(U_s*HU).sum(1)+2*(U_s@rhs); b=costs.min(); w=torch.exp(-(costs-b)/lam); w=w/w.sum()
    U_new=U_nom+(w.unsqueeze(1)*eps).sum(0); u=U_new[:m]
    U_shift=torch.zeros(mN,dtype=torch.float64,device=device); U_shift[:(N_hor-1)*m]=U_new[m:]
    return u,U_shift,1/(w**2).sum()

def simulate_states(x0,M,sigma_w,T,gen):
    x=tt(x0); states=[tn(x).copy()]; ess=[float(M)]; U_nom=torch.zeros(mN,dtype=torch.float64,device=device)
    for _ in range(T):
        u,U_nom,e=mppi_step(x,U_nom,M,gen)
        noise=sigma_w*torch.randn(n,dtype=torch.float64,device=device,generator=gen) if sigma_w>0 else torch.zeros(n,dtype=torch.float64,device=device)
        x=A@x+B@u+noise; states.append(tn(x).copy()); ess.append(float(e))
    return np.asarray(states),np.asarray(ess)

def simulate_norms(x0,M,sigma_w,T,gen):
    x,e=simulate_states(x0,M,sigma_w,T,gen); return np.linalg.norm(x,axis=1),e

def lqr_norms(x0,sigma_w,T,gen):
    x=tt(x0); K=tt(K_np); out=[float(torch.linalg.norm(x))]
    for _ in range(T):
        noise=sigma_w*torch.randn(n,dtype=torch.float64,device=device,generator=gen) if sigma_w>0 else torch.zeros(n,dtype=torch.float64,device=device)
        x=A@x-B@(K@x)+noise; out.append(float(torch.linalg.norm(x)))
    return np.asarray(out)

def mc_mppi(x0,M,sigma_w,T,nmc,seed):
    gen=torch.Generator(device=device).manual_seed(seed); z=np.zeros((nmc,T+1))
    for i in range(nmc): z[i],_=simulate_norms(x0,M,sigma_w,T,gen)
    return z.mean(0),z.std(0)
def mc_lqr(x0,sigma_w,T,nmc,seed):
    gen=torch.Generator(device=device).manual_seed(seed); z=np.zeros((nmc,T+1))
    for i in range(nmc): z[i]=lqr_norms(x0,sigma_w,T,gen)
    return z.mean(0),z.std(0)
def lyap(states): return np.einsum("bi,ij,bj->b",states,P_np,states)

def estimate_decay_rate(M,ntraj=300,T=20,seed=1):
    gen=torch.Generator(device=device).manual_seed(seed); rng=np.random.default_rng(seed); vals=[]
    for _ in range(ntraj):
        x0=rng.standard_normal(n)*3
        if np.linalg.norm(x0)<0.1: continue
        states,_=simulate_states(x0,M,0.,T,gen); V=lyap(states); valid=V[:-1]>1e-10
        q=V[1:][valid]/V[:-1][valid]
        if q.size>3:
            a=q[2:min(10,q.size)]
            if a.size: vals.append(float(np.median(a)))
    if not vals: return np.nan,np.nan
    qh=float(np.median(vals)); return qh,float(np.sqrt(max(qh,0)))

x0=np.array([5.,5.]); T_sim=80 if args.quick else 200; sw0=sigma_w_ref
N_MC=40 if args.quick else 300; N_TRAJ=50 if args.quick else 300
M_SCAN=[10,20,50,100,200,500] if args.quick else [10,20,50,100,200,500,1000,5000,10000]
t=np.arange(T_sim+1); nominal=c_const*rho_lambda**t*np.linalg.norm(x0)

# EXP 1
sec("Experiment 1: Bound Envelope and Closed-Loop Response")
e1={}
for M in [50,200,1000]:
    st=time.perf_counter(); mn,sd=mc_mppi(x0,M,sw0,T_sim,N_MC,99); e1[M]=(mn,sd)
    print(f"  M={M:4d}: E||x_T||={mn[-1]:.5f}, std={sd[-1]:.5f}, {time.perf_counter()-st:.2f}s")
lqr_mean,lqr_std=mc_lqr(x0,sw0,T_sim,N_MC,199)
print("  The plotted c*rho_lambda^k||x0|| is only the theorem transient term;")
print("  confidence prefactor and practical residual are pending D_U/epsilon choices.")

# EXP 2
sec("Experiment 2: True Lyapunov Decay Diagnostic")
rho_hats=[]
for M in M_SCAN:
    qh,rh=estimate_decay_rate(M,N_TRAJ,20,1); rho_hats.append(rh)
    print(f"  M={M:5d}: q_hat={qh:.5f}, rho_hat={rh:.5f}")
print("  rho_hat=sqrt(median trajectory-wise median(V_{k+1}/V_k))).")

# EXP 3
sec("Experiment 3: Phase Portrait")
npp=12 if args.quick else 30; Tpp=25 if args.quick else 35; rng=np.random.default_rng(2)
inits=rng.standard_normal((npp,n))*2.5+np.array([2.5,2.5])
def phase(M,seed):
    gen=torch.Generator(device=device).manual_seed(seed); return np.asarray([simulate_states(z,M,sw0,Tpp,gen)[0] for z in inits])
txy50,txy500=phase(50,2),phase(500,3)
vals,vecs=np.linalg.eigh(Sigma_lqr_ss_np); th=np.linspace(0,2*np.pi,361); circ=np.vstack((np.cos(th),np.sin(th)))
ellipse=vecs@((2*np.sqrt(np.maximum(vals,0)))[:,None]*circ)

# EXP 4
sec("Experiment 4: Finer Low-M Empirical Sweep")
FINE=[5,10,15,20,30,50,75,100,150,200]; fine=[]; n4=60 if args.quick else 300
for M in FINE:
    qh,rh=estimate_decay_rate(M,n4,20,3); fine.append((M,qh,rh)); print(f"  M={M:4d}: rho_hat={rh:.5f}")
hits=[r for r in fine if np.isfinite(r[2]) and r[2]<0.99]; empirical_M=hits[0][0] if hits else None
print(f"  First grid hit rho_hat<0.99: {empirical_M}; diagnostic only, not theorem M_star.")

# EXP 5
sec("Experiment 5: ESS Diagnostic")
gen=torch.Generator(device=device).manual_seed(5); norm5,ess=simulate_norms(x0,500,sw0,T_sim,gen); ess=ess/500
print(f"  mean ESS/M={ess[1:].mean():.6f}; final ||x_T||={norm5[-1]:.6f}")

# Consistency
sec("Consistency checks")
checks=[("DARE residual",dare_residual<1e-10),("Q_lambda PD",Q_lambda_eigs.min()>0),("1/2<=q<1",0.5<=q_lambda<1),("rho^2=q",abs(rho_lambda**2-q_lambda)<1e-12),("Exp2 finite",np.all(np.isfinite(rho_hats)))]
for name,ok in checks: print(f"  {name:<24s}{'PASS' if ok else 'FAIL'}")
print("  Pending certificate: D_U, epsilon, beta, Zbar_beta, C_beta, M_star")

# Figures — filenames match the current LaTeX.
fig,ax=plt.subplots(figsize=(6.4,4.2))
for M in [50,200,1000]:
    mn,sd=e1[M]; ax.fill_between(t,np.maximum(mn-sd,0),mn+sd,alpha=.10); ax.plot(t,mn,lw=1.6,label=rf"MPPI $M={M}$")
ax.plot(t,nominal,"--",lw=1.8,label=rf"Nominal transient $c\rho_\lambda^k\|x_0\|$ ($\rho_\lambda={rho_lambda:.3f}$)")
ax.plot(t,lqr_mean,":",lw=1.8,label="LQR Monte Carlo reference"); ax.set(xlabel="Time step $k$",ylabel=r"$\mathbb{E}[\|x_k\|_2]$",title="Experiment 1: Bound Envelope and Closed-Loop Response"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig1_bound_verification.{ext}"),bbox_inches="tight",dpi=150 if ext=="png" else None)
plt.close(fig)

fig,ax=plt.subplots(figsize=(5.9,4.2)); ax.semilogx(M_SCAN,rho_hats,"o-",label=r"Empirical $\hat\rho(M)$"); ax.axhline(rho_lambda,ls="--",label=rf"Theorem transient $\rho_\lambda={rho_lambda:.3f}$"); ax.axhline(rho_LQR,ls="-.",label=rf"LQR spectral radius $={rho_LQR:.3f}$"); ax.axhline(1,ls=":"); ax.set(xlabel="Sample count $M$ (log scale)",ylabel=r"Empirical $\hat\rho(M)$",title="Experiment 2: True Lyapunov Decay Diagnostic"); ax.legend(); ax.grid(True,which="both",alpha=.2); fig.tight_layout()
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig2_decay_rate.{ext}"),bbox_inches="tight",dpi=150 if ext=="png" else None)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9.4,4.3),sharey=True)
for ax,trajs,M in [(axes[0],txy50,50),(axes[1],txy500,500)]:
    for tr in trajs: ax.plot(tr[:,0],tr[:,1],lw=.8,alpha=.35); ax.plot(tr[0,0],tr[0,1],"o",ms=3); ax.plot(tr[-1,0],tr[-1,1],"s",ms=3)
    ax.plot(ellipse[0],ellipse[1],"--",lw=1.5,label=r"LQR $2\sigma$ steady-state ellipse"); ax.plot(0,0,"k*",ms=10); ax.set(xlabel="$x_1$ (position)",title=rf"$M={M}$"); ax.grid(alpha=.2); ax.set_aspect("equal",adjustable="box")
axes[0].set_ylabel("$x_2$ (velocity)"); axes[0].legend(fontsize=8); fig.tight_layout()
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig3_phase_portrait.{ext}"),bbox_inches="tight",dpi=150 if ext=="png" else None)
plt.close(fig)

fig,ax=plt.subplots(figsize=(5.8,4)); ax.plot([r[0] for r in fine],[r[2] for r in fine],"o-"); ax.axhline(.99,ls=":",label=r"Empirical threshold $\hat\rho<0.99$");
if empirical_M is not None: ax.axvline(empirical_M,ls="--",label=rf"First grid hit $M={empirical_M}$")
ax.set(xlabel="Sample count $M$",ylabel=r"Empirical $\hat\rho(M)$",title="Experiment 4: Finer Low-$M$ Empirical Sweep"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig4_mstar_comparison.{ext}"),bbox_inches="tight",dpi=150 if ext=="png" else None)
plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(6.3,5.4),sharex=True); t5=np.arange(norm5.size); axes[0].plot(t5,ess); axes[0].set_ylabel("Normalized ESS"); axes[0].grid(alpha=.2); axes[1].plot(t5,norm5); axes[1].set(xlabel="Time step $k$",ylabel=r"$\|x_k\|_2$"); axes[1].grid(alpha=.2); fig.tight_layout()
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig5_ess_diagnostic.{ext}"),bbox_inches="tight",dpi=150 if ext=="png" else None)
plt.close(fig)
print(f"\nFigures saved to {os.path.abspath(OUTDIR)}")
