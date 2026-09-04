"""P1 MPPI/LTI canonical paper simulation aligned with the repaired theorem.

The main experiments enforce the bounded nominal-sequence assumption using a
radial projection with D_U=9. The current applied action is not clipped; only
the shifted nominal sequence used as the next sampling center is projected.
The finite-sample numerical certificate is audited separately in
finite_sample_certificate_scan.py.
"""
import argparse, os, time, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="figures")
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
n,m=2,1; N_hor=10; lam=1.; sig_eps=1.; sigma_w_ref=0.10; D_U_MAIN=9.0
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
hdr("MPPI STABILITY EXPERIMENTS — CANONICAL REPAIRED VERSION")
row("Device",str(device)); row("Mode","QUICK" if args.quick else "FULL")
sec("Analytical parameters")
for k,v in [
    ("P eigenvalues",f"[{P_eigs[0]:.6f}, {P_eigs[1]:.6f}]"),
    ("K",f"[{K_np[0,0]:.6f}, {K_np[0,1]:.6f}]"),
    ("rho(A_cl)",f"{rho_LQR:.6f}"),("DARE residual",f"{dare_residual:.3e}"),
    ("c",f"{c_const:.6f}"),("C_w^(0), sigma_w=0.10",f"{Cw0_ref:.6f}"),
    ("D_U (enforced nominal-sequence bound)",f"{D_U_MAIN:.6f}"),
    ("||L_lambda||_2",f"{np.linalg.norm(L_lambda_np,2):.6f}"),
    ("G_lambda",np.array2string(G_lambda_np,precision=6)),
    ("eig(Q_lambda)",np.array2string(Q_lambda_eigs,precision=6)),
    ("alpha_lambda=lambda_min(Q_lambda)",f"{alpha_lambda:.6f}"),
    ("q_lambda=1-alpha/(2 lambda_max(P))",f"{q_lambda:.6f}"),
    ("rho_lambda=sqrt(q_lambda)",f"{rho_lambda:.6f}")]: row(k,v)
print("  NOTE: rho_lambda is the theorem transient factor after Young's inequality.")
print("  Finite-sample rho_hat(M) below is diagnostic, not a certificate.")
print("  Main simulations radially project only the next nominal sampling center.")

# MPPI / simulation
def mppi_step(x,U_nom,M,gen,D_U=D_U_MAIN):
    eps=torch.randn(M,mN,dtype=torch.float64,device=device,generator=gen)*sig_eps
    U_s=U_nom.unsqueeze(0)+eps; rhs=Ft@x; HU=U_s@H
    costs=(U_s*HU).sum(1)+2*(U_s@rhs); b=costs.min(); w=torch.exp(-(costs-b)/lam); w=w/w.sum()
    U_new=U_nom+(w.unsqueeze(1)*eps).sum(0); u=U_new[:m]
    U_shift=torch.zeros(mN,dtype=torch.float64,device=device); U_shift[:(N_hor-1)*m]=U_new[m:]
    if D_U is not None:
        unorm=torch.linalg.norm(U_shift)
        if float(unorm)>D_U: U_shift=U_shift*(D_U/unorm)
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

def collect_nominal_sequence_norms(x0,M,sigma_w,T,ntraj,seed):
    """Collect pre-sampling ||U_bar_k||_2 for the unprojected implementation."""
    gen=torch.Generator(device=device).manual_seed(seed); out=[]
    for _ in range(ntraj):
        x=tt(x0); U_nom=torch.zeros(mN,dtype=torch.float64,device=device)
        for _ in range(T):
            out.append(float(torch.linalg.norm(U_nom)))
            u,U_nom,_=mppi_step(x,U_nom,M,gen,D_U=None)
            noise=sigma_w*torch.randn(n,dtype=torch.float64,device=device,generator=gen) if sigma_w>0 else torch.zeros(n,dtype=torch.float64,device=device)
            x=A@x+B@u+noise
    return np.asarray(out)

x0=np.array([5.,5.]); T_sim=80 if args.quick else 200; sw0=sigma_w_ref
N_MC=40 if args.quick else 300; N_TRAJ=50 if args.quick else 300
M_SCAN=[10,20,50,100,200,500] if args.quick else [10,20,50,100,200,500,1000,5000,10000]
t=np.arange(T_sim+1); nominal=c_const*rho_lambda**t*np.linalg.norm(x0)

# EXP 1
sec("Experiment 1: Closed-Loop Response and Nominal Transient Reference")
e1={}
for M in [50,200,1000]:
    st=time.perf_counter(); mn,sd=mc_mppi(x0,M,sw0,T_sim,N_MC,99); e1[M]=(mn,sd)
    print(f"  M={M:4d}: E||x_T||={mn[-1]:.5f}, std={sd[-1]:.5f}, {time.perf_counter()-st:.2f}s")
lqr_mean,lqr_std=mc_lqr(x0,sw0,T_sim,N_MC,199)
print("  The plotted c*rho_lambda^k||x0|| is only the theorem transient term;")
print("  confidence prefactor and practical residual depend on the epsilon/confidence choices.")

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
for M,tr in [(50,txy50),(500,txy500)]:
    fn=np.linalg.norm(tr[:,-1,:],axis=1)
    print(f"  M={M:4d}: mean ||x_T||={fn.mean():.5f}, max={fn.max():.5f}, count(||x_T||>5)={(fn>5).sum()}")
vals,vecs=np.linalg.eigh(Sigma_lqr_ss_np); th=np.linspace(0,2*np.pi,361); circ=np.vstack((np.cos(th),np.sin(th)))
ellipse=vecs@((2*np.sqrt(np.maximum(vals,0)))[:,None]*circ)

# EXP 4
sec("Experiment 4: ESS Diagnostic")
gen=torch.Generator(device=device).manual_seed(5); norm4,ess=simulate_norms(x0,500,sw0,T_sim,gen); ess=ess/500
print(f"  mean ESS/M={ess[1:].mean():.6f}; final ||x_T||={norm4[-1]:.6f}")

# EXP 5
sec("Experiment 5: Unprojected Nominal-Sequence Norm Diagnostic")
N_NOM=40 if args.quick else 300
nominal_norms={}
print(f"  {'M':>6} {'samples':>9} {'p50':>9} {'p95':>9} {'p99':>9} {'p99.9':>9} {'max':>9}")
for M,seed in [(50,551),(200,701),(1000,1501)]:
    z=collect_nominal_sequence_norms(x0,M,sw0,T_sim,N_NOM,seed); nominal_norms[M]=z
    p=np.percentile(z,[50,95,99,99.9])
    print(f"  {M:6d} {z.size:9d} {p[0]:9.4f} {p[1]:9.4f} {p[2]:9.4f} {p[3]:9.4f} {z.max():9.4f}")
print(f"  Diagnostic is deliberately unprojected; the main experiments enforce D_U={D_U_MAIN:g}.")
print("  The upper tail documents that the safeguard is rarely active in the benchmark.")

# Consistency
sec("Consistency checks")
checks=[("DARE residual",dare_residual<1e-10),("Q_lambda PD",Q_lambda_eigs.min()>0),("1/2<=q<1",0.5<=q_lambda<1),("rho^2=q",abs(rho_lambda**2-q_lambda)<1e-12),("Exp2 finite",np.all(np.isfinite(rho_hats))),("Exp5 finite",all(np.all(np.isfinite(z)) for z in nominal_norms.values()))]
for name,ok in checks: print(f"  {name:<24s}{'PASS' if ok else 'FAIL'}")
print(f"  Enforced implementation bound: D_U={D_U_MAIN:g}")
print("  Numerical certificate is audited separately in finite_sample_certificate_scan.py")

# Figures
# Paper figures 1 and 2 intentionally use identical canvas size and margins.
# Their internal titles are omitted because the LaTeX subcaptions provide them.
PAPER_FIGSIZE=(6.2,3.65)
PAPER_MARGINS=dict(left=.135,right=.985,bottom=.20,top=.985)

fig,ax=plt.subplots(figsize=PAPER_FIGSIZE)
for M in [50,200,1000]:
    mn,sd=e1[M]; ax.fill_between(t,np.maximum(mn-sd,0),mn+sd,alpha=.10); ax.plot(t,mn,lw=1.6,label=rf"MPPI $M={M}$")
ax.plot(t,nominal,"--",lw=1.8,label=rf"Nominal transient $c\rho_\lambda^k\|x_0\|$ ($\rho_\lambda={rho_lambda:.3f}$)")
ax.plot(t,lqr_mean,":",lw=1.8,label="LQR Monte Carlo reference")
ax.set(xlabel="Time step $k$",ylabel=r"$\mathbb{E}[\|x_k\|_2]$")
ax.legend(fontsize=9,handlelength=2.2)
ax.grid(alpha=.25)
fig.subplots_adjust(**PAPER_MARGINS)
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig1_bound_verification.{ext}"),dpi=150 if ext=="png" else None)
plt.close(fig)

fig,ax=plt.subplots(figsize=PAPER_FIGSIZE)
ax.semilogx(M_SCAN,rho_hats,"o-",label=r"Empirical $\hat\rho(M)$")
ax.axhline(rho_lambda,ls="--",label=rf"Theorem transient $\rho_\lambda={rho_lambda:.3f}$")
ax.axhline(rho_LQR,ls="-.",label=rf"LQR spectral radius $={rho_LQR:.3f}$")
ax.axhline(1,ls=":")
ax.set(xlabel="Sample count $M$ (log scale)",ylabel=r"Empirical $\hat\rho(M)$")
ax.legend(fontsize=9,handlelength=2.2)
ax.grid(True,which="both",alpha=.2)
fig.subplots_adjust(**PAPER_MARGINS)
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig2_decay_rate.{ext}"),dpi=150 if ext=="png" else None)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9.4,4.3),sharex=True,sharey=True)
all_x=np.concatenate([txy50[:,:,0].ravel(),txy500[:,:,0].ravel(),ellipse[0].ravel(),np.array([0.])])
all_y=np.concatenate([txy50[:,:,1].ravel(),txy500[:,:,1].ravel(),ellipse[1].ravel(),np.array([0.])])
xpad=.08*(all_x.max()-all_x.min()+1e-12); ypad=.08*(all_y.max()-all_y.min()+1e-12)
xlim=(all_x.min()-xpad,all_x.max()+xpad); ylim=(all_y.min()-ypad,all_y.max()+ypad)
for ax,trajs,M in [(axes[0],txy50,50),(axes[1],txy500,500)]:
    for tr in trajs: ax.plot(tr[:,0],tr[:,1],lw=.8,alpha=.35); ax.plot(tr[0,0],tr[0,1],"o",ms=3); ax.plot(tr[-1,0],tr[-1,1],"s",ms=3)
    ax.plot(ellipse[0],ellipse[1],"--",lw=1.5,label=r"LQR $2\sigma$ steady-state ellipse"); ax.plot(0,0,"k*",ms=10); ax.set(xlabel="$x_1$ (position)",title=rf"$M={M}$"); ax.grid(alpha=.2); ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal",adjustable="box")
axes[0].set_ylabel("$x_2$ (velocity)")
handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.02),fontsize=8)
fig.tight_layout(rect=[0,0,1,.95])
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig3_phase_portrait.{ext}"),bbox_inches="tight",dpi=150 if ext=="png" else None)
plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(6.3,5.4),sharex=True); t4=np.arange(norm4.size); axes[0].plot(t4,ess); axes[0].set_ylabel("Normalized ESS"); axes[0].set_title("Experiment 4: ESS Diagnostic"); axes[0].grid(alpha=.2); axes[1].plot(t4,norm4); axes[1].set(xlabel="Time step $k$",ylabel=r"$\|x_k\|_2$"); axes[1].grid(alpha=.2); fig.tight_layout()
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig4_ess_diagnostic.{ext}"),bbox_inches="tight",dpi=150 if ext=="png" else None)
plt.close(fig)

fig,ax=plt.subplots(figsize=(6.0,4.2))
for M in [50,200,1000]:
    z=np.sort(nominal_norms[M]); ecdf=np.arange(1,z.size+1)/z.size
    ax.plot(z,ecdf,lw=1.7,label=rf"$M={M}$")
for p in [0.95,0.99,0.999]: ax.axhline(p,ls=":",lw=1.0)
ax.set(xlabel=r"Pre-sampling nominal-sequence norm $\|\bar U_k\|_2$",ylabel="Empirical CDF",title="Experiment 5: Nominal-Sequence Norm Diagnostic (no projection)")
ax.set_ylim(0.9,1.001); ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
for ext in ["pdf","png"]: fig.savefig(os.path.join(OUTDIR,f"fig5_nominal_sequence_norm_cdf.{ext}"),bbox_inches="tight",dpi=150 if ext=="png" else None)
plt.close(fig)
print(f"\nFigures saved to {os.path.abspath(OUTDIR)}")