#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FIGURE 7: Stochastic processes comparison
# ============================================================
np.random.seed(123)
T_sim = 1.0; N_steps = 1000; N_paths = 8
dt_s = T_sim / N_steps
t_s = np.linspace(0, T_sim, N_steps+1)

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='white')

# GBM
for _ in range(N_paths):
    W = np.cumsum(np.sqrt(dt_s)*np.random.randn(N_steps))
    W = np.insert(W, 0, 0)
    S_gbm = 100 * np.exp((0.08 - 0.5*0.25**2)*t_s + 0.25*W)
    axes[0,0].plot(t_s, S_gbm, lw=0.8, alpha=0.7)
axes[0,0].set_title('Geometric Brownian Motion\n$dS = \\mu S\\,dt + \\sigma S\\,dW$', fontsize=11, fontweight='bold')
axes[0,0].set_xlabel('Time'); axes[0,0].set_ylabel('Price')
axes[0,0].grid(alpha=0.3)

# Ornstein-Uhlenbeck (mean-reverting, e.g. freight rates)
kappa, theta_ou, sigma_ou = 3.0, 50000, 8000
for _ in range(N_paths):
    X = np.zeros(N_steps+1)
    X[0] = 40000
    for i in range(1, N_steps+1):
        X[i] = X[i-1] + kappa*(theta_ou - X[i-1])*dt_s + sigma_ou*np.sqrt(dt_s)*np.random.randn()
    axes[0,1].plot(t_s, X/1000, lw=0.8, alpha=0.7)
axes[0,1].axhline(theta_ou/1000, color='k', lw=2, ls='--', label=f'θ={theta_ou/1000:.0f}k')
axes[0,1].set_title('Ornstein-Uhlenbeck Process\n$dX = \\kappa(\\theta-X)dt + \\sigma\\,dW$\n(Freight Rates)', fontsize=11, fontweight='bold')
axes[0,1].set_xlabel('Time'); axes[0,1].set_ylabel('Rate (USD/day × 10³)')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

# Heston model paths
v0, kappa_h, theta_h, xi, rho = 0.04, 2.0, 0.04, 0.3, -0.7
for _ in range(N_paths):
    S_h = np.zeros(N_steps+1); v_h = np.zeros(N_steps+1)
    S_h[0] = 100; v_h[0] = v0
    for i in range(1, N_steps+1):
        z1 = np.random.randn(); z2 = np.random.randn()
        zv = z1; zs = rho*z1 + np.sqrt(1-rho**2)*z2
        v_h[i] = max(v_h[i-1] + kappa_h*(theta_h - v_h[i-1])*dt_s + xi*np.sqrt(max(v_h[i-1],0)*dt_s)*zv, 1e-6)
        S_h[i] = S_h[i-1] * np.exp((0.05 - 0.5*v_h[i-1])*dt_s + np.sqrt(max(v_h[i-1],0)*dt_s)*zs)
    axes[0,2].plot(t_s, S_h, lw=0.8, alpha=0.7)
axes[0,2].set_title('Heston Stochastic Volatility\n$(dS, dv)$ with Correlation $\\rho$', fontsize=11, fontweight='bold')
axes[0,2].set_xlabel('Time'); axes[0,2].set_ylabel('Price')
axes[0,2].grid(alpha=0.3)

# Jump diffusion (Merton) — Red Sea shock events
lam_j, mu_j, sigma_j = 2.0, -0.05, 0.08
for _ in range(N_paths):
    S_jd = np.zeros(N_steps+1); S_jd[0] = 100
    for i in range(1, N_steps+1):
        N_jumps = np.random.poisson(lam_j*dt_s)
        jump = np.sum(np.random.normal(mu_j, sigma_j, N_jumps)) if N_jumps > 0 else 0
        S_jd[i] = S_jd[i-1] * np.exp((0.05 - 0.5*0.2**2 - lam_j*(np.exp(mu_j+0.5*sigma_j**2)-1))*dt_s
                                        + 0.2*np.sqrt(dt_s)*np.random.randn() + jump)
    axes[1,0].plot(t_s, S_jd, lw=0.8, alpha=0.7)
axes[1,0].set_title('Merton Jump-Diffusion\n$dS/S = \\mu\\,dt + \\sigma\\,dW + J\\,dN$\n(Geopolitical Shocks)', fontsize=11, fontweight='bold')
axes[1,0].set_xlabel('Time'); axes[1,0].set_ylabel('Price')
axes[1,0].grid(alpha=0.3)

# CIR for interest rates
a_cir, b_cir, sigma_cir = 0.5, 0.04, 0.1
for _ in range(N_paths):
    r_cir = np.zeros(N_steps+1); r_cir[0] = 0.03
    for i in range(1, N_steps+1):
        r_cir[i] = max(r_cir[i-1] + a_cir*(b_cir - r_cir[i-1])*dt_s +
                       sigma_cir*np.sqrt(max(r_cir[i-1],0)*dt_s)*np.random.randn(), 0)
    axes[1,1].plot(t_s, r_cir*100, lw=0.8, alpha=0.7)
axes[1,1].axhline(b_cir*100, color='k', lw=2, ls='--', label=f'Long-run mean {b_cir*100:.0f}%')
axes[1,1].set_title('Cox-Ingersoll-Ross (CIR)\n$dr = a(b-r)dt + \\sigma\\sqrt{r}\\,dW$\n(Interest Rates)', fontsize=11, fontweight='bold')
axes[1,1].set_xlabel('Time'); axes[1,1].set_ylabel('Rate (%)')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

# Fractional Brownian Motion (long memory, tidal modeling)
def fbm_path(H, N):
    from scipy.linalg import cholesky
    cov = np.array([[0.5*(abs(i)**( 2*H) + abs(j)**(2*H) - abs(i-j)**(2*H))
                     for j in range(N)] for i in range(N)])
    cov += 1e-8*np.eye(N)
    L = cholesky(cov, lower=True)
    return L @ np.random.randn(N)

t_fbm = np.linspace(0, 1, 200)
for H_val, color, label in [(0.3,'#e74c3c','H=0.3 (Anti-persist.)'),
                              (0.5,'#2ecc71','H=0.5 (BM)'),
                              (0.8,'#3498db','H=0.8 (Long Memory)')]:
    path = fbm_path(H_val, 200)
    axes[1,2].plot(t_fbm, path, color=color, lw=1.5, alpha=0.8, label=label)
axes[1,2].set_title('Fractional Brownian Motion\n(Hurst Exponent H — Tidal Memory)', fontsize=11, fontweight='bold')
axes[1,2].set_xlabel('Time'); axes[1,2].set_ylabel('fBm(t)')
axes[1,2].legend(fontsize=9); axes[1,2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig07_stochastic_processes.pdf', bbox_inches='tight')
plt.close()
print("Fig 7 done")
