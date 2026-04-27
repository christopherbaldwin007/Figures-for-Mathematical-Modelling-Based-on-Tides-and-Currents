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
# FIGURE 6: PDE heat map (BS equation numerical solution)
# ============================================================
# Crank-Nicolson for Black-Scholes PDE
Nx, Nt = 200, 500
S_max = 250; K = 100; r = 0.05; sigma = 0.25; T_exp = 1.0
dS = S_max / Nx; dt = T_exp / Nt
S_arr = np.linspace(0, S_max, Nx+1)

V = np.maximum(S_arr - K, 0)  # initial condition (payoff)
V_history = [V.copy()]

# Tridiagonal solve
def crank_nicolson_bs(V, S_arr, dt, dS, r, sigma, n_steps=50):
    N = len(S_arr) - 1
    results = [V.copy()]
    for _ in range(n_steps):
        a = 0.25*dt*(sigma**2 * np.arange(N+1)**2 - r*np.arange(N+1))
        b = -0.5*dt*(sigma**2 * np.arange(N+1)**2 + r)
        c = 0.25*dt*(sigma**2 * np.arange(N+1)**2 + r*np.arange(N+1))
        A = np.diag(1-b[1:-1]) + np.diag(-a[2:-1], -1) + np.diag(-c[1:-2], 1)
        rhs = (b[1:-1]+1)*V[1:-1] + a[1:-1]*V[:-2] + c[1:-1]*V[2:]
        V[1:-1] = np.linalg.solve(A, rhs)
        V[0] = 0
        V[-1] = S_arr[-1] - K*np.exp(-r*dt)
        results.append(V.copy())
    return results

V_steps = crank_nicolson_bs(V.copy(), S_arr, dt, dS, r, sigma, n_steps=Nt)
V_matrix = np.array(V_steps)

t_fig = np.linspace(0, T_exp, V_matrix.shape[0])

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

# Heatmap
im = axes[0].imshow(V_matrix.T, aspect='auto', cmap='inferno', origin='lower',
                     extent=[0, T_exp, 0, S_max])
plt.colorbar(im, ax=axes[0], label='Option Value (USD)')
axes[0].axhline(K, color='white', lw=1.5, ls='--', label=f'Strike K={K}')
axes[0].set_xlabel('Time to Maturity (yrs)', fontsize=11)
axes[0].set_ylabel('Asset Price S (USD)', fontsize=11)
axes[0].set_title('Black-Scholes PDE Solution\nCrank-Nicolson Heat Map', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)

# Cross sections
for idx, color, label in [(0,'#e74c3c','t=0 (Payoff)'), (20,'#e67e22','t=T/4'),
                           (50,'#2ecc71','t=T/2'), (79,'#3498db','t≈T (Today)')]:
    axes[1].plot(S_arr, V_steps[idx], color=color, lw=2, label=label)
axes[1].axvline(K, color='k', ls='--', lw=1, alpha=0.5)
axes[1].set_xlabel('Asset Price S (USD)', fontsize=11)
axes[1].set_ylabel('Option Value V(S,t)', fontsize=11)
axes[1].set_title('PDE Cross-Sections\n(Evolution of Option Value)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# Greeks from PDE
delta_pde = np.gradient(V_steps[79], dS)
gamma_pde = np.gradient(delta_pde, dS)
ax_twin = axes[2].twinx()
axes[2].plot(S_arr[5:-5], delta_pde[5:-5], color='#3498db', lw=2.5, label='Delta (Δ)')
ax_twin.plot(S_arr[5:-5], gamma_pde[5:-5], color='#e74c3c', lw=2.5, ls='--', label='Gamma (Γ)')
axes[2].set_xlabel('Asset Price S (USD)', fontsize=11)
axes[2].set_ylabel('Delta', color='#3498db', fontsize=11)
ax_twin.set_ylabel('Gamma', color='#e74c3c', fontsize=11)
axes[2].set_title('Greeks from PDE Numerical Solution\n(Delta & Gamma)', fontsize=12, fontweight='bold')
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax_twin.get_legend_handles_labels()
axes[2].legend(lines1+lines2, labels1+labels2, fontsize=9)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig06_pde_solution.pdf', bbox_inches='tight')
plt.close()
print("Fig 6 done")
