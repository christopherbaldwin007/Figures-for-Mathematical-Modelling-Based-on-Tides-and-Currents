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
import os
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True)

# ============================================================
# FIGURE 2: 3D Black-Scholes option surface
# ============================================================
from scipy.stats import norm

def bs_call(S, K, T_exp, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_exp) / (sigma*np.sqrt(T_exp) + 1e-9)
    d2 = d1 - sigma*np.sqrt(T_exp)
    return S * norm.cdf(d1) - K * np.exp(-r*T_exp) * norm.cdf(d2)

def bs_delta(S, K, T_exp, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T_exp) / (sigma*np.sqrt(T_exp) + 1e-9)
    return norm.cdf(d1)

S = 100; r = 0.05
K_arr = np.linspace(70, 130, 60)
T_arr = np.linspace(0.05, 2.0, 60)
KK, TT = np.meshgrid(K_arr, T_arr)
sigma_surface = 0.20 + 0.15 * np.exp(-((KK-100)**2)/800) + 0.05*(1/TT)

price_surface = bs_call(S, KK, TT, r, sigma_surface)
delta_surface = bs_delta(S, KK, TT, r, sigma_surface)

fig = plt.figure(figsize=(18, 7), facecolor='white')

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(KK, TT, price_surface, cmap='plasma', alpha=0.9, linewidth=0, antialiased=True)
ax1.set_xlabel('Strike K', fontsize=10)
ax1.set_ylabel('Maturity T (yrs)', fontsize=10)
ax1.set_zlabel('Option Price (USD)', fontsize=10)
ax1.set_title('3D Black-Scholes Call Price Surface\n(Implied Vol Smile Incorporated)', fontsize=11, fontweight='bold')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, label='Price')

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(KK, TT, delta_surface, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
ax2.set_xlabel('Strike K', fontsize=10)
ax2.set_ylabel('Maturity T (yrs)', fontsize=10)
ax2.set_zlabel('Delta', fontsize=10)
ax2.set_title('3D Delta Surface\n(Sensitivity to Underlying)', fontsize=11, fontweight='bold')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, label='Delta')

plt.tight_layout()
plt.savefig('figures/fig02_bs_surface.pdf', bbox_inches='tight')
plt.close()
print("Fig 2 done")
