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
# FIGURE 13: Yield curve / interest rate futures
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
# Different yield curve regimes
curves = {
    'Normal (Pre-2022)': [1.5, 1.7, 2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.4, 3.5, 3.6],
    'Inverted (2023)':   [5.4, 5.5, 5.5, 5.3, 5.1, 4.8, 4.6, 4.4, 4.3, 4.2, 4.1],
    'Flat':              [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    'Red Sea Risk-Off':  [5.2, 5.4, 5.6, 5.3, 5.0, 4.7, 4.5, 4.3, 4.2, 4.1, 4.0],
}
colors_yc = ['#2ecc71','#e74c3c','#f39c12','#3498db']
for (name, rates), color in zip(curves.items(), colors_yc):
    axes[0,0].plot(mats, rates, 'o-', color=color, lw=2.5, ms=6, label=name)
axes[0,0].set_xlabel('Maturity (years)', fontsize=11)
axes[0,0].set_ylabel('Yield (%)', fontsize=11)
axes[0,0].set_title('US Treasury Yield Curves\n(Multiple Market Regimes)', fontsize=12, fontweight='bold')
axes[0,0].legend(fontsize=9); axes[0,0].grid(alpha=0.3)

# Nelson-Siegel fit
def nelson_siegel(tau, beta0, beta1, beta2, lambda_ns):
    factor = (1 - np.exp(-tau/lambda_ns)) / (tau/lambda_ns)
    return beta0 + beta1*factor + beta2*(factor - np.exp(-tau/lambda_ns))

tau_ns = np.linspace(0.25, 30, 200)
tau_pts = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
rates_pts = np.array([5.4, 5.5, 5.5, 5.3, 5.1, 4.8, 4.6, 4.4, 4.2, 4.1])

from scipy.optimize import curve_fit
def ns_fit(tau, b0, b1, b2, lam):
    return nelson_siegel(tau, b0, b1, b2, lam)

try:
    popt, _ = curve_fit(ns_fit, tau_pts, rates_pts, p0=[4.5, -2, 1, 2], maxfev=5000)
    ns_fitted = nelson_siegel(tau_ns, *popt)
    axes[0,1].plot(tau_pts, rates_pts, 'ko', ms=8, zorder=5, label='Market Yields')
    axes[0,1].plot(tau_ns, ns_fitted, 'r-', lw=2.5, label=f'Nelson-Siegel Fit\n(β₀={popt[0]:.2f}, β₁={popt[1]:.2f})')
except:
    axes[0,1].plot(tau_pts, rates_pts, 'ko', ms=8)

axes[0,1].set_xlabel('Maturity (years)', fontsize=11)
axes[0,1].set_ylabel('Yield (%)', fontsize=11)
axes[0,1].set_title('Nelson-Siegel Yield Curve Fitting', fontsize=12, fontweight='bold')
axes[0,1].legend(fontsize=9); axes[0,1].grid(alpha=0.3)

# 3D yield curve evolution
T_evolve = np.linspace(0, 2, 30)
mats_3d = np.linspace(0.25, 30, 50)
TT3, MM3 = np.meshgrid(T_evolve, mats_3d)
shock_t = 0.8
yield_surface_3d = (4.5 - 1.5*np.exp(-MM3/5) +
                    0.8*np.exp(-((TT3-shock_t)**2)/0.05) * np.exp(-MM3/10) +
                    0.3*np.sin(2*np.pi*TT3) * np.exp(-MM3/20))

ax_3d = fig.add_subplot(223, projection='3d')
surf = ax_3d.plot_surface(TT3, MM3, yield_surface_3d, cmap='RdYlGn_r', alpha=0.9, linewidth=0)
ax_3d.set_xlabel('Calendar Time (yrs)', fontsize=8)
ax_3d.set_ylabel('Maturity (yrs)', fontsize=8)
ax_3d.set_zlabel('Yield (%)', fontsize=8)
ax_3d.set_title('3D Yield Surface Evolution\n(Including Geopolitical Shock)', fontsize=11, fontweight='bold')
fig.colorbar(surf, ax=ax_3d, shrink=0.4)

# Forward rate agreement pricing
f_rates = np.diff(np.array([5.4, 5.5, 5.5, 5.3, 5.1, 4.8, 4.6, 4.4, 4.2, 4.1]) *
                   np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])) / np.diff(np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]))
mat_fwd = (np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20]) + np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30]))/2

axes[1,1].bar(mat_fwd, f_rates, width=np.diff(np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]))*0.8,
              color=['#e74c3c' if r > 5.5 else '#3498db' for r in f_rates], alpha=0.8, label='Forward Rates')
axes[1,1].axhline(np.mean(f_rates), color='k', lw=2, ls='--', label=f'Mean = {np.mean(f_rates):.2f}%')
axes[1,1].set_xlabel('Maturity (years)', fontsize=11)
axes[1,1].set_ylabel('Forward Rate (%)', fontsize=11)
axes[1,1].set_title('Implied Forward Rates\n(Bootstrapped from Yield Curve)', fontsize=12, fontweight='bold')
axes[1,1].legend(fontsize=9); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig13_yield_curves.pdf', bbox_inches='tight')
plt.close()
print("Fig 13 done")
