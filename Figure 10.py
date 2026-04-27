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
# FIGURE 10: Monte Carlo VaR / CVaR + tail risk
# ============================================================
np.random.seed(99)
N_mc = 50000
# Portfolio: oil futures (long), freight (short), FX USD/SAR
port_returns = (0.6 * np.random.normal(0.0005, 0.018, N_mc) +
                0.3 * np.random.standard_t(4, N_mc) * 0.015 +
                0.1 * np.random.normal(0.0001, 0.003, N_mc))

# Stress scenarios (Red Sea crisis)
crisis_returns = port_returns.copy()
crisis_idx = np.random.choice(N_mc, 500, replace=False)
crisis_returns[crisis_idx] -= 0.05 + np.random.exponential(0.02, 500)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')

# Distribution with VaR/CVaR
axes[0].hist(port_returns*100, bins=200, density=True, color='#3498db', alpha=0.7, label='Normal Conditions')
axes[0].hist(crisis_returns*100, bins=200, density=True, color='#e74c3c', alpha=0.4, label='Red Sea Crisis')
var_95 = np.percentile(crisis_returns, 5) * 100
cvar_95 = crisis_returns[crisis_returns <= np.percentile(crisis_returns, 5)].mean() * 100
axes[0].axvline(var_95, color='darkred', lw=2.5, ls='--', label=f'VaR(95%) = {var_95:.2f}%')
axes[0].axvline(cvar_95, color='black', lw=2.5, ls=':', label=f'CVaR(95%) = {cvar_95:.2f}%')
axes[0].set_xlabel('Daily Return (%)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Portfolio Return Distribution\nVaR & CVaR Risk Metrics', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# VaR surface over confidence levels and horizons
conf_levels = np.linspace(0.90, 0.99, 40)
horizons = np.arange(1, 22)
CL, HZ = np.meshgrid(conf_levels, horizons)
VaR_surface = np.array([[np.percentile(crisis_returns, (1-cl)*100) * np.sqrt(h) * 100
                          for cl in conf_levels] for h in horizons])

ax3 = fig.add_subplot(132, projection='3d')
surf = ax3.plot_surface(CL*100, HZ, -VaR_surface, cmap='Reds', alpha=0.9, linewidth=0)
ax3.set_xlabel('Confidence %', fontsize=8)
ax3.set_ylabel('Horizon (days)', fontsize=8)
ax3.set_zlabel('VaR (%)', fontsize=8)
ax3.set_title('3D VaR Surface\nConfidence × Horizon', fontsize=11, fontweight='bold')
fig.colorbar(surf, ax=ax3, shrink=0.4)

# Expected Shortfall time series (rolling)
T_roll = 500
daily_rets = np.random.normal(0.0003, 0.015, T_roll)
shock_days = [150, 200, 250, 300]
for sd in shock_days:
    daily_rets[sd:sd+10] -= np.random.exponential(0.03, 10)
window = 60
es_ts = []
for i in range(window, T_roll):
    w = daily_rets[i-window:i]
    q5 = np.percentile(w, 5)
    es_ts.append(w[w <= q5].mean() * 100)

axes[2].plot(range(window, T_roll), es_ts, color='#922b21', lw=1.5, label='60d Rolling CVaR')
axes[2].fill_between(range(window, T_roll), es_ts, alpha=0.2, color='#922b21')
for sd in shock_days:
    axes[2].axvline(sd, color='orange', lw=1.5, ls='--', alpha=0.7)
axes[2].axvline(shock_days[0], color='orange', lw=1.5, ls='--', alpha=0.7, label='Shock Events')
axes[2].set_xlabel('Trading Days', fontsize=11)
axes[2].set_ylabel('Expected Shortfall (%)', fontsize=11)
axes[2].set_title('Rolling Expected Shortfall\n(Red Sea Shock Events)', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig10_var_cvar.pdf', bbox_inches='tight')
plt.close()
print("Fig 10 done")
