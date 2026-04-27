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

plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'figure.dpi': 150})

# ============================================================
# FIGURE 1: Red Sea financial time series & GARCH volatility
# ============================================================
np.random.seed(42)
T = 1500
t = np.arange(T)

# Simulate oil price (Brent crude - Red Sea shipping corridor)
oil_drift = 0.0002
oil_vol_base = 0.018
vol_regime = np.ones(T)
vol_regime[300:500] = 2.5   # Houthi crisis spike
vol_regime[700:850] = 1.8   # geopolitical tension
vol_regime[1100:1200] = 2.0 # blockade events

returns = np.zeros(T)
prices = np.zeros(T)
prices[0] = 80.0
garch_vol = np.zeros(T)
garch_vol[0] = oil_vol_base
alpha, beta_g = 0.12, 0.82

for i in range(1, T):
    eps = np.random.randn()
    garch_vol[i] = np.sqrt(0.000015 + alpha * (returns[i-1]**2) + beta_g * garch_vol[i-1]**2)
    returns[i] = oil_drift + garch_vol[i] * vol_regime[i] * eps
    prices[i] = prices[i-1] * np.exp(returns[i])

fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor='white')
axes[0].plot(t, prices, color='#1a5276', lw=0.9, label='Brent Crude (USD/bbl)')
axes[0].axvspan(300, 500, alpha=0.15, color='red', label='Houthi Crisis')
axes[0].axvspan(700, 850, alpha=0.10, color='orange', label='Geopolitical Tension')
axes[0].set_ylabel('Price (USD/bbl)', fontsize=11)
axes[0].set_title('Red Sea Corridor: Brent Crude Oil Price Simulation with Regime Changes', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].plot(t, returns*100, color='#922b21', lw=0.6, alpha=0.8, label='Log-Returns (%)')
axes[1].axhline(0, color='k', lw=0.7, ls='--')
axes[1].set_ylabel('Return (%)', fontsize=11)
axes[1].set_title('Daily Log-Returns with Clustered Volatility', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

axes[2].plot(t, garch_vol*100*vol_regime, color='#1e8449', lw=1.0, label='GARCH(1,1) Conditional Vol')
axes[2].fill_between(t, 0, garch_vol*100*vol_regime, alpha=0.2, color='#1e8449')
axes[2].set_ylabel('Volatility (%)', fontsize=11)
axes[2].set_xlabel('Trading Days', fontsize=11)
axes[2].set_title('Conditional Volatility: GARCH(1,1) Estimates', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig01_redsea_timeseries.pdf', bbox_inches='tight')
plt.close()
print("Fig 1 done")
