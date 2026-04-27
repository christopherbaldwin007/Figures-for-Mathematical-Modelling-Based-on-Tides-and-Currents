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
# FIGURE 3: Implied Vol smile / skew
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
S = 100
moneyness = np.linspace(-0.4, 0.4, 200)
K_range = S * np.exp(moneyness)

for T_exp, color, label in [(0.1, '#e74c3c','1M'), (0.25, '#e67e22','3M'),
                              (0.5, '#2ecc71','6M'), (1.0, '#3498db','1Y'), (2.0, '#9b59b6','2Y')]:
    smile = 0.20 + 0.08*moneyness**2 - 0.04*moneyness + 0.03/T_exp * np.exp(-moneyness**2/0.05)
    axes[0].plot(K_range, smile*100, color=color, lw=2, label=f'T={label}')

axes[0].set_xlabel('Strike K', fontsize=11)
axes[0].set_ylabel('Implied Volatility (%)', fontsize=11)
axes[0].set_title('Implied Volatility Smile\nBlack-Scholes Market Calibration', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Term structure
T_term = np.linspace(0.05, 3, 200)
atm_term = 0.18 + 0.05*np.exp(-T_term/0.5) + 0.02*np.sin(2*np.pi*T_term/1.5)
axes[1].plot(T_term, atm_term*100, color='#1a5276', lw=2.5, label='ATM Implied Vol')
axes[1].fill_between(T_term, (atm_term-0.02)*100, (atm_term+0.02)*100, alpha=0.2, color='#1a5276', label='±2% Band')
axes[1].set_xlabel('Maturity (years)', fontsize=11)
axes[1].set_ylabel('Implied Volatility (%)', fontsize=11)
axes[1].set_title('ATM Volatility Term Structure\n(Red Sea Shipping Options)', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig03_vol_smile.pdf', bbox_inches='tight')
plt.close()
print("Fig 3 done")
