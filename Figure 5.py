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
# FIGURE 5: Futures curves (contango/backwardation)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

maturities = np.arange(1, 25)

# Oil futures (contango → backwardation)
scenarios = {
    'Deep Contango\n(Oversupply)': (60, 0.8, 'Contango'),
    'Mild Contango': (80, 0.3, 'Contango'),
    'Backwardation\n(Supply Squeeze)': (95, -0.5, 'Backwardation'),
    'Severe Backwardation\n(Red Sea Crisis)': (110, -1.2, 'Backwardation'),
}

colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
for (name, (F0, slope, regime)), color in zip(scenarios.items(), colors):
    curve = F0 + slope * np.sqrt(maturities) * np.sign(slope)
    axes[0,0].plot(maturities, curve, 'o-', color=color, lw=2, ms=4, label=f'{name}')

axes[0,0].set_xlabel('Maturity (months)', fontsize=11)
axes[0,0].set_ylabel('Futures Price (USD/bbl)', fontsize=11)
axes[0,0].set_title('Brent Crude Futures Curves\nContango vs Backwardation', fontsize=12, fontweight='bold')
axes[0,0].legend(fontsize=8)
axes[0,0].grid(alpha=0.3)

# 3D futures surface: price vs time vs maturity
T_hist = np.linspace(0, 2, 50)
M_arr = np.arange(1, 13)
TH, MH = np.meshgrid(T_hist, M_arr)
shock_factor = 1 + 0.3*np.exp(-((TH-0.8)**2)/0.05)  # crisis spike
F_surface = (80 + 5*np.sin(2*np.pi*TH) - 2*np.log(MH)) * shock_factor

ax3d = fig.add_subplot(222, projection='3d')
surf = ax3d.plot_surface(TH, MH, F_surface, cmap='RdYlGn_r', alpha=0.9, linewidth=0)
ax3d.set_xlabel('Calendar Time (yrs)', fontsize=8)
ax3d.set_ylabel('Maturity (months)', fontsize=8)
ax3d.set_zlabel('Futures Price', fontsize=8)
ax3d.set_title('3D Oil Futures Surface\nCalendar × Maturity Evolution', fontsize=11, fontweight='bold')
fig.colorbar(surf, ax=ax3d, shrink=0.4, label='Price')

# Shipping freight futures (Baltic Exchange)
freight_mats = np.arange(1, 13)
routes = {
    'VLCC Middle East→China': (40000, 0.05),
    'Suez Canal Detour (Cape)': (65000, 0.08),
    'Red Sea Direct': (35000, -0.03),
    'LNG Spot': (80000, 0.12),
}
colors2 = ['#1a5276','#7d6608','#922b21','#0e6251']
for (route, (base, slope)), color in zip(routes.items(), colors2):
    curve = base * (1 + slope * np.sqrt(freight_mats))
    axes[1,0].plot(freight_mats, curve, 's-', color=color, lw=2, ms=5, label=route)
axes[1,0].set_xlabel('Maturity (months)', fontsize=11)
axes[1,0].set_ylabel('Freight Rate (USD/day)', fontsize=11)
axes[1,0].set_title('Shipping Freight Futures Curves\nRed Sea vs Alternative Routes', fontsize=12, fontweight='bold')
axes[1,0].legend(fontsize=8)
axes[1,0].grid(alpha=0.3)
axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))

# VIX-like volatility futures
vix_mats = np.arange(1, 9)
vix_normal = 15 + 2*np.log(vix_mats)
vix_stressed = 35 - 5*np.sqrt(vix_mats)
vix_redsea = 28 - 3.5*np.sqrt(vix_mats)
axes[1,1].plot(vix_mats, vix_normal, 'g-o', lw=2, ms=6, label='Normal Markets')
axes[1,1].plot(vix_mats, vix_stressed, 'r-s', lw=2, ms=6, label='Global Crisis')
axes[1,1].plot(vix_mats, vix_redsea, 'b-^', lw=2, ms=6, label='Red Sea Crisis 2024')
axes[1,1].fill_between(vix_mats, vix_normal, vix_redsea, alpha=0.15, color='blue', label='Crisis Premium')
axes[1,1].set_xlabel('Maturity (months)', fontsize=11)
axes[1,1].set_ylabel('Volatility Index Level', fontsize=11)
axes[1,1].set_title('Volatility Futures Term Structure\n(VIX-Style Geopolitical Risk)', fontsize=12, fontweight='bold')
axes[1,1].legend(fontsize=9)
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig05_futures_curves.pdf', bbox_inches='tight')
plt.close()
print("Fig 5 done")
