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
# FIGURE 9: Red Sea wind field + shipping routes
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

lon = np.linspace(32, 44, 100)
lat = np.linspace(11, 29, 100)
LON, LAT = np.meshgrid(lon, lat)

# Seasonal wind (NW monsoon pattern)
U_wind = -8 * np.ones_like(LON) + 3*np.sin(2*np.pi*(LAT-11)/18) + np.random.randn(*LON.shape)*0.5
V_wind = -3 * np.ones_like(LAT) + 2*np.cos(2*np.pi*(LON-32)/12) + np.random.randn(*LAT.shape)*0.3
wind_speed = np.sqrt(U_wind**2 + V_wind**2)

c1 = axes[0].contourf(LON, LAT, wind_speed, levels=15, cmap='YlOrRd', alpha=0.8)
skip = 6
axes[0].quiver(LON[::skip, ::skip], LAT[::skip, ::skip],
               U_wind[::skip, ::skip], V_wind[::skip, ::skip],
               scale=100, color='navy', alpha=0.7, width=0.002)
plt.colorbar(c1, ax=axes[0], label='Wind Speed (knots)')

# Major Red Sea shipping routes
routes_data = [
    ([32.5, 33.5, 35.0, 37.0, 39.0, 40.5, 42.0, 43.5],
     [27.5, 26.5, 25.0, 23.0, 21.0, 18.5, 16.0, 14.0], '#e74c3c', 'Northbound VLCC', 3.0),
    ([43.0, 41.5, 40.0, 38.5, 37.0, 35.5, 33.5, 32.0],
     [14.5, 16.0, 18.0, 20.0, 22.0, 24.0, 26.5, 27.8], '#3498db', 'Southbound Container', 2.5),
    ([43.5, 42.0, 40.5, 39.0, 37.5, 36.0, 34.5, 33.0],
     [12.5, 13.5, 15.0, 17.0, 19.5, 22.0, 24.5, 26.5], '#2ecc71', 'LNG Route', 2.0),
]
for lons, lats, color, label, lw in routes_data:
    axes[0].plot(lons, lats, color=color, lw=lw, label=label, zorder=5)
    axes[0].plot(lons[-1], lats[-1], 'o', color=color, ms=8, zorder=6)

key_ports = {'Suez Canal': (32.5, 30.0), 'Jeddah': (39.2, 21.5),
             'Aden': (45.0, 12.8), 'Hodeidah': (42.9, 14.8),
             "Bab-el-Mandeb": (43.4, 12.6)}
for port, (plon, plat) in key_ports.items():
    axes[0].plot(plon, plat, 'k*', ms=12, zorder=7)
    axes[0].annotate(port, (plon, plat), textcoords='offset points', xytext=(5, 5),
                     fontsize=7, fontweight='bold')

axes[0].set_xlabel('Longitude (°E)', fontsize=11)
axes[0].set_ylabel('Latitude (°N)', fontsize=11)
axes[0].set_title('Red Sea Wind Field & Shipping Routes\n(NW Monsoon Season)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8, loc='upper right')

# Monthly wind roses (polar)
months = ['Jan','Apr','Jul','Oct']
ax_polar = axes[1]
ax_polar.axis('off')

gs_inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=axes[1].get_subplotspec(),
                                             hspace=0.4, wspace=0.4)
for idx, (month, seed_v) in enumerate(zip(months, [1, 2, 3, 4])):
    ax_p = fig.add_subplot(gs_inner[idx//2, idx%2], projection='polar')
    np.random.seed(seed_v*10)
    directions = np.random.vonmises(np.pi*(1 + 0.5*idx/4), 2.0, 500) % (2*np.pi)
    speeds = np.random.weibull(2.0, 500) * (10 + 3*idx)
    bins = np.linspace(0, 2*np.pi, 17)
    freq, _ = np.histogram(directions, bins=bins)
    bars = ax_p.bar(bins[:-1], freq/freq.max(), width=2*np.pi/16, alpha=0.7,
                     color=plt.cm.RdYlBu_r(idx/4))
    ax_p.set_title(month, fontsize=9, fontweight='bold', pad=2)
    ax_p.set_yticklabels([])

plt.suptitle('Red Sea Seasonal Wind Roses\n(Directional Frequency)', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figures/fig09_redsea_wind.pdf', bbox_inches='tight')
plt.close()
print("Fig 9 done")
