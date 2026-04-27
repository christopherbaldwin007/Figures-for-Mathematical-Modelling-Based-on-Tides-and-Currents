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
# FIGURE 4: Red Sea tidal / current data 3D
# ============================================================
x = np.linspace(32, 43, 80)   # longitude (Red Sea)
y = np.linspace(12, 28, 80)   # latitude
XX, YY = np.meshgrid(x, y)

# Tidal height model
tide_height = (1.2 * np.sin(2*np.pi*(XX-32)/4 + YY*0.3) +
               0.6 * np.cos(2*np.pi*(YY-12)/8) +
               0.4 * np.sin(2*np.pi*(XX+YY)/10) +
               0.3 * np.exp(-((XX-37)**2 + (YY-21)**2)/20))

fig = plt.figure(figsize=(18, 7), facecolor='white')

ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(XX, YY, tide_height, cmap='ocean', alpha=0.9, linewidth=0)
ax1.set_xlabel('Longitude (°E)', fontsize=9)
ax1.set_ylabel('Latitude (°N)', fontsize=9)
ax1.set_zlabel('Tidal Height (m)', fontsize=9)
ax1.set_title('Red Sea Tidal Height Model\n(3D Spatial Distribution)', fontsize=11, fontweight='bold')
fig.colorbar(surf, ax=ax1, shrink=0.5, label='Height (m)')

ax2 = fig.add_subplot(122)
c = ax2.contourf(XX, YY, tide_height, levels=20, cmap='ocean')
# Wind/current vectors
x_q = x[::8]; y_q = y[::8]
XQ, YQ = np.meshgrid(x_q, y_q)
U = -0.5 * np.sin(2*np.pi*(XQ-32)/6)  # zonal current
V = 0.3 * np.cos(2*np.pi*(YQ-12)/10)  # meridional current
ax2.quiver(XQ, YQ, U, V, scale=8, color='white', alpha=0.8, width=0.003)
plt.colorbar(c, ax=ax2, label='Tidal Height (m)')
ax2.set_xlabel('Longitude (°E)', fontsize=11)
ax2.set_ylabel('Latitude (°N)', fontsize=11)
ax2.set_title('Red Sea Current Vectors & Tidal Map', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig04_redsea_tidal.pdf', bbox_inches='tight')
plt.close()
print("Fig 4 done")
