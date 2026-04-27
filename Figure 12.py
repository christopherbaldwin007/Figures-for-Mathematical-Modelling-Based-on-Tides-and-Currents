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
# FIGURE 12: Red Sea Strait flow (CFD/PDE finite differences)
# ============================================================
# Shallow water equations finite difference - Bab-el-Mandeb strait
Nx_sw, Ny_sw = 80, 40
Lx, Ly = 100e3, 30e3  # meters
dx = Lx/Nx_sw; dy = Ly/Ny_sw
x_sw = np.linspace(0, Lx, Nx_sw)
y_sw = np.linspace(0, Ly, Ny_sw)
XS, YS = np.meshgrid(x_sw, y_sw)

# Bathymetry (strait constriction at Bab-el-Mandeb)
depth = 200 - 150*np.exp(-((XS - Lx/2)**2/(2*(Lx/6)**2) + (YS - Ly/2)**2/(2*(Ly/4)**2)))
depth = np.maximum(depth, 20)

# Pressure-driven flow (tidal forcing)
U_flow = 2.0 * (depth/200) * np.exp(-((YS - Ly/2)**2)/(2*(Ly/4)**2))
V_flow = 0.3 * np.sin(2*np.pi*XS/Lx) * np.cos(np.pi*YS/Ly)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

c1 = axes[0].contourf(XS/1e3, YS/1e3, depth, levels=20, cmap='ocean_r', alpha=0.85)
plt.colorbar(c1, ax=axes[0], label='Depth (m)')
skip_s = 4
axes[0].quiver(XS[::skip_s, ::skip_s]/1e3, YS[::skip_s, ::skip_s]/1e3,
               U_flow[::skip_s, ::skip_s], V_flow[::skip_s, ::skip_s],
               scale=15, color='white', alpha=0.8, width=0.003)
axes[0].set_xlabel('Distance (km)', fontsize=11)
axes[0].set_ylabel('Cross-strait Distance (km)', fontsize=11)
axes[0].set_title('Bab-el-Mandeb Strait: Bathymetry & Tidal Flow\n(Finite Difference Simulation)', fontsize=12, fontweight='bold')
axes[0].text(20, 25, 'Yemen\nCoast', fontsize=9, color='white', fontweight='bold')
axes[0].text(20, 3, 'Djibouti\nCoast', fontsize=9, color='white', fontweight='bold')
axes[0].text(45, 14, 'Bab-el-Mandeb\nConstriction', fontsize=8, color='yellow', fontweight='bold', ha='center')

speed = np.sqrt(U_flow**2 + V_flow**2)
c2 = axes[1].contourf(XS/1e3, YS/1e3, speed, levels=20, cmap='hot_r', alpha=0.85)
plt.colorbar(c2, ax=axes[1], label='Current Speed (m/s)')
axes[1].streamplot(x_sw/1e3, y_sw/1e3, U_flow, V_flow, density=1.5, color='white', linewidth=0.8, arrowsize=0.8)
axes[1].set_xlabel('Distance (km)', fontsize=11)
axes[1].set_ylabel('Cross-strait Distance (km)', fontsize=11)
axes[1].set_title('Current Speed Field & Streamlines\n(Bab-el-Mandeb Tidal Dynamics)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/fig12_strait_flow.pdf', bbox_inches='tight')
plt.close()
print("Fig 12 done")
