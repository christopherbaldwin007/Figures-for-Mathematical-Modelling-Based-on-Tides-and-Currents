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
# FIGURE 11: Kalman Filter for tidal state estimation
# ============================================================
np.random.seed(77)
T_kf = 500
t_kf = np.linspace(0, 50, T_kf)
# True tidal state (sum of harmonics)
true_tide = (1.5*np.sin(2*np.pi*t_kf/12.42) +
             0.8*np.cos(2*np.pi*t_kf/24.0) +
             0.4*np.sin(2*np.pi*t_kf/8.0) +
             0.2*np.sin(2*np.pi*t_kf/6.0))
obs_noise = 0.3
observations = true_tide + obs_noise*np.random.randn(T_kf)

# Simple Kalman filter
def kalman_filter(obs, Q=0.01, R=0.09):
    n = len(obs)
    x_est = np.zeros(n); P = np.zeros(n)
    x_est[0] = obs[0]; P[0] = 1.0
    K_gains = np.zeros(n)
    for t in range(1, n):
        x_pred = x_est[t-1]
        P_pred = P[t-1] + Q
        K = P_pred / (P_pred + R)
        K_gains[t] = K
        x_est[t] = x_pred + K * (obs[t] - x_pred)
        P[t] = (1-K) * P_pred
    return x_est, P, K_gains

kf_est, kf_var, kf_gains = kalman_filter(observations)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor='white')
axes[0].plot(t_kf, true_tide, 'b-', lw=2, label='True Tidal State', zorder=3)
axes[0].plot(t_kf, observations, 'k.', ms=2, alpha=0.4, label='Noisy Observations')
axes[0].plot(t_kf, kf_est, 'r-', lw=1.5, label='Kalman Filter Estimate', zorder=4)
axes[0].fill_between(t_kf, kf_est - 2*np.sqrt(kf_var), kf_est + 2*np.sqrt(kf_var),
                      alpha=0.2, color='red', label='95% CI')
axes[0].set_title('Kalman Filter: Red Sea Tidal Height Estimation', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Tidal Height (m)', fontsize=11); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

axes[1].plot(t_kf, (kf_est - true_tide)*100, color='#922b21', lw=1, label='Estimation Error (cm)')
axes[1].fill_between(t_kf, -(2*np.sqrt(kf_var))*100, (2*np.sqrt(kf_var))*100, alpha=0.15, color='red')
axes[1].axhline(0, color='k', lw=1, ls='--')
axes[1].set_ylabel('Error (cm)', fontsize=11); axes[1].set_title('Kalman Filter Estimation Error', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

axes[2].plot(t_kf, kf_gains, color='#1a5276', lw=1.5, label='Kalman Gain K(t)')
axes[2].set_ylabel('Kalman Gain', fontsize=11); axes[2].set_xlabel('Time (hours)', fontsize=11)
axes[2].set_title('Kalman Gain Convergence', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig11_kalman.pdf', bbox_inches='tight')
plt.close()
print("Fig 11 done")
