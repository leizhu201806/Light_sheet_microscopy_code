# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:27:46 2025

@author: zhu
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
ObjTrans = 0.67
nNL = 5.8  # Nonlinear photodamage order
P_0 = 70  # mW
PNL_0 = 1097  # mW
tau_0 = 300  # femtoseconds
nS = 2  # Signal order
T_0 = 1 / 80  # Reference period

# Time periods and frequencies
T = 1 / np.array([1, 2, 4, 8, 10, 16, 32, 80])  # µs
f = np.arange(1, 81)  # Frequency in MHz
T = 1 / f
tau1 = 300
tau2 = 210

# Compute PNL for two cases
PNL1 = (T_0 / T * tau1 / tau_0) ** (1 - 1 / nNL) * PNL_0
PNLlimit1 = PNL1 / 2.8

PNL2 = (T_0 / T * tau2 / tau_0) ** (1 - 1 / nNL) * PNL_0
PNLlimit2 = PNL2 / 2.8

# Figure 1
plt.figure(figsize=(8, 6))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
plt.plot(1 / T, PNL1 / ObjTrans, 'k-', linewidth=2, label=r'$\tau = 300\,$fs')
plt.plot(1 / T, PNL2 / ObjTrans, '--', color=[0.5, 0.5, 0.5], linewidth=2, label=r'$\tau = 210\,$fs')
plt.plot(1 / T, PNLlimit1 / ObjTrans, 'k:', linewidth=2)
plt.plot(1 / T, PNLlimit2 / ObjTrans, ':', color=[0.5, 0.5, 0.5], linewidth=2)


plt.ylabel(r'$P_{NL}$ in mW at back aperture', fontsize=14, weight='bold')
plt.xlabel('Laser frequency in MHz', fontsize=14, weight='bold')
plt.legend()
plt.grid(True)
plt.axis([0, 18, 0, 450])
plt.tight_layout()
plt.show()

#%% Figure 2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# # Constants
# T = 1 / np.arange(1, 81)
# PNL1 = (1 / T * 300 / 300) ** (1 - 1 / 5.8) * 1097

# Data points
dataf = np.array([4, 4, 8, 8, 8, 8, 16, 4, 8])
dataPNL = np.array([99.149236, 104.059584, 158.4624176, 172.01382, 233.2923077,
                    183.076944, 417.20618, 77.854764, 150.62])

k1 = [0, 2]
k2 = [1, 3, 5]
k3 = [4, 6]
k4 = [7]
k5 = [8]

# Plot
plt.figure(figsize=(7, 7))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2


plt.plot(1 / T, PNL1, 'k-.', linewidth=2, label=r'Maioli 2020 Scaling law @ 1030 $nm$')
plt.plot([1, 80], [115, 115], '--', linewidth=2, color=[0.5, 0.5, 0.5], label='Thermal effect threshold')

plt.plot(dataf[k1], dataPNL[k1], 'ko', linewidth=2, markersize=8, label= r'No splitting Satsuma @ 1030 $nm$')
plt.plot(dataf[k2], dataPNL[k2], 'o', color=[1, 0.5, 0], linewidth=2, markersize=8, label= r'2x splitting Satsuma @ 1030 $nm$ (20 $mm$ and 10$mm$)')
plt.plot(dataf[k3], dataPNL[k3], 'o', color=[0, 0, 1], linewidth=2, markersize=8, label=r'2x2x splitting Satsuma @ 1030 $nm$')
plt.plot(dataf[k4], dataPNL[k4], 'kx', linewidth=2, markersize=12, label=r'No splitting Opera @ 1070 $nm$')
plt.plot(dataf[k5], dataPNL[k5], 'x', color=[1, 0.5, 0], linewidth=2, markersize=12, label=r'2x splitting Satsuma @ 1070 $nm$')

plt.ylabel(r'$P_{NL}$ ($mW$) at sample', fontsize=14)
plt.xlabel(r'Repetition rate ($MHz$)', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.axis([3, 18, 60, 500])
# plt.xlim([3,18])
# Format ticks
plt.xticks([4, 8, 12, 16])

plt.yticks([100, 200, 300, 400, 500])
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ScalarFormatter())

plt.legend(loc='upper left', fontsize=10, frameon=False)
plt.tight_layout()
plt.show()
