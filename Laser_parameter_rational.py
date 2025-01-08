# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:23:52 2025

@author: zhu
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
ObjectiveTrans = 0.65

# Maioli paper parameters
nNL = 5.8  # nonlinear photodamage order
P_0 = 70  # mW
PNL_0 = 1097  # mW
tau_0 = 300  # femtoseconds
nS = 2  # signal order
T_0 = 1 / 80  # reference time in microseconds

# Laser frequencies (in MHz)
f = np.arange(2, 25, 2)  # 1 to 16 MHz
T = 1 / f  # Corresponding periods

def calculate_pnl(T, tau):
    return (T_0 / T * tau / tau_0)**(1 - 1 / nNL) * PNL_0

# Calculate PNL for different pulse durations
tau1 = 300  # femtoseconds
PNL1 = calculate_pnl(T, tau1)
PNLlimit1 = PNL1 / 2.7

tau2 = 180  # femtoseconds
PNL2 = calculate_pnl(T, tau2)
PNLlimit2 = PNL2 / 2.7

# Plotting
# Create a violin plot with the reordered groups
plt.figure(figsize=(6, 6))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

# Figure 1: PNL at back aperture

plt.plot(1 / T, PNL1 / ObjectiveTrans, 'ko-', linewidth=2, label=r'$\tau =300fs$')
plt.plot(1 / T, PNL2 / ObjectiveTrans, 'ro-', linewidth=2, label=r'$\tau=180fs$')
plt.plot(1 / T, PNLlimit1 / ObjectiveTrans, 'k:', linewidth=2, label=r'Limit $\tau =300fs$')
plt.plot(1 / T, PNLlimit2 / ObjectiveTrans, 'r:', linewidth=2, label=r'Limit $\tau =180fs$')
plt.xlabel('Repetition rate (MHz)', fontsize=14)
plt.ylabel(r'$P_{NL}$ (mW) at back aperture (65% transmission)', fontsize=14)
plt.legend()
plt.grid(True)
# plt.title('PNL at Back Aperture', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Figure 2: PNL at sample
# Create a violin plot with the reordered groups
plt.figure(figsize=(6, 6))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
plt.plot(1 / T, PNL1, 'ko-', linewidth=2, label=r'$\tau =300fs$')
plt.plot(1 / T, PNL2, 'ro-', linewidth=2,  label=r'$\tau=180fs$')
plt.plot(1 / T, PNLlimit1, 'k:', linewidth=2, label=r'Limit $\tau =300fs$')
plt.plot(1 / T, PNLlimit2, 'r:', linewidth=2, label=r'Limit $\tau =180fs$')
plt.axhline(y=70, color='gray', linestyle='--', linewidth=3, label='70 mW threshold')
plt.xlabel(r'Repetition rate ($MHz$)', fontsize=14)
plt.ylabel(r'$P_{NL}$ ($mW$) at sample', fontsize=14)
plt.legend()
plt.grid(True)
# plt.title('PNL at Sample', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

# Array calculations
A = np.array([
    [92, 68, 60],
    [163, 121, 107],
    [196, 146, 129],
    [290, 216, 190]
])

A_adjusted = A / ObjectiveTrans
print("Adjusted Array:\n", A_adjusted)
