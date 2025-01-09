# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:09:29 2025

@author: zhu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load WaterAbs data (assuming WaterAbs is loaded as a NumPy array)
# WaterAbs should be a 2D array where column 0 is wavelengths and column 1 is absorption
from scipy.io import loadmat

mat_data = loadmat('D:\\Light_sheet_code\WaterAbs.mat')
WaterAbs = mat_data['WaterAbs']  # Replace 'WaterAbs' with the actual key in the .mat file


# Wavelength range
lambda_vals = np.arange(900, 1201, 5)

# Interpolate WaterAbs data to match the wavelength range
WaterAbsi = []
for l in lambda_vals:
    idx = np.argmin(np.abs(WaterAbs[:, 0] - l))
    WaterAbsi.append(WaterAbs[idx, 1])
WaterAbsi = np.array(WaterAbsi)

# Plot WaterAbs data
plt.figure(1)
plt.plot(lambda_vals, WaterAbsi, 'o', linewidth=4, label='Interpolated Data')
plt.plot(WaterAbs[:, 0], WaterAbs[:, 1], '-', linewidth=4, label='Original Data')
plt.axvline(1030, color=[0.5, 0.5, 0.5], linewidth=4)
plt.axhline(WaterAbsi[np.where(lambda_vals == 1030)[0][0]], color=[0.5, 0.5, 0.5], linewidth=4)
plt.axis([900, 1150, 0, 0.05])
plt.xlabel('Wavelength in nm', fontsize=18, fontweight='bold')
plt.ylabel('Water absorption in a.u.', fontsize=18, fontweight='bold')
plt.grid(True, which='both')
plt.legend(fontsize=14)
plt.show()

# Calculate SL values
SL_1030 = 0.177
wa_1030 = WaterAbsi[np.where(lambda_vals == 1030)[0][0]]
SL = SL_1030 * WaterAbsi / wa_1030

# Plot SL values
plt.figure(2)
plt.plot(lambda_vals, 100 * SL, 'o-', linewidth=4)
plt.axvline(1030, color=[0.5, 0.5, 0.5], linewidth=4)
plt.axhline(100 * SL[np.where(lambda_vals == 1030)[0][0]], color=[0.5, 0.5, 0.5], linewidth=4)
plt.axis([900, 1150, 0, 40])
plt.xlabel('Wavelength in nm', fontsize=18, fontweight='bold')
plt.ylabel('SL in %', fontsize=18, fontweight='bold')
plt.grid(True, which='both')
plt.show()

# Calculate power limits
P_1030 = 70  # max power at 1030 nm
P = P_1030 / SL * SL_1030

# Plot power limits
plt.figure(3)
plt.plot(lambda_vals, P, 'o-', linewidth=4)
plt.axvline(1030, color=[0.5, 0.5, 0.5], linewidth=4)
plt.axhline(P[np.where(lambda_vals == 1030)[0][0]], color=[0.5, 0.5, 0.5], linewidth=4)
plt.axis([900, 1150, 0, 300])
plt.xlabel('Wavelength in nm', fontsize=18, fontweight='bold')
plt.ylabel('Max Power in mW (limited by heat)', fontsize=18, fontweight='bold')
plt.grid(True, which='both')
plt.show()

# Maioli paper constants
nNL = 5.8  # nonlinear photodamage order
P_0 = 70  # mW
PNL_0 = 1097  # mW
tau_0 = 300  # femtoseconds
nS = 2  # signal order
T_0 = 1 / 80

# Calculate optimal laser frequency
Plimit = P_1030 / SL * SL_1030

plt.figure(4)
for tau in [300, 210, 180]:
    if tau == 180:
        T_opt = tau / tau_0 * T_0 * (2.8 * Plimit / (PNL_0 * 1.4))**(-nNL / (nNL - 1))
    else:
        T_opt = tau / tau_0 * T_0 * (2.8 * Plimit / PNL_0)**(-nNL / (nNL - 1))
    plt.plot(lambda_vals, 1 / T_opt, 'o-', linewidth=4, label=f'\u03C4={tau}fs')

plt.axvline(1030, color=[0.5, 0.5, 0.5], linewidth=4)
plt.axhline(10, color=[0.5, 0.5, 0.5], linewidth=4)
plt.axis([900, 1150, 0, 80])
plt.xlabel('Wavelength in nm', fontsize=18, fontweight='bold')
plt.ylabel('Optimal laser frequency in MHz', fontsize=18, fontweight='bold')
plt.legend()
plt.grid(True, which='both')
plt.show()

# Load mCherry spectrum
# Table = pd.read_csv('FPbase_Spectra_mCherry.csv')
Table = loadmat('mCherryAbsi.mat')
# Ensure the column is treated as a string before applying string operations
mCherryAbsi = np.squeeze(Table["mCherryAbsi"])

# Signal enhancement
plt.figure(5)
tau = 300
T_opt = tau / tau_0 * T_0 * (2.8 * Plimit / PNL_0)**(-nNL / (nNL - 1))
SignalEnhancement = np.squeeze((T_opt / T_0) * (Plimit / P_0)**2 * (tau_0 / tau) * mCherryAbsi)

plt.plot(lambda_vals, SignalEnhancement, '-', linewidth=4)
plt.axvline(1030, color=[0.5, 0.5, 0.5], linewidth=4)
plt.axhline(SignalEnhancement[np.where(lambda_vals == 1030)[0][0]], color=[0.5, 0.5, 0.5], linewidth=4)
plt.axis([900, 1150, 0, 30])
plt.xlabel('Wavelength in nm', fontsize=18, fontweight='bold')
plt.ylabel('2PEF enhancement (mCherry)', fontsize=18, fontweight='bold')
plt.grid(True, which='both')
plt.show()


plt.figure(figsize=(6, 6))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
plt.plot(lambda_vals, mCherryAbsi, '-', linewidth=4)
plt.plot(lambda_vals, WaterAbsi, 'o', linewidth=4, label='Interpolated Data')

#%% Create a figure
# Plot mCherryAbsi on the primary y-axis

fig, ax1 = plt.subplots(figsize=(8, 5))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
ax1.plot(lambda_vals, mCherryAbsi, '-', linewidth=2, label='mCherry Absorption', color='red')
ax1.set_xlabel(r'Wavelength ($nm$)', fontsize=14)
ax1.set_ylabel('mCherry Absorption (a.u.)', fontsize=14, color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Add the secondary y-axis for WaterAbsi
ax2 = ax1.twinx()
ax2.plot(lambda_vals, WaterAbsi, '--', linewidth=2, label='Water Absorption', color='blue')
ax2.set_ylabel('Water Absorption (a.u.)', fontsize=14, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# # Optional: Add a legend
# fig.legend(loc='upper right', bbox_to_anchor=(0.4, 1.0), bbox_transform=ax1.transAxes)

# Show the plot
# plt.title('Absorption vs Wavelength')
plt.grid(True)
plt.xlim([900,1200])
plt.tight_layout()
plt.show()
