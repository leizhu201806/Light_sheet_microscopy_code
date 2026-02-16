# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:22:54 2024

@author: zhu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Function to process each file and calculate ΔF/F
def process_file(file_path):
    # Load data from the CSV file
    data = pd.read_csv(file_path)
    # time = data.iloc[:, 0].values  # First column as time (adjust depending on the file structure)
    F_raw = data.iloc[:, 1].values  # Second column as fluorescence data
    # F_raw = medfilt(F_raw, kernel_size=51)
    # Parameters for the sliding window
    window_size = 4000  # Size of the sliding window (number of points)
    half_window = window_size // 2  # Half of the window size for centering
    delta_F_F = np.zeros_like(F_raw)  # Pre-allocate delta F/F
    delta_F_F = np.float32(delta_F_F)
    # Calculate ΔF/F using a sliding baseline window
    for i in range(len(F_raw)):
        # Define the indices for the current window
        start_idx = max(0, i - half_window)  # Ensure index is not less than 0
        end_idx = min(len(F_raw), i + half_window)  # Ensure index does not exceed length of data

        # Calculate the baseline as the 10th percentile of fluorescence in the current window
        F_baseline = np.percentile(F_raw[start_idx:end_idx], 10)
               
        D = np.floor(F_baseline)-1
        # D =  np.nanmin(F_raw[start_idx:end_idx]) - 2
        # Calculate ΔF/F for the current time point
        delta_F_F[i] = (F_raw[i] - F_baseline) / (F_baseline - D)

    # Apply median filter to the ΔF/F data
    # delta_F_F = medfilt(delta_F_F, kernel_size=201)  # Adjust the window size for the median filter
    # delta_F_F = medfilt(delta_F_F, kernel_size=101)  # Adjust the window size for the median filter
    
    return delta_F_F


# Configure matplotlib settings
plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

# Define file paths for three CSV files
# file_path = r'C:\Users\zhu\Desktop\Figure\Neurons Activity\original data'
file_path = r'C:\Users\zhu\Desktop\Figure\Neurons Activity\original data2'
file_path1 = file_path+ '\Plot Values 1.csv'
file_path2 = file_path+ '\Plot Values 2.csv'
file_path3 = file_path+ '\Plot Values 3.csv'
file_path4 = file_path+ '\Plot Values 4.csv'
file_path5 = file_path+ '\Plot Values 5.csv'
file_path6 = file_path+ '\Plot Values 6.csv'
file_path7 = file_path+ '\Plot Values 7.csv'
file_path8 = file_path+ '\Plot Values 8.csv'
file_path9 = file_path+ '\Plot Values 9.csv'
file_path10 = file_path+ '\Plot Values 10.csv'
file_path11 = file_path+ '\Plot Values 11.csv'
file_path12 = file_path+ '\Plot Values 12.csv'
file_path13 = file_path+ '\Plot Values 13.csv'
file_path14 = file_path+ '\Plot Values 14.csv'


# Process each file and calculate ΔF/F
delta_F_F1 = process_file(file_path1)
delta_F_F2 = process_file(file_path2)
delta_F_F3 = process_file(file_path3)
delta_F_F4 = process_file(file_path4)
delta_F_F5 = process_file(file_path5)
delta_F_F6 = process_file(file_path6)
delta_F_F7 = process_file(file_path7)
delta_F_F8 = process_file(file_path8)
delta_F_F9 = process_file(file_path9)
delta_F_F10 = process_file(file_path10)
delta_F_F11 = process_file(file_path11)
delta_F_F12 = process_file(file_path12)
delta_F_F13 = process_file(file_path13)
delta_F_F14 = process_file(file_path14)
# Load time from the first file (assuming all files have the same time vector)
time_data = pd.read_csv(file_path1)
# time = 0.001 * time_data.iloc[:, 0].values # Adjust based on your needs kiloHz
time = 0.00237 * time_data.iloc[:, 0].values # Adjust based on your needs 465

# Add offsets to the profiles for better visualization
offset2 = 1.0  # Offset for the second profile
offset3 = 2.0  # Offset for the third profile

# Plot the original, profile_2, and profile_3 in the same figure but with different heights
plt.figure(figsize=(5, 2.0))
plt.plot(time, delta_F_F1 + 0, linewidth=2, label='ROI 1')  # Profile 1 (original)
plt.plot(time, delta_F_F2 + offset2, linewidth=2, label='ROI 2')  # Profile 2 (with offset)
plt.plot(time, delta_F_F3 + offset3, linewidth=2, label='ROI 3')  # Profile 3 (with offset)
plt.plot(time, delta_F_F4 + 3, linewidth=2, label='ROI 4')  # Profile 1 (original)
plt.plot(time, delta_F_F5 + 4, linewidth=2, label='ROI 5')  # Profile 2 (with offset)
plt.plot(time, delta_F_F6 + 5, linewidth=2, label='ROI 6')  # Profile 3 (with offset)
plt.plot(time, delta_F_F7 + 6, linewidth=2, label='ROI 7')  # Profile 3 (with offset)
plt.plot(time, delta_F_F8 + 7, linewidth=2, label='ROI 8')  # Profile 1 (original)
# plt.plot(time, delta_F_F9 + 8, linewidth=2, label='ROI 9')  # Profile 2 (with offset)
# plt.plot(time, delta_F_F10 + 9, linewidth=2, label='ROI 10')  # Profile 3 (with offset)
# plt.plot(time, delta_F_F11 + 10, linewidth=2, label='ROI 11')  # Profile 3 (with offset)
plt.plot(time, delta_F_F12 + 11, linewidth=2, label='ROI 12')  # Profile 3 (with offset)
plt.plot(time, delta_F_F13 + 12, linewidth=2, label='ROI 13')  # Profile 3 (with offset)
plt.plot(time, delta_F_F14 + 13, linewidth=2, label='ROI 14')  # Profile 3 (with offset)


# Change the font size of the ticks
plt.tick_params(axis='both', which='major', labelsize=10)  # Major ticks
plt.tick_params(axis='both', which='minor', labelsize=10)   # Minor ticks (if any)
plt.yticks([0,1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14])  # Set specific Y-axis tick values
# plt.yticks([0,1, 2, 3, 4,5,6])  # Set specific Y-axis tick values
# Label the plot
plt.xlabel('Time[s]', fontsize=14)
# plt.ylabel('100% ΔF/F', fontsize=12)
plt.ylabel('100% $\\Delta F/F$', fontsize=12)
# plt.ylabel('100% ΔF/F', fontsize=12, fontstyle='italic')
plt.title('WO median filter')
plt.legend(loc='best', fontsize=10 )
plt.grid(False)
plt.xlim([0, max(time)])
plt.ylim([0, 15.0])
plt.show()
