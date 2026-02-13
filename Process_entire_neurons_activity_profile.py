# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:28:18 2026

@author: zhu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Function to process each file and calculate ΔF/F
import tifffile as tiff

# def process_tiff_stack(tiff_path, window_size=8000):
    
#     # Load TIFF (x, y, t)
#     data = tiff.imread(tiff_path)
    
#     # Ensure float32
#     data = data.astype(np.float32)
    
#     x_dim, y_dim, t_dim = data.shape
    
#     half_window = window_size // 2
    
#     # Output array
#     delta_F_F = np.zeros_like(data, dtype=np.float32)
    
#     # Loop over pixels
#     for x in range(x_dim):
#         for y in range(y_dim):
            
#             F_raw = data[x, y, :]
            
#             for i in range(t_dim):
#                 start_idx = max(0, i - half_window)
#                 end_idx = min(t_dim, i + half_window)
                
#                 F_baseline = np.percentile(F_raw[start_idx:end_idx], 10)
                
#                 D = np.floor(F_baseline) - 1
                
#                 delta_F_F[x, y, i] = (F_raw[i] - F_baseline) / (F_baseline - D)
    
#     return delta_F_F

def process_tiff_stack(tiff_path, window_size=8000):
    
    # Load TIFF (x, y, t)
    data = tiff.imread(tiff_path)
    data = np.transpose(data, (1, 2, 0))
    
    # Ensure float32
    data = data.astype(np.float32)
    
    x_dim, y_dim, t_dim = data.shape
    
    half_window = window_size // 2
    
    # Output array
    delta_F_F = np.zeros_like(data, dtype=np.float32)
    
    # Loop over pixels
    for x in range(x_dim):
        for y in range(y_dim):
            
            F_raw = data[x, y, :]
            
            for i in range(t_dim):
                start_idx = max(0, i - half_window)
                end_idx = min(t_dim, i + half_window)
                
                F_baseline = np.percentile(F_raw[start_idx:end_idx], 10)
                
                D = np.floor(F_baseline) - 1
                
                delta_F_F[x, y, i] = (F_raw[i] - F_baseline) / (F_baseline - D)
    
    return delta_F_F

tiff_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Original data\Z05_5dpf_OT_jrgeco1b_XYT_1070nm_2x2x4MHz_130mW_3Daverage.tif'

delta_F_F = process_tiff_stack(tiff_path, window_size=8000)

#%%

import numpy as np
import tifffile as tiff
from scipy.ndimage import percentile_filter

def process_tiff_stack_fast(tiff_path, window_size=8000):

    # Load (t, x, y)
    data = tiff.imread(tiff_path)

    # Convert to (x, y, t)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)

    # Sliding 10th percentile along time axis only
    F_baseline = percentile_filter(
        data,
        percentile=10,
        size=(1, 1, window_size),  # only filter along time
        mode='nearest'
    )

    D = np.floor(F_baseline) - 1
    epsilon = 1e-6

    delta_F_F = (data - F_baseline) / (F_baseline - D + epsilon)

    return delta_F_F

tiff_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Original data\Z05_5dpf_OT_jrgeco1b_XYT_1070nm_2x2x4MHz_130mW_3Daverage.tif'

delta_F_F = process_tiff_stack_fast(tiff_path, window_size=8000)