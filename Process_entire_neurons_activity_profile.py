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

# def process_tiff_stack(tiff_path, window_size=8000):
    
#     # Load TIFF (x, y, t)
#     data = tiff.imread(tiff_path)
#     data = np.transpose(data, (1, 2, 0))
    
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

# tiff_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Original data\Z05_5dpf_OT_jrgeco1b_XYT_1070nm_2x2x4MHz_130mW_3Daverage.tif'

# delta_F_F = process_tiff_stack(tiff_path, window_size=8000)

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

# Assuming delta_F_F is a 2D or 3D NumPy array
output_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Processed data\delta_F_F.tif'

# Save as TIFF
tiff.imwrite(output_path, delta_F_F.astype(np.float32))

#%%
import numpy as np
import tifffile as tiff
from scipy.ndimage import percentile_filter


def process_tiff_stack_safe(tiff_path,
                            window_size=8000,
                            block_size=50):

    print("Loading TIFF...")

    with tiff.TiffFile(tiff_path) as tif:
        data = tif.asarray()

    print("Original shape (t, x, y):", data.shape)

    # Convert to (x, y, t)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)
    x, y, t = data.shape

    print("Working shape (x, y, t):", data.shape)

    if window_size >= t:
        window_size = t - 1

    F_baseline = np.zeros_like(data, dtype=np.float32)

    total_blocks = x // block_size + 1
    block_counter = 0

    print("Starting rolling percentile computation...")

    for i in range(0, x, block_size):

        i_end = min(i + block_size, x)

        # Apply percentile filter only to this spatial block
        F_baseline[i:i_end, :, :] = percentile_filter(
            data[i:i_end, :, :],
            percentile=10,
            size=(1, 1, window_size),
            mode='nearest'
        )

        block_counter += 1
        progress = block_counter / total_blocks * 100
        print(f"Processing: {progress:.2f}% complete")

    print("Percentile filtering finished.")

    # ΔF/F
    D = np.floor(F_baseline) - 1
    epsilon = 1e-6

    delta_F_F = (data - F_baseline) / (F_baseline - D + epsilon)

    return delta_F_F


# -----------------------------
# Run
# -----------------------------
tiff_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Original data\Z05_5dpf_OT_jrgeco1b_XYT_1070nm_2x2x4MHz_130mW_3Daverage.tif'
output_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Processed data\delta_F_F.tif'

delta_F_F = process_tiff_stack_safe(
    tiff_path,
    window_size=8000,
    block_size=50
)

print("Saving result...")

tiff.imwrite(output_path,
             delta_F_F.astype(np.float32),
             bigtiff=True)

print("Done.")
#%%
import numpy as np
import tifffile as tiff
from scipy.ndimage import percentile_filter
import warnings
import os

# Optional: suppress harmless tifffile warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")


def process_tiff_stack(
        tiff_path,
        output_path,
        window_size=1500,   # 1500 recommended for 12000 frames
        block_size=50):

    print("Opening TIFF (memory-mapped)...")

    # Memory-map to avoid loading entire stack into RAM
    data = tiff.memmap(tiff_path)

    print("Original shape (t, x, y):", data.shape)

    # Convert to (x, y, t)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)
    x, y, t = data.shape

    print("Working shape (x, y, t):", data.shape)

    # Safety check
    if window_size >= t:
        window_size = t - 1
        print(f"Adjusted window_size to {window_size}")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Prepare output TIFF writer (BigTIFF for large file)
    with tiff.TiffWriter(output_path, bigtiff=True) as tif:

        total_blocks = (x + block_size - 1) // block_size
        block_counter = 0

        print("Starting rolling percentile computation...")

        for i in range(0, x, block_size):

            i_end = min(i + block_size, x)

            # Extract spatial block
            block = data[i:i_end, :, :]

            # Compute rolling 10th percentile along time axis
            F_baseline = percentile_filter(
                block,
                percentile=10,
                size=(1, 1, window_size),
                mode='nearest'
            )

            # ΔF/F
            D = np.floor(F_baseline) - 1
            epsilon = 1e-6
            delta_F_F_block = (block - F_baseline) / (F_baseline - D + epsilon)

            # Write block progressively to TIFF
            # Transpose back to (t, x_block, y)
            tif.write(
                np.transpose(delta_F_F_block, (2, 0, 1)).astype(np.float32),
                contiguous=True
            )

            block_counter += 1
            progress = block_counter / total_blocks * 100
            print(f"Processing: {progress:.2f}% complete")

    print("Processing finished and saved.")


# -------------------------------------------------
# FILE PATHS
# -------------------------------------------------
tiff_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Original data\Z05_5dpf_OT_jrgeco1b_XYT_1070nm_2x2x4MHz_130mW_3Daverage.tif'

output_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Processed data\delta_F_F.tif'


# -------------------------------------------------
# RUN
# -------------------------------------------------
process_tiff_stack(
    tiff_path,
    output_path,
    window_size=1500,   # <-- recommended
    block_size=50
)
#%%
import numpy as np
import tifffile as tiff
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

def process_tiff_stack_fast(
        tiff_path,
        output_path,
        window_size=1500,
        block_size=50,
        percentile=10):
    """
    Fast rolling percentile ΔF/F computation using block processing.
    """

    print("Opening TIFF (memory-mapped)...")
    data = tiff.memmap(tiff_path)  # memory-mapped read
    print("Original shape (t, x, y):", data.shape)

    # Convert to (x, y, t)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)
    x, y, t = data.shape
    print("Working shape (x, y, t):", data.shape)

    if window_size >= t:
        window_size = t - 1
        print(f"Adjusted window_size to {window_size}")

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare output TIFF
    with tiff.TiffWriter(output_path, bigtiff=True) as tif:

        total_blocks = (x + block_size - 1) // block_size
        block_counter = 0

        print("Starting fast rolling percentile computation...")

        for i in range(0, x, block_size):
            i_end = min(i + block_size, x)
            block = data[i:i_end, :, :]  # (block_size, y, t)

            # --- Fast running percentile per pixel ---
            # Pre-allocate baseline
            F_baseline = np.zeros_like(block, dtype=np.float32)

            half_win = window_size // 2

            # Compute rolling percentile along time axis
            for xi in range(block.shape[0]):
                for yi in range(block.shape[1]):
                    signal = block[xi, yi, :]
                    # Use sliding windows with padding
                    padded = np.pad(signal, (half_win, half_win), mode='edge')
                    # Compute percentile using vectorized stride trick
                    windows = np.lib.stride_tricks.sliding_window_view(
                        padded, window_shape=window_size)
                    F_baseline[xi, yi, :] = np.percentile(windows, percentile, axis=1)[:t]

            # ΔF/F
            D = np.floor(F_baseline) - 1
            epsilon = 1e-6
            delta_F_F_block = (block - F_baseline) / (F_baseline - D + epsilon)

            # Write block progressively
            tif.write(np.transpose(delta_F_F_block, (2, 0, 1)).astype(np.float32), contiguous=True)

            block_counter += 1
            progress = block_counter / total_blocks * 100
            print(f"Processing: {progress:.2f}% complete")

    print("Processing finished and saved.")

# -----------------------------
# FILE PATHS
# -----------------------------
tiff_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Original data\Z05_5dpf_OT_jrgeco1b_XYT_1070nm_2x2x4MHz_130mW_3Daverage.tif'
output_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Processed data\delta_F_F_fast.tif'

# -----------------------------
# RUN
# -----------------------------
process_tiff_stack_fast(
    tiff_path,
    output_path,
    window_size=1500,  # recommended for 12000 frames
    block_size=2,
    percentile=10
)
#%%
import numpy as np
import tifffile as tiff
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

def process_tiff_stack_fast(
        tiff_path,
        output_path,
        window_size=1500,
        block_size=20,
        percentile=10):
    """
    Fast rolling percentile ΔF/F computation using block processing.

    INPUT  shape: (t, x, y)
    OUTPUT shape: (t, x, y)

    ΔF/F(x,y,t) preserved exactly.
    """

    print("Opening TIFF (memory-mapped)...")
    data = tiff.memmap(tiff_path)  # shape: (t, x, y)
    print("Original shape (t, x, y):", data.shape)

    t, x, y = data.shape

    if window_size >= t:
        window_size = t - 1
        print(f"Adjusted window_size to {window_size}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---------------------------------------------------------
    # Allocate output memmap with FINAL CORRECT SHAPE
    # ---------------------------------------------------------
    print("Allocating output memmap...")
    output_memmap = np.memmap(
        output_path + ".tmp",
        dtype=np.float16,
        mode='w+',
        shape=(t, x, y)
    )

    total_blocks = (x + block_size - 1) // block_size
    block_counter = 0

    print("Starting rolling percentile computation...")

    half_win = window_size // 2

    for i in range(0, x, block_size):
        i_end = min(i + block_size, x)
        current_block_size = i_end - i

        # Load block: (t, block_x, y)
        block = data[:, i:i_end, :].astype(np.float32)

        # Rearrange for easier pixelwise processing → (block_x, y, t)
        block_xyz = np.transpose(block, (1, 2, 0))

        F_baseline = np.zeros_like(block_xyz, dtype=np.float32)

        # ---------------------------------------------------------
        # Rolling percentile per pixel
        # ---------------------------------------------------------
        for xi in range(current_block_size):
            for yi in range(y):
                signal = block_xyz[xi, yi, :]
                padded = np.pad(signal, (half_win, half_win), mode='edge')

                windows = np.lib.stride_tricks.sliding_window_view(
                    padded, window_shape=window_size
                )

                F_baseline[xi, yi, :] = np.percentile(
                    windows, percentile, axis=1
                )[:t]

        # ---------------------------------------------------------
        # ΔF/F
        # ---------------------------------------------------------
        D = np.floor(F_baseline) - 1
        epsilon = 1e-6

        delta_F_F = (block_xyz - F_baseline) / (F_baseline - D + epsilon)

        # Convert back to (t, block_x, y)
        delta_F_F_txy = np.transpose(delta_F_F, (2, 0, 1)).astype(np.float16)

        # Insert into correct x-location
        output_memmap[:, i:i_end, :] = delta_F_F_txy

        block_counter += 1
        progress = block_counter / total_blocks * 100
        print(f"Processing: {progress:.2f}% complete")

    # Flush memmap to disk
    output_memmap.flush()

    print("Writing final BigTIFF...")

    # Save as proper TIFF with correct shape
    tiff.imwrite(
        output_path,
        output_memmap,
        bigtiff=True
    )

    # Remove temporary memmap file
    del output_memmap
    os.remove(output_path + ".tmp")

    print("Processing finished.")
    print("Final output shape:", (t, x, y))


# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------

tiff_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Original data\Z05_5dpf_OT_jrgeco1b_XYT_1070nm_2x2x4MHz_130mW_3Daverage.tif'
output_path = r'D:\SPIM_2P\20231010 zebrafish Jrgeco1b\Processed data\delta_F_F_fast.tif'

process_tiff_stack_fast(
    tiff_path,
    output_path,
    window_size=1500,   # good for ~12000 frames
    block_size=20,
    percentile=10
)
