# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:34:02 2024

@author: zhu
"""

import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage import measure
import imutils
from imutils import contours
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy import interpolate
from skimage import io
import scipy
import matplotlib as mpl
from sklearn.metrics import r2_score

# Configure matplotlib settings
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

# Define global variable for all points
allpoints = []

# Function to get immediate subdirectories
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# Mono-exponential decay function
def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + 1*b

# Define a double-exponential decay function
def double_exponential(t, a1, t1, a2, t2, c):
    return a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2) + c

# Define the loss function to minimize (sum of squared errors)
def loss_function(params, t, y):
    a1, t1, a2, t2, c = params
    y_pred = double_exponential(t, a1, t1, a2, t2, c)
    return np.sum((y - y_pred) ** 2)

def rebin_3d(arr, new_shape):
    """
    Rebin a 3D array to a new shape by averaging.

    Parameters:
    arr (numpy.ndarray): The original 3D array.
    new_shape (tuple): The new shape (depth, rows, columns).

    Returns:
    numpy.ndarray: The rebinned array.
    """
    # Ensure the new shape is a factor of the original shape
    if (arr.shape[0] % new_shape[0] != 0 or
        arr.shape[1] % new_shape[1] != 0 or
        arr.shape[2] % new_shape[2] != 0):
        raise ValueError("New shape must be a factor of the original shape.")
    
    # Calculate the shape of the blocks
    depth_factor = arr.shape[0] // new_shape[0]
    row_factor = arr.shape[1] // new_shape[1]
    col_factor = arr.shape[2] // new_shape[2]
    
    # Reshape the array into blocks and compute the mean
    reshaped = arr.reshape(new_shape[0], depth_factor,
                           new_shape[1], row_factor,
                           new_shape[2], col_factor)
    
    return reshaped.mean(axis=(1, 3, 5))

#binning data
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1],)
    return arr.reshape(shape).mean(-1).mean(1)

# Initialize a dictionary to store lists of points
points_dict = {}
Inten_points_dict = {}

# Directory and data setup
DIR = 'F://Dale_data/01-07-2024 Bleachning - Copy/'
# DIR = 'F://Dale_data/22-03-16 Bleaching/'
test_name_pool = ['8MHz','4x2MHz','2x2x2MHz','4MHz']
# test_name_pool = ['2x2x2MHz']
ratio_threshold = [0.97,0.97,0.97,0.98]
# test_name_pool = ['2x2x2MHz']
x_dim = 200
case_value = 1
#%%
from scipy.optimize import differential_evolution
# new_shape = (75, 250, 100•)
for index_file, test_name in enumerate(test_name_pool):
    data_arr = []

    # Get all folders at the requested rep. rate
    for val in [x[0] for x in os.walk(DIR)]:
        if test_name in val:
            if os.path.isdir(os.path.join(DIR, val)):
                data_arr.append(os.path.basename(val))

    allpoints = []
    inetn_allpoints = []

    # Process each dataset
    for data_name in data_arr:
        print(data_name)
        
        # Load image stack and convert to grayscale
        img_arr = io.imread(os.path.join(DIR, data_name, data_name + '.tiff'))[:, :, 100:-100]
        new_shape = (img_arr.shape[0], 500, 300)
        # new_shape = (75, 250, 100)
        img_arr = rebin_3d(img_arr, new_shape)
        img_grayscale = np.mean(img_arr, axis=0)
        
        # # Thresholding to remove background
        # thresh = cv2.threshold(img_arr[0], 120, 255, cv2.THRESH_BINARY)[1]

        # Create a map of the initial slopes m*t
        xs = 1*np.linspace(0, img_arr.shape[0] - 1, img_arr.shape[0])
        t_arr = np.zeros((img_arr.shape[2], img_arr.shape[1]))
        mt_arr = np.zeros((img_arr.shape[2], img_arr.shape[1]))
        err_arr = np.zeros((img_arr.shape[2], img_arr.shape[1]))
        
        # Threshold and mask
        # thresh = cv2.threshold(img_arr[1],120, 255, cv2.THRESH_BINARY)[1]
        normalized_img = cv2.normalize(img_arr[1], None, 0, 255, cv2.NORM_MINMAX)
        normalized_img = normalized_img.astype(np.uint8)
 
        # Apply adaptive thresholding
        # thresh = cv2.adaptiveThreshold(normalized_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                         cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.threshold(normalized_img, 120, 255, cv2.THRESH_BINARY)[1]
        # plt.figure()
        # plt.imshow(normalized_img)
        # plt.colorbar()
        labels = measure.label(thresh, connectivity=2, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
        
        # plt.figure(figsize=(8, 6))
        
        # # First subplot: normalized image
        # plt.subplot(1, 2, 1)
        # plt.imshow(normalized_img, cmap='gray')
        # cbar1 = plt.colorbar(shrink=0.4, aspect=10, pad=0.02)  # Adjust colorbar size
        # cbar1.ax.tick_params(labelsize=10)  # Set fontsize for colorbar ticks
        # plt.title('Normalized Image', fontsize=14)
        # plt.xlabel('X-axis label', fontsize=12)
        # plt.ylabel('Y-axis label', fontsize=12)
        # plt.tick_params(axis='both', labelsize=10)
        
        # # Second subplot: labels
        # plt.subplot(1, 2, 2)
        # plt.imshow(labels)
        # cbar2 = plt.colorbar(shrink=0.4, aspect=10, pad=0.02)  # Adjust colorbar size
        # cbar2.ax.tick_params(labelsize=10)  # Set fontsize for colorbar ticks
        # plt.title('Labels', fontsize=14)
        # plt.xlabel('X-axis label', fontsize=12)
        # plt.ylabel('Y-axis label', fontsize=12)
        # plt.tick_params(axis='both', labelsize=10)
        
        # # Show the plot with tight layout
        # plt.tight_layout()
        # plt.show()
        
        # Analyze each unique cell
        for label in np.unique(labels):
            if label == 0:
                continue
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            # plt.figure()
            # plt.imshow(labelMask)
            numPixels = cv2.countNonZero(labelMask)  
            
            if numPixels > 40 and case_value == 1:
                # Extract data for the current cell
                cell_data = []
                indices = np.where(labelMask == 255)
                if np.size(indices[0]) > 40:
                    # plt.figure()
                    # plt.imshow(labelMask)
                    cell_img_arr = img_arr[:, indices[0], indices[1]]
                    # Fit the double-exponential model to each cell
                    try:
                        p0 = (2000, .1, 50)
                        params, cv = scipy.optimize.curve_fit(monoExp, xs, np.mean(cell_img_arr,axis = 1)-100, p0)
                        m, t, b = params
                        y_pred = monoExp(xs, *params)
                        # plt.figure(figsize=(6, 4))
                        # # plt.figure()
                        # plt.plot(0.26*xs,np.mean(cell_img_arr,axis = 1)-100,label='raw data')
                        # plt.plot(0.26*xs,y_pred,"o",label='Exp fitting')
                        # plt.xlabel('T[s] ', fontsize=12)
                        # plt.ylabel('2P signal a.u.', fontsize=12)
                        # plt.tick_params(axis='both', labelsize=10)
                        # plt.legend(fontsize=12)
                        
                        err = r2_score(np.mean(cell_img_arr,axis = 1)-100, y_pred) ** 2
                        if err > 0.98 and  t<=0.2 and np.size(cell_img_arr,axis=0) == 75:
                            # plt.figure()
                            # plt.plot(np.mean(cell_img_arr,axis = 1)-100)
                            # plt.plot(y_pred)
                            # plt.figure()
                            # plt.imshow(labelMask)
                            allpoints.append(t/40)
                            inetn_allpoints.append(np.mean(cell_img_arr,axis = 1)-100)
                    except:
                        pass
            
            if numPixels > 40 and case_value == 2:
                # Extract data for the current cell
                cell_data = []
                indices = np.where(labelMask == 255)
                if np.size(indices[0]) > 40:
                    cell_img_arr = img_arr[:, indices[0], indices[1]]
                    # Fit the double-exponential model to each cell
                    try:
                        
                        # Example data (t = time points, y = observed data)
                        # t = xs  # Replace with your time points
                        # y = np.mean(cell_img_arr, axis=1) - 100  # Replace with your observed data
                        
                        # # Set parameter bounds (use reasonable ranges based on your data)
                        # bounds = [(0, 400),  # Bounds for a1
                        #           (0, 50),  # Bounds for t1
                        #           (0, 400),  # Bounds for a2
                        #           (0, 200),  # Bounds for t2
                        #           (0, 100)]  # Bounds for c
                        
                        # # Perform differential evolution
                        # result = differential_evolution(loss_function, bounds, args=(t, y))
                        
                        # # Extract the optimized parameters
                        # a1, t1, a2, t2,_ = result.x
                        # y_pred = double_exponential(xs, *result.x)
                        
                        p0 = [0.8, 10, 0.5, 40, 20]
                        params, _ = scipy.optimize.curve_fit(double_exponential, xs, np.mean(cell_img_arr,axis = 1)-100, p0)
                        a1, t1, a2, t2,_ = params
                        y_pred = double_exponential(xs, *params)
                        # plt.figure()
                        # plt.plot(np.mean(cell_img_arr,axis = 1)-100)
                        # plt.plot(y_pred)
                        err = r2_score(np.mean(cell_img_arr,axis = 1)-100, y_pred) ** 2
                        if err > 0.99 and  0<t1<50 and np.size(cell_img_arr,axis=0) == 75:
                            allpoints.append(t1)
                            # allpoints.append(t2)
                            inetn_allpoints.append(np.mean(cell_img_arr,axis = 1)-100)
                    except:
                        pass

    # Store the results in the dictionaries
    points_dict['All_' + str(test_name) + 'points'] = allpoints
    Inten_points_dict['All_' + str(test_name) + 'points'] = inetn_allpoints
    print(points_dict['All_' + str(test_name) + 'points'])
    
#%%   
from scipy import stats
test_name_pool = test_name_pool    
all_points = [np.array(points_dict['All_' + label + 'points']) for label in test_name_pool]

# Order and colors
order = [0, 1, 2, 3]
colors = plt.cm.tab20(np.linspace(0, 0.4, len(order)))

# Setting up matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

# Creating the figure and axis
fig = plt.figure(figsize=(5.0, 4.0))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # Adjusted to give space for labels

# Box plot for the data distributions
ax1.boxplot(all_points, positions=range(len(test_name_pool)), widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='lightgray', color='black', alpha=0.4),
            medianprops=dict(color='r'))

second_row_labels = [len(points_dict['All_' + test_name_pool[0] + 'points']),len(points_dict['All_' + test_name_pool[1] + 'points']),
                     len(points_dict['All_' + test_name_pool[2] + 'points']),len(points_dict['All_' + test_name_pool[3] + 'points'])]

# Loop through the test_name_pool and plot individual points and means with error bars
mean_values = []

for i, label in enumerate(test_name_pool):
    key = 'All_' + label + 'points'
    points = np.array(points_dict[key])

    # Calculate mean and standard deviation
    mean_value = np.mean(points)
    std_value = np.std(points)
    
    # Store mean values for later use
    mean_values.append(mean_value)
    
    # Scatter plot for individual points
    x_vals = np.full(points.shape, i)+ np.random.uniform(-0.1, 0.1, size=len(points))  # small random jitter  # Create an array of the same value i, for x positions
    
    ax1.scatter(x_vals, points, color=colors[order[i]], alpha=0.4, label=label if i == 0 else "")  # Add label only for the first set for the legend

    # Scatter plot for the mean
    ax1.scatter(i, mean_value, color='k', zorder=5, alpha=1.0)
    
    # Annotate the mean value
    
    ax1.text(i, mean_value, f'$\hat{{\mu}}$ = {mean_value:.3f}', color='k', ha='center', va='bottom')
    ax1.text(i, 0.23, f'$N$ = {second_row_labels[i]:.0f}', color='k', ha='center', va='top')
    # ax1.text(i, mean_value, f'$\\hat{{\\mu}}_{{mean}}$ = {mean_value:.3f}', color='k', ha='right', va='bottom')

    # ax1.text(i, mean_value + std_value, f'$\hat{{\n}}$ ={points.shape:.2f}', color='k', ha='left', va='bottom')

# Setting x-ticks and labels
ax1.set_xticks(range(len(test_name_pool)))
ax1.set_xticklabels(test_name_pool, rotation=0, fontsize = 10,ha='center')
# ax1.set_title("Bleaching Rates",fontsize = 16)
ax1.set_ylabel('Decay (a.u.)',fontsize = 14)
ax1.set_xlabel('Repetition rate',fontsize = 14)
# ax1.set_ylim([0, 0.2])
# # Setting y-axis limit
# ax1.set_ylim([0, 0.1 * max([np.max(points) for points in all_points])])

# # Create a secondary x-axis
# ax2 = ax1.twiny()
# second_row_labels = [len(points_dict['All_' + test_name_pool[0] + 'points']),len(points_dict['All_' + test_name_pool[1] + 'points']),
#                      len(points_dict['All_' + test_name_pool[2] + 'points']),len(points_dict['All_' + test_name_pool[3] + 'points'])]
# # Set the second row of x-axis labels
# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks(range(len(test_name_pool)))
# ax2.set_xticklabels(second_row_labels, rotation=0, fontsize=12, ha='center')

# # Adjust the position of the second x-axis
# ax2.xaxis.set_ticks_position('top')
# ax2.xaxis.set_label_position('top')
# ax2.spines['bottom'].set_position(('outward', 36))
# Pairwise comparisons
comparisons = [("8MHz", "4x2MHz"), ("8MHz", "2x2x2MHz"), ("8MHz", "4MHz")]
Height = [0.005,0.026,0.03]
from scipy import stats
for i, label in enumerate(comparisons):
    group1 = points_dict['All_' + label[0] + 'points']
    group2 = points_dict['All_' + label[1] + 'points']
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    # Find y-position for the annotation
    y_max = max(np.max(group1), np.max(group2))
    x1, x2 = test_name_pool.index(label[0]), test_name_pool.index(label[1])
    y, h, col = y_max + Height[i], 0.01, 'k'
    
    # Plot the line and annotate p-value
    ax1.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color=col)
    ax1.text((x1 + x2) * 0.5, y + h, f"$p$ = {p_val:.3f}", ha='center', va='bottom', color=col)
# Setting y-axis limit
ax1.set_ylim([0, 1.4 * max([np.max(points) for points in all_points])])

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
test_name_pool = ['4MHz','2x2x2MHz','4x2MHz','8MHz']
# test_name_pool = ['8MHz','4x2MHz','2x2x2MHz','4MHz']
# Data preparation
all_points = [np.array(points_dict['All_' + label + 'points']) for label in test_name_pool]

order = [0, 1, 2, 3]  # Exclude irrelevant categories
# order = [3, 2, 1, 0]  # Exclude irrelevant categories
colors = plt.cm.tab20(np.linspace(0, 0.4, len(order)))

# Setting up matplotlib parameters
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

# Creating the figure and axis
fig = plt.figure(figsize=(5.0, 3.73))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # Adjusted to give space for labels

# Violin plot
sns.violinplot(data=all_points, palette=colors, ax=ax1)

# Setting x-ticks and labels
ax1.set_xticks(range(len(test_name_pool)))
# ax1.set_xticklabels(test_name_pool, rotation=0, fontsize=12, ha='center')
ax1.set_ylabel(r'Photobleaching rate $k$', fontsize=14)
ax1.set_xlabel('Average pulse frequency', fontsize=14)

import matplotlib.ticker as mticker

# Scale y-axis values by 10³
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 1e3:.1f}"))

# Update y-axis label to include 10⁻³ as a factor
ax1.set_ylabel(r'Photobleaching rate $k$ ($\times 10^{-3}$)', fontsize=14)
# Pairwise comparisons
comparisons = [("8MHz", "4x2MHz"), ("8MHz", "2x2x2MHz"), ("8MHz", "4MHz")]
heights = [0.0006, 0.00105, 0.00095]
for i, label in enumerate(comparisons):
    group1 = points_dict['All_' + label[0] + 'points']
    group2 = points_dict['All_' + label[1] + 'points']
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    # Find y-position for the annotation
    y_max = max(np.max(group1), np.max(group2))
    x1, x2 = test_name_pool.index(label[0]), test_name_pool.index(label[1])
    y, h, col = y_max + heights[i], 0.0001, 'k'
    
    # Plot the line and annotate p-value
    ax1.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color=col)
    ax1.text((x1 + x2) * 0.5, y + h, f"$p$ = {p_val:.2f}", ha='center', va='bottom', color=col)

# Setting y-axis limit
ax1.set_ylim([-0.001, 1.4 * max([np.max(points) for points in all_points])])
# ax1.set_ylim([-0.002, 0.008])
# Adding scatter points and mean annotations
mean_values = []
median_values = []
for i, label in enumerate(test_name_pool):
    key = 'All_' + label + 'points'
    points = np.array(points_dict[key])

    # Calculate mean and median
    mean_value = np.mean(points)
    median_value = np.median(points)
    
    mean_values.append(mean_value)
    median_values.append(median_value)
# Connect the median values with a line
ax1.plot(range(len(median_values)), median_values, color='k', linestyle='--', linewidth=1.5, marker='o', markersize=5, label="Median")
# Modify the x-tick labels with LaTeX formatting for "MHz"
x_labels = [r'$\mathit{' + label.replace("MHz", r'\ MHz') + '}$' for label in test_name_pool]
plt.gca().set_xticklabels(x_labels, fontsize=12)


# plt.tight_layout()
plt.show()

#%%
from scipy import stats
test_name_pool = test_name_pool    
all_points = [np.array(points_dict['All_' + label + 'points']) for label in test_name_pool]

# Order and colors
order = [0, 1, 2, 3]
colors = plt.cm.tab20(np.linspace(0, 0.4, len(order)))

# Setting up matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

# Creating the figure and axis
fig = plt.figure(figsize=(6, 5.5))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # Adjusted to give space for labels

second_row_labels = [len(points_dict['All_' + test_name_pool[0] + 'points']),len(points_dict['All_' + test_name_pool[1] + 'points']),
                     len(points_dict['All_' + test_name_pool[2] + 'points']),len(points_dict['All_' + test_name_pool[3] + 'points'])]

# Box plot for the data distributions
ax1.boxplot(all_points, positions=range(len(test_name_pool)), widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='lightgray', color='black', alpha=0.4),
            medianprops=dict(color='r'))

# Loop through the test_name_pool and plot individual points and means with error bars
mean_values = []

for i, label in enumerate(test_name_pool):
    key = 'All_' + label + 'points'
    points = np.array(points_dict[key])

    # Calculate mean and standard deviation
    mean_value = np.mean(points)
    std_value = np.std(points)
    
    # Store mean values for later use
    mean_values.append(mean_value)
    
    # Scatter plot for individual points
    x_vals = np.full(points.shape, i)+ np.random.uniform(-0.1, 0.1, size=len(points))  # small random jitter  # Create an array of the same value i, for x positions
    
    ax1.scatter(x_vals, points, color=colors[order[i]], alpha=0.4, label=label if i == 0 else "")  # Add label only for the first set for the legend

    # Scatter plot for the mean
    ax1.scatter(i, mean_value, color='k', zorder=5, alpha=1.0)
    
    # Annotate the mean value
    ax1.text(i, mean_value, f'$\hat{{\mu}}$ ={mean_value:.3f}', color='k', ha='center', va='bottom')
    
    # ax1.text(i, mean_value, f'$\\hat{{\\mu}}_{{mean}}$ = {mean_value:.3f}', color='k', ha='right', va='bottom')

    # ax1.text(i, mean_value + std_value, f'$\hat{{\n}}$ ={points.shape:.2f}', color='k', ha='left', va='bottom')

# Setting x-ticks and labels
ax1.set_xticks(range(len(test_name_pool)))
ax1.set_xticklabels(test_name_pool, rotation=0, fontsize = 12,ha='center')
# ax1.set_title("Bleaching Rates",fontsize = 16)
ax1.set_ylabel('Decay (a.u.)',fontsize = 14)

# Setting y-axis limit
ax1.set_ylim([0, 2.0 * max([np.max(points) for points in all_points])])
# Create a secondary x-axis
ax2 = ax1.twiny()
second_row_labels = [len(points_dict['All_' + test_name_pool[0] + 'points']),len(points_dict['All_' + test_name_pool[1] + 'points']),
                     len(points_dict['All_' + test_name_pool[2] + 'points']),len(points_dict['All_' + test_name_pool[3] + 'points'])]
# Set the second row of x-axis labels
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(range(len(test_name_pool)))
ax2.set_xticklabels(second_row_labels, rotation=0, fontsize=12, ha='center')

# Adjust the position of the second x-axis
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
# ax2.spines['bottom'].set_position(('outward', 36))
# Pairwise comparisons
comparisons = [("8MHz", "4x2MHz"), ("8MHz", "2x2x2MHz"), ("8MHz", "4MHz")]
Height = [0.005,0.026,0.03]
from scipy import stats
for i, label in enumerate(comparisons):
    group1 = points_dict['All_' + label[0] + 'points']
    group2 = points_dict['All_' + label[1] + 'points']
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    # Find y-position for the annotation
    y_max = max(np.max(group1), np.max(group2))
    x1, x2 = test_name_pool.index(label[0]), test_name_pool.index(label[1])
    y, h, col = y_max + Height[i], 0.01, 'k'
    
    # Plot the line and annotate p-value
    ax1.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color=col)
    ax1.text((x1 + x2) * 0.5, y + h, f"$p$ = {p_val:.3f}", ha='center', va='bottom', color=col)
# Setting y-axis limit
ax1.set_ylim([0, 1.4 * max([np.max(points) for points in all_points])])
#%%
# figsize=(5.0, 3.73)
from sklearn import preprocessing
plt.rcParams['figure.figsize'] = [5.0,3.375]
plt.figure()
# order = [1,2,3,0]
# # Order and colors
order = [0, 1, 2, 3]
# colors = plt.cm.tab20(np.linspace(0, 0.2, len(order)))
test_name_pool = ['8MHz','4x2MHz','2x2x2MHz','4MHz']

num_rows = 75
# Order and colors

order = [0, 1, 2, 3]
colors = plt.cm.tab20(np.linspace(0, 0.4, len(order)))

for i, label in enumerate(test_name_pool):
    key = 'All_' + label + 'points'
    D2_points = Inten_points_dict[key]
    original_data = np.zeros((len(D2_points), D2_points[0].shape[0]))
    column_data =  np.zeros(D2_points[0].shape[0])
    for index_yin in range(len(D2_points)):
        # Extract column data
        column_data = D2_points[index_yin]
        # Subtract 100 and normalize the column data
        # avg = column_data.mean()
        # std = column_data.std()
        # normalized = (column_data - avg) / std
        
        a = 0.00
        # a = 0
        b = column_data.max()
        normalized = (column_data-a)/(b-a)
        # Update original_data with the normalized column data
        original_data[index_yin,:] = normalized
    
    # Calculate mean and standard error (assuming standard error of the mean)
    mean_values = np.mean(original_data, axis=0)
    std_err = np.std(original_data, axis=0) / np.sqrt(original_data.shape[0])  # Standard error of the mean
    
    label = test_name_pool[i]
    
    # Plotting error bars with Pastel1 colors
    plt.errorbar(np.linspace(1, num_rows, num_rows)*0.26, mean_values, yerr=std_err, alpha=1.0, fmt=':', capsize=3, capthick=2, color=colors[order[i]], label=label)
    plt.fill_between(np.linspace(1, 75, num_rows)*0.26, mean_values - std_err, mean_values + std_err, alpha=0.1, color=colors[order[i]])
    

# plt.text(10, 0.06, f'$\hat{{\mu}}$ = {mean_values[-1]:.2f}', ha='center', va='bottom', color=colors[index], fontsize=20)
    
# Customize plot parameters
plt.xlim((0.01, num_rows*0.26))  # Adjust x-axis limit
plt.legend(fontsize=10)  # Show legend
#get handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
plt.xlabel('T[s] ',fontsize=14)  # X-axis label
plt.ylabel('2P signal(a.u.)',fontsize=14)  # Y-axis label
plt.show()
plt.tight_layout()
#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['figure.figsize'] = [5.2,3.88]
plt.figure()
factor = 40
# Order and colors
order = [0, 1, 2, 3]
colors = plt.cm.tab20(np.linspace(0, 0.4, len(order)))

# test_name_pool = ['8MHz', '4x2MHz', '2x2x2MHz', '4MHz']
test_name_pool = ['4MHz','2x2x2MHz','4x2MHz','8MHz']
num_rows = 75

for i, label in enumerate(test_name_pool):
    key = 'All_' + label + 'points'
    D2_points = np.array(Inten_points_dict[key])
    original_data = np.zeros(D2_points.shape)
    
    for index_yin in range(D2_points.shape[0]):
        column_data = D2_points[index_yin, :] 
        
        # Normalize the column data
        # a = column_data.min()
        a = 0
        b = column_data.max()
        normalized = (column_data - a) / (b - a)
        original_data[index_yin, :] = normalized
    
    mean_values = np.mean(original_data, axis=0)
    std_err = np.std(original_data, axis=0) / np.sqrt(original_data.shape[0])
    legend_label = '2x4MHz' if label == '4x2MHz' else label
    # Plotting error bars with specified colors
    plt.errorbar(np.linspace(1, num_rows, num_rows) * factor, mean_values, yerr=std_err, alpha=1.0, fmt=':', capsize=3, capthick=2, color=colors[order[i]], label=legend_label)
    plt.fill_between(np.linspace(1, 75, num_rows) * factor, mean_values - std_err, mean_values + std_err, alpha=0.1, color=colors[order[i]])
legend_label = '2x4MHz' if label == '4x2MHz' else label
plt.xlim((0.01, num_rows * factor))
plt.legend(fontsize=14, frameon=False)
plt.xlabel('Image number', fontsize=15)
plt.ylabel('2PEF signal (a.u.)', fontsize=15)

# Create a secondary x-axis for the second row of labels
ax1 = plt.gca()

# Add zoomed inset
x1, x2, y1, y2 = 3*4*factor, 3*7*factor, 0.30, 0.55
axins = inset_axes(ax1, width="70%", height="70%", loc='upper left', bbox_to_anchor=(0.25, 0.25, 0.47, 0.47), bbox_transform=ax1.transAxes)


for i, label in enumerate(test_name_pool):
    key = 'All_' + label + 'points'
    D2_points = np.array(Inten_points_dict[key])
    original_data = np.zeros(D2_points.shape)
    
    for index_yin in range(D2_points.shape[0]):
        column_data = D2_points[index_yin, :] - 100
        a = column_data.min()
        b = column_data.max()
        normalized = (column_data - a) / (b - a)
        original_data[index_yin, :] = normalized
    
    mean_values = np.mean(original_data, axis=0)
    std_err = np.std(original_data, axis=0) / np.sqrt(original_data.shape[0])
    
    # Plotting error bars with specified colors
    axins.errorbar(np.linspace(1, num_rows, num_rows) * factor, mean_values, yerr=std_err, alpha=1.0, fmt=':', capsize=3, capthick=2, color=colors[order[i]])
    axins.fill_between(np.linspace(1, 75, num_rows) * factor, mean_values - std_err, mean_values + std_err, alpha=0.1, color=colors[order[i]])

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])

mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.show()
plt.tight_layout()

#%% Logscale
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['figure.figsize'] = [5.2, 3.88]
plt.figure()
factor = 40

# Order and colors
order = [0, 1, 2, 3]
colors = plt.cm.tab20(np.linspace(0, 0.4, len(order)))

test_name_pool = ['4MHz','2x2x2MHz','4x2MHz','8MHz']
num_rows = 75
eps = 1e-3  # small offset to avoid log(0)

for i, label in enumerate(test_name_pool):
    key = 'All_' + label + 'points'
    D2_points = np.array(Inten_points_dict[key])
    original_data = np.zeros(D2_points.shape)
    
    for index_yin in range(D2_points.shape[0]):
        column_data = D2_points[index_yin, :]
        a = 0
        b = column_data.max()
        normalized = (column_data - a) / (b - a + 1e-12)
        original_data[index_yin, :] = normalized
    
    mean_values = np.mean(original_data, axis=0) + eps
    std_err = np.std(original_data, axis=0) / np.sqrt(original_data.shape[0])
    
    plt.errorbar(np.linspace(1, num_rows, num_rows) * factor,
                 mean_values, yerr=std_err,
                 alpha=1.0, fmt=':', capsize=3, capthick=2,
                 color=colors[order[i]], label=label)
    plt.fill_between(np.linspace(1, 75, num_rows) * factor,
                     mean_values - std_err, mean_values + std_err,
                     alpha=0.1, color=colors[order[i]])

plt.xlim((0.01, num_rows * factor))
plt.legend(fontsize=12, frameon=False)
plt.xlabel('Image number', fontsize=14)
plt.ylabel('2P signal (a.u.)', fontsize=14)

from matplotlib.ticker import LogLocator, ScalarFormatter

plt.yscale("log")

ax = plt.gca()

# Major ticks at powers of 10
ax.yaxis.set_major_locator(LogLocator(base=10.0))

# Optional: add minor ticks between (2,3,...9)
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))

# Force numeric labels instead of scientific notation
formatter = ScalarFormatter()
formatter.set_scientific(False)
formatter.set_useOffset(False)
ax.yaxis.set_major_formatter(formatter)



# # Create inset
# ax1 = plt.gca()
# x1, x2, y1, y2 = 3*4*factor, 3*7*factor, 0.30, 0.55  # keep raw values
# axins = inset_axes(ax1, width="70%", height="70%", loc='upper right',
#                    bbox_to_anchor=(0.25, 0.25, 0.47, 0.47),
#                    bbox_transform=ax1.transAxes)

# for i, label in enumerate(test_name_pool):
#     key = 'All_' + label + 'points'
#     D2_points = np.array(Inten_points_dict[key])
#     original_data = np.zeros(D2_points.shape)
    
#     for index_yin in range(D2_points.shape[0]):
#         column_data = D2_points[index_yin, :] - 100
#         a = column_data.min()
#         b = column_data.max()
#         normalized = (column_data - a) / (b - a + 1e-12)
#         original_data[index_yin, :] = normalized
    
#     mean_values = np.mean(original_data, axis=0) + eps
#     std_err = np.std(original_data, axis=0) / np.sqrt(original_data.shape[0])
    
#     axins.errorbar(np.linspace(1, num_rows, num_rows) * factor,
#                    mean_values, yerr=std_err,
#                    alpha=1.0, fmt=':', capsize=3, capthick=2,
#                    color=colors[order[i]])
#     axins.fill_between(np.linspace(1, 75, num_rows) * factor,
#                        mean_values - std_err, mean_values + std_err,
#                        alpha=0.1, color=colors[order[i]])

# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)      # works fine with log scale
# axins.set_yscale("log")     # let matplotlib do the log transform
# axins.set_xticklabels([])
# axins.set_yticklabels([])

# mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

#%% save file
newpath = DIR + "/draft/" + 'paper'
if not os.path.exists(newpath):
       os.makedirs(newpath)
import json  # Import the json module      
# Save points_dict to a JSON file
file_path = os.path.join(newpath, 'points_dict5.json')
with open(file_path, 'w') as f:
    json.dump(points_dict, f, indent=4)
    
file_path = os.path.join(newpath, 'points_dict6.json')
with open(file_path, 'w') as f:
    json.dump(Inten_points_dict, f, indent=4)    

print(f'points_dict saved to: {file_path}') 

#%% read file
import json  # Import the json module    
file_path = os.path.join(DIR, 'draft', 'paper', 'points_dict5.json')

# Check if the file exists
if os.path.exists(file_path):
    # Read the JSON file and load it into a dictionary
    with open(file_path, 'r') as f:
        points_dict_loaded = json.load(f)
    
    # Print the loaded dictionary to verify
    print("Successfully loaded points_dict:")
    print(points_dict_loaded)
else:
    print(f"Error: File '{file_path}' does not exist.")
points_dict = points_dict_loaded
#%% Loading data from .csv file
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import interpolate
import os
from numpy import genfromtxt
import csv

plt.rcParams['figure.figsize'] = [8, 5]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def signaltonoise(a, axis=0):
    m = a.mean(axis)
    ma = np.amax(a, axis)
    return ma / m

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)]

def read_csv_file(file_path):
    """
    Read a CSV file and return its contents as a list of dictionaries.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    list: A list of dictionaries, each representing a row in the CSV file.
    """
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]
def read_column_from_data(data, column_name):
    """
    Extract a specific column from the data.

    Parameters:
    data (list): The list of dictionaries representing the CSV data.
    column_name (str): The name of the column to extract.

    Returns:
    list: A list of values from the specified column.
    """
    return [row[column_name] for row in data]

def get_column_names_from_data(data):
    """
    Get the column names from the CSV data.

    Parameters:
    data (list): The list of dictionaries representing the CSV data.

    Returns:
    list: A list of column names.
    """
    if data:
        return list(data[0].keys())
    else:
        return []

# data_dir = 'Z:\\2016_2020_ZebrafishHeartProject\\Dale - Selected Data\\'
data_dir = 'C://Users/zhu\Dropbox/2022.PulseSplittingPaper/Results/05_Photobleaching_done/'
# save_dir = 'M:\\Lei\\Processed_data_results\\'
save_dir = 'C://Users/zhu\Dropbox/2022.PulseSplittingPaper/Results/05_Photobleaching_done/'

folder_name = ['raw data']
plt.figure()
# Define a list of colors for plotting

for folder in folder_name:
    folder_path = os.path.join(data_dir, folder)
    
    # Get all CSV files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    added_labels = set()

    for index, data_name in enumerate(files):
        file_path = os.path.join(folder_path, data_name)
        data_file = read_csv_file(file_path)
        file_key = get_column_names_from_data(data_file)
        num_rows = len(data_file)
        
        if num_rows == 0 or len(file_key) == 0:
            continue
        
        original_data = np.empty((num_rows, len(file_key)))
        
        for index_y, column_name in enumerate(file_key):
            column_data = read_column_from_data(data_file, column_name)
            original_data[:, index_y] = column_data

        label = data_name
        plt.plot(np.linspace(1, 75, num_rows), np.mean(original_data, axis=1), '.', color=colors[index], label=label)

plt.legend()
plt.show() 
#%%
from sklearn import preprocessing
plt.figure()
plt.rcParams['figure.figsize'] = [6,5]
# order = [1,2,3,0]
# # Order and colors
order = [0, 1, 2, 3]
# colors = plt.cm.tab20(np.linspace(0, 0.2, len(order)))
test_name_pool = ['8MHz','4x2MHz','2x2x2MHz','4MHz']
for folder in folder_name:
    folder_path = os.path.join(data_dir, folder)
    
    # Get all CSV files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Generate colors from Pastel1 colormap based on number of files
    colors = plt.cm.tab20(np.linspace(0, 0.2, len(files)))
    
    for index, data_name in enumerate(files):
        file_path = os.path.join(folder_path, test_name_pool[index]+'.csv')
        # file_path = os.path.join(folder_path, data_name)
        data_file = read_csv_file(file_path)
        file_key = get_column_names_from_data(data_file)
        num_rows = len(data_file)
        
        if num_rows == 0 or len(file_key) == 0:
            continue
        
        original_data = np.empty((num_rows, len(file_key)))
        
        for index_y, column_name in enumerate(file_key):
            column_data = read_column_from_data(data_file, column_name)
            original_data[:, index_y] = column_data
            # original_data[:, index_y] = preprocessing.normalize(np.array(column_data))
        # Iterate over each column index
        for index_yin in range(original_data.shape[1]):
            # Extract column data
            column_data = original_data[:, index_yin]-100
            
            # Subtract 100 and normalize the column data
            # avg = column_data.mean()
            # std = column_data.std()
            # normalized = (column_data - avg) / std
            
            a = column_data.min()
            b = column_data.max()
            normalized = (column_data-a)/(b-a)
            # Update original_data with the normalized column data
            original_data[:, index_yin] = normalized
        
        # Calculate mean and standard error (assuming standard error of the mean)
        mean_values = np.mean(original_data, axis=1)
        std_err = np.std(original_data, axis=1) / np.sqrt(original_data.shape[1])  # Standard error of the mean
        
        label = test_name_pool[index]
        
        # Plotting error bars with Pastel1 colors
        plt.errorbar(np.linspace(1, num_rows, num_rows)*0.26, mean_values, yerr=std_err, alpha=1.0, fmt=':', capsize=3, capthick=1, color=colors[order[index]], label=label)
        plt.fill_between(np.linspace(1, 75, num_rows)*0.26, mean_values - std_err, mean_values + std_err, alpha=0.1, color=colors[order[index]])
        # Add text annotation for each errorbar entry
        # plt.text(10, 0.06, f'$\hat{{\mu}}$ = {mean_values[-1]:.2f}', ha='center', va='bottom', color=colors[index], fontsize=20)
    
    # Customize plot parameters
    plt.xlim((0.5, num_rows*0.26))  # Adjust x-axis limit
    plt.legend()  # Show legend
    #get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.xlabel('T[s] ',fontsize=14)  # X-axis label
    plt.ylabel('2P signal(a.u.)',fontsize=14)  # Y-axis label
    # plt.xlabel('T in s ',fontsize=14,style='italic')  # X-axis label
    # plt.ylabel('2P signal(a.u.)',fontsize=14,style='italic')  # Y-axis label
    # plt.title('Plot with Error Bars')  # Plot title
    
    # Display the plot
    plt.show()
    plt.tight_layout() 
#%%

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
from scipy import stats

def read_csv_file(file_path):
    """
    Read a CSV file and return its contents as a list of dictionaries.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    list: A list of dictionaries, each representing a row in the CSV file.
    """
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Get the headers from the first row
        return [dict(zip(headers, row)) for row in reader]
    
def read_pd_csv_file(file_path):
    """
    Read a CSV file and return its column names as a list of numeric values.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    list: A list of numeric values representing the column names.
    """
    # Read the CSV file using pandas
    input_df = pd.read_csv(file_path)
    
    # Extract the column names
    column_names = input_df.columns
    
    # Convert the column names from strings to numeric values
    numeric_values = [float(col.strip()) for col in column_names]
    
    return numeric_values  

# Provided test names
# test_name_pool = ['8MHz', '2x2x2MHz', '4MHz', '4x2MHz']
test_name_pool = ['8MHz','4x2MHz','2x2x2MHz','4MHz']

folder_path = 'C://Users/zhu/Dropbox/2022.PulseSplittingPaper/Results/05_Photobleaching_done/'

# file = os.path.join(folder_path, test_name_pool[1]+'points.csv')

# Example points_dict (replace with your actual data)
points_dict = {
    'All_8MHzpoints': read_pd_csv_file(os.path.join(folder_path, test_name_pool[0]+'points.csv')),  # Replace with your data
    'All_4x2MHzpoints': read_pd_csv_file(os.path.join(folder_path, test_name_pool[1]+'points.csv')),
    'All_2x2x2MHzpoints': read_pd_csv_file(os.path.join(folder_path, test_name_pool[2]+'points.csv')),  # Replace with your data
    'All_4MHzpoints': read_pd_csv_file(os.path.join(folder_path, test_name_pool[3]+'points.csv'))  # Replace with your data
      # Replace with your data
}
                                         
# Extract all points for the boxplot
all_points = [np.array(points_dict['All_' + label + 'points']) for label in test_name_pool]

# Order and colors
order = [0, 1, 2, 3]
colors = plt.cm.tab20(np.linspace(0, 0.2, len(order)))

# Setting up matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

# Creating the figure and axis
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # Adjusted to give space for labels

# Box plot for the data distributions
ax1.boxplot(all_points, positions=range(len(test_name_pool)), widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='lightgray', color='black', alpha=0.4),
            medianprops=dict(color='r'))

# Loop through the test_name_pool and plot individual points and means with error bars
mean_values = []

for i, label in enumerate(test_name_pool):
    key = 'All_' + label + 'points'
    points = np.array(points_dict[key])
    
    # Calculate mean and standard deviation
    mean_value = np.mean(points)
    std_value = np.std(points)
    
    # Store mean values for later use
    mean_values.append(mean_value)
    
    # Scatter plot for individual points
    x_vals = np.full(points.shape, i)+ np.random.uniform(-0.1, 0.1, size=len(points))  # small random jitter  # Create an array of the same value i, for x positions
    
    ax1.scatter(x_vals, points, color=colors[order[i]], alpha=0.4, label=label if i == 0 else "")  # Add label only for the first set for the legend

    # Scatter plot for the mean
    ax1.scatter(i, mean_value, color='k', zorder=5, alpha=1.0)
    
    # Annotate the mean value
    ax1.text(i, mean_value, f'$\hat{{\mu}}$ ={mean_value:.3f}', color='k', ha='right', va='bottom')
    # ax1.text(i, mean_value, f'$\\hat{{\\mu}}_{{mean}}$ = {mean_value:.3f}', color='k', ha='right', va='bottom')

    # ax1.text(i, mean_value + std_value, f'$\hat{{\n}}$ ={points.shape:.2f}', color='k', ha='left', va='bottom')

# Setting x-ticks and labels
ax1.set_xticks(range(len(test_name_pool)))
ax1.set_xticklabels(test_name_pool, rotation=0, fontsize = 12,ha='center')
# ax1.set_title("Bleaching Rates",fontsize = 16)
ax1.set_ylabel('Decay (a.u.)',fontsize = 14)

# Setting y-axis limit
ax1.set_ylim([0, 2.0 * max([np.max(points) for points in all_points])])

# Pairwise comparisons
comparisons = [("8MHz", "4x2MHz"), ("8MHz", "2x2x2MHz"), ("8MHz", "4MHz")]
Height = [0.005,0.026,0.03]

for i, label in enumerate(comparisons):
    group1 = points_dict['All_' + label[0] + 'points']
    group2 = points_dict['All_' + label[1] + 'points']
    t_stat, p_val = stats.ttest_ind(group1, group2)
    
    # Find y-position for the annotation
    y_max = max(np.max(group1), np.max(group2))
    x1, x2 = test_name_pool.index(label[0]), test_name_pool.index(label[1])
    y, h, col = y_max + Height[i], 0.01, 'k'
    
    # Plot the line and annotate p-value
    ax1.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color=col)
    ax1.text((x1 + x2) * 0.5, y + h, f"$p$ = {p_val:.3f}", ha='center', va='bottom', color=col)
# Setting y-axis limit
ax1.set_ylim([0, 1.5 * max([np.max(points) for points in all_points])])
# # Setting x-ticks and labels
# ax1.set_xticks(range(len(test_name_pool)))
# ax1.set_xticklabels(test_name_pool, rotation=0, fontsize=12, ha='center')
# ax1.set_title("Bleaching Rates", fontsize=16)
# ax1.set_ylabel('Decay (a.u.)', fontsize=14)

# # Setting y-axis limit
# ax1.set_ylim([0, 1.1 * max([np.max(points) for points in all_points])])

# plt.show()    