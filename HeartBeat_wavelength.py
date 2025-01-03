# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:56:30 2024

@author: zhu
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
from scipy import stats
import seaborn as sns

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
test_name_pool = ['1030','1070']

folder_path = 'C://Users/zhu/Dropbox/2022.PulseSplittingPaper/Results/06_SL_done/'
# file = os.path.join(folder_path, test_name_pool[1]+'points.csv')

# Example points_dict (replace with your actual data)
points_dict = {
    'slope_1030': read_pd_csv_file(os.path.join(folder_path, 'slope'+test_name_pool[0]+'.csv')),  # Replace with your data
    'slope_1070': read_pd_csv_file(os.path.join(folder_path, 'slope'+test_name_pool[1]+'.csv'))  # Replace with your data
      # Replace with your data
}
                                         
# Extract all points for the boxplot
all_points = [100*np.array(points_dict['slope_' + label]) for label in test_name_pool]

#%%
# Order and colors
order = [0, 1]
colors = plt.cm.tab20(np.linspace(0, 0.1, len(order)))

# Setting up matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

# Creating the figure and axis
fig = plt.figure(figsize=(4.5, 4))
ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # Adjusted to give space for labels

# Box plot for the data distributions
ax1.boxplot(all_points, positions=range(len(test_name_pool)), widths=0.3, patch_artist=True,
            boxprops=dict(facecolor='lightgray', color='black', alpha=0.4),
            medianprops=dict(color='r'))

# Loop through the test_name_pool and plot individual points and means with error bars
mean_values = []

for i, label in enumerate(test_name_pool):
    key = 'slope_' + label
    points =  100*np.array(points_dict[key])
    
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
    ax1.text(i, mean_value, f'$\hat{{\mu}}$ ={mean_value:.4f}', color='k', ha='center', va='bottom')
    ax1.text(i, 7, f'$n$ = {len(points):.0f}', color='k', ha='center', va='bottom')
    # ax1.text(i, mean_value, f'$\\hat{{\\mu}}_{{mean}}$ = {mean_value:.3f}', color='k', ha='right', va='bottom')

    # ax1.text(i, mean_value + std_value, f'$\hat{{\n}}$ ={points.shape:.2f}', color='k', ha='left', va='bottom')

# Setting x-ticks and labels
ax1.set_xticks(range(len(test_name_pool)))
ax1.set_xticklabels(test_name_pool, rotation=0, fontsize = 12,ha='center')
# ax1.set_title("Bleaching Rates",fontsize = 16)
ax1.set_ylabel(r'$\Delta$HBR/HBR$_0$ in %',fontsize = 14)
ax1.set_xlabel('Wavelenght (nm)',fontsize = 14)

# Setting y-axis limit
ax1.set_ylim([6.8, 1.05 * max([np.max(points) for points in all_points])])
#%%
order = [0, 1]
colors = plt.cm.tab20(np.linspace(0, 0.1, len(order)))
# Setting up matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

# Creating the figure and axis
fig = plt.figure(figsize=(4.72, 4))

sns.violinplot(data=all_points, inner="quart", palette=colors)
# Calculate and annotate mean values
for i, points in enumerate(all_points):
    mean_value = np.mean(points)
    # Annotate the mean value on the plot
    plt.text(i, mean_value, f'$\hat{{\mu}}$ ={mean_value:.2f}', color='white', ha='center', va='bottom')
    plt.text(i, 3.5, f'$N$ = {len(points):.0f}', color='k', ha='center', va='bottom')
    
# Set the x-tick labels and other plot settings
plt.xticks(ticks=np.arange(len(test_name_pool)), labels=test_name_pool, rotation=0, fontsize=10)
plt.ylabel(r'$\Delta$HBR/HBR$_0$ in %', fontsize=12)
plt.xlabel('Wavelength (nm)', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()

fig = plt.gcf()  # Get current figure
fig_size = fig.get_size_inches()  # Get figure size in inches
print(f"Current figure size: {fig_size} inches")

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
test_name_pool = ['1030','1070']

folder_path = 'C://Users/zhu/Dropbox/2022.PulseSplittingPaper/Results/06_SL_done/'
# file = os.path.join(folder_path, test_name_pool[1]+'points.csv')

# Example points_dict (replace with your actual data)
points_dict = {
    'profile_1030': read_csv_file(os.path.join(folder_path, 'profile_'+test_name_pool[0]+'nm.csv')),  # Replace with your data
    'profile_1070': read_csv_file(os.path.join(folder_path, 'profile_'+test_name_pool[1]+'nm.csv'))  # Replace with your data
      # Replace with your data
}
                                         
# Extract all points for the boxplot
all_points = [np.array(points_dict['profile_' + label]) for label in test_name_pool]

# Order and colors
order = [0, 1]
colors = plt.cm.tab20(np.linspace(0, 0.1, len(order)))

# Setting up matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

# Creating the figure and axis
fig = plt.figure(figsize=(4.8, 3.84))
# ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # Adjusted to give space for labels

# # Box plot for the data distributions
# ax1.boxplot(all_points, positions=range(len(test_name_pool)), widths=0.3, patch_artist=True,
#             boxprops=dict(facecolor='lightgray', color='black', alpha=0.4),
#             medianprops=dict(color='r'))

# # Loop through the test_name_pool and plot individual points and means with error bars
# mean_values = []

for i, label in enumerate(test_name_pool):
    key = 'profile_' + label
    points = np.array(points_dict[key])
    # Extracting x and y data from points
    x_data = []
    y_data = []

    for point in points:
        key = list(point.keys())[0]  # Extract the single key
        value = list(point.values())[0]  # Extract the single value
        x_data.append(float(key))
        y_data.append(float(value))

    label = '$\lambda_{illumination}$='+label+'nm'
    if i == 0:
        y_data_offset = [y - 0.05 if y <= 2.5 else y + 0.05 for y in y_data]  # Subtract 0.1 from each y_data point
        plt.plot(0.65*np.linspace(1, len(points), len(points)), y_data_offset, ':',linewidth = 4,color=colors[i],label=label)
    if i == 1:
        y_data_offset = [y - 0 for y in y_data]  # Subtract 0.1 from each y_data point
        plt.plot(0.65*np.linspace(1, len(points), len(points)), y_data_offset, ':',linewidth = 4,color=colors[i],label=label)
    # Scatter plot for individual points
    # x_vals = np.full(points.shape, i)+ np.random.uniform(-0.1, 0.1, size=len(points))  # small random jitter  # Create an array of the same value i, for x positions
    
    # ax1.scatter(x_vals, points, color=colors[order[i]], alpha=0.4, label=label if i == 0 else "")  # Add label only for the first set for the legend

    # # Scatter plot for the mean
    # ax1.scatter(i, mean_value, color='k', zorder=5, alpha=1.0)
    
    # # Annotate the mean value
    # ax1.text(i, mean_value, f'$\hat{{\mu}}$ ={mean_value:.4f}', color='k', ha='center', va='bottom')
    # ax1.text(i, 7, f'$n$ = {len(points):.0f}', color='k', ha='center', va='bottom')
    # ax1.text(i, mean_value, f'$\\hat{{\\mu}}_{{mean}}$ = {mean_value:.3f}', color='k', ha='right', va='bottom')

    # ax1.text(i, mean_value + std_value, f'$\hat{{\n}}$ ={points.shape:.2f}', color='k', ha='left', va='bottom')
plt.ylabel('HBR(Hz)', fontsize=12)
plt.xlabel('T[s] ',fontsize=12)  # X-axis label
plt.legend(fontsize=10)  # Show legend
fig = plt.gcf()  # Get current figure
fig_size = fig.get_size_inches()  # Get figure size in inches
print(f"Current figure size: {fig_size} inches")

# # Setting x-ticks and labels
# ax1.set_xticks(range(len(test_name_pool)))
# ax1.set_xticklabels(test_name_pool, rotation=0, fontsize = 12,ha='center')
# # ax1.set_title("Bleaching Rates",fontsize = 16)
# ax1.set_ylabel(r'$\Delta$HBR/HBR$_0$ in %',fontsize = 14)
# ax1.set_xlabel('Wavelenght (nm)',fontsize = 14)

# # Setting y-axis limit
# ax1.set_ylim([6.8, 1.05 * max([np.max(points) for points in all_points])])

# # Pairwise comparisons
# comparisons = [("8MHz", "4x2MHz"), ("8MHz", "2x2x2MHz"), ("8MHz", "4MHz")]
# Height = [0.005,0.026,0.03]

# for i, label in enumerate(comparisons):
#     group1 = points_dict['All_' + label[0] + 'points']
#     group2 = points_dict['All_' + label[1] + 'points']
#     t_stat, p_val = stats.ttest_ind(group1, group2)
    
#     # Find y-position for the annotation
#     y_max = max(np.max(group1), np.max(group2))
#     x1, x2 = test_name_pool.index(label[0]), test_name_pool.index(label[1])
#     y, h, col = y_max + Height[i], 0.01, 'k'
    
#     # Plot the line and annotate p-value
#     ax1.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color=col)
#     ax1.text((x1 + x2) * 0.5, y + h, f"$p$ = {p_val:.3f}", ha='center', va='bottom', color=col)
# # Setting y-axis limit
# ax1.set_ylim([0, 1.5 * max([np.max(points) for points in all_points])])
# # # Setting x-ticks and labels
# # ax1.set_xticks(range(len(test_name_pool)))
# # ax1.set_xticklabels(test_name_pool, rotation=0, fontsize=12, ha='center')
# # ax1.set_title("Bleaching Rates", fontsize=16)
# # ax1.set_ylabel('Decay (a.u.)', fontsize=14)

# # # Setting y-axis limit
# # ax1.set_ylim([0, 1.1 * max([np.max(points) for points in all_points])])

# # plt.show()    