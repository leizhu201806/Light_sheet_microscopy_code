# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:29:57 2024

@author: zhu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:48:00 2023

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
from numpy import asarray as ar
# from scipy import exp


#binning data
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1],)
    return arr.reshape(shape).mean(-1).mean(1)

# Returns a list of all the folders inside the directories
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


#%%
date_arr = ['21-12-07']
for date in date_arr:
    print("############### #" + date)
    # Where my data is stored.
    # data_dir = 'O://Lei/Processed data/' +date + '/'
    data_dir = 'F://2_singal_intensity/' +date + '/'
    # # Where I want to save it.
    # Tiff_location = '//129.104.18.21/Public/Lei/Processed data/' + date + '/'

    # # If the folder doesn't exist, make it. The folder name is the same as the one entered in date_arr
    # if not os.path.exists(Tiff_location):
    #     os.makedirs(Tiff_location)
    Non_zero_data = []
    Seperation = []
    # Go through the list of folders inside the data_dir folder
    print(get_immediate_subdirectories(data_dir))
    # Seperation=np.zeros(np.size(get_immediate_subdirectories(data_dir)))
    for data_name in get_immediate_subdirectories(data_dir):
    # if len(get_immediate_subdirectories(data_dir)) >= 7:
    #     data_name = get_immediate_subdirectories(data_dir)[2]  # Get the seventh subdirectory (index 6)    
        print(data_name)

        plt.close('all')
        plt.rcParams['figure.figsize'] = [10,10]
        #First, open the whole stack and average every 10 images. Use the center image of the stack for analysis
        #It is assumed the beads in the center of the stack are the ones best in focus
        # DIR = 'O://Lei/Processed data/PSF NA0.04'
        
        # data_name = 'agarose+redfluo V3'
        #%%
        import pandas as pd
        import xlwings as xw
        
        path = data_dir+'/'+data_name+'/'+'Param_Acquisition'+'.xlsx'
        path2 = data_dir+'/'+data_name+'\\renamed_'+'Param_Acquisition'+'.xlsx'  
        
        while True:
            try:
                df = pd.read_excel(path, engine='openpyxl')
            except Exception as e:
                print("Failed to open workbook; error: ")
                print(e)
                wingsbook = xw.Book(path)
                wingsapp = xw.apps.active
                wingsbook.save(path2)
                wingsapp.quit()
                path = path2
            else:
                break
        #%%
        radius_threshold = 3
        z_stack = int(df.iloc[0]['Nimg Z']) # Directly take the parameter form the .XLSX file.
        
        # DIR = 'M:\\2016_2020_ZebrafishHeartProject\\Dale - Selected Data\\22-07-25 opera psf'
        
        # data_name = 'red beads agarose 4MHz 1070nm 50mW 1dg 1dp 10z psf'
        
        Vertical = 1
        Horizontal = 0
        
        img_arr = np.moveaxis(io.imread(data_dir+'/'+data_name+'/'+data_name+'.tiff'),0,2)

        # DIR = 'M:\\2016_2020_ZebrafishHeartProject\\Dale - Selected Data\\22-07-25 opera psf'
        
        # data_name = 'red beads agarose 4MHz 1070nm 50mW 1dg 1dp 10z psf'
        
        test_name_pool = list(range(0,1,1)) 
        distribution_map = np.zeros([img_arr.shape[0], img_arr.shape[1],np.size(test_name_pool)])
        
        # img_arr = img_arr - np.min(img_arr)
        num_images = img_arr.shape[2]
        
        bin_size = int(df.iloc[0]['Nimg T'])
        
        num_bins = img_arr.shape[2] // bin_size
        
        # Create an empty array to store the binned image
        binned_img_arr = np.zeros((img_arr.shape[0], img_arr.shape[1], num_bins))
        
        # Bin the image along the Z-axis
        for i in range(num_bins):
            # Define the start and end indices for each bin
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size-1
            
            # Slice the image array to select the frames for the current bin
            bin_frames = img_arr[:, :, start_idx:end_idx]
            
            # Compute the mean along the Z-axis for the current bin
            binned_frame = np.mean(bin_frames, axis=2)
            
            # Store the binned frame in the binned image array
            binned_img_arr[:, :, i] = binned_frame
        
        img_arr = binned_img_arr -100
        num_images = img_arr.shape[2]
        
        binfac = 1
        bnx, bny = img_arr.shape[0]//binfac, img_arr.shape[1]//binfac
        Bin_img_arr = np.empty((bnx, bny,num_images))
        
        for ii in range(0,img_arr.shape[2]):
            Bin_img_arr[:,:,ii] = rebin(img_arr[:,:,ii], (bnx, bny))
        
        flattened_data = Bin_img_arr.flatten()

        # Create a mask where values >= 20 are True
        mask = flattened_data >= 0

        # Apply the mask to filter out values < 20
        filtered_data = flattened_data[mask]    
        
        Bin_img_arr.flatten()
            
       
        Non_zero_data.append(Bin_img_arr.flatten())
        Seperation.append(np.size(Bin_img_arr.flatten()))
#%%       
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example setup for Non_zero_data and Seperation
# Non_zero_data should be a list of arrays
# Seperation should be a list of integers indicating the number of elements in each group

# Flatten Non_zero_data and generate labels
combined_non_zero_data = []
group_labels = []

# Flatten Non_zero_data and create corresponding labels
for i, data in enumerate(Non_zero_data):
    combined_non_zero_data.extend(data)
    group_labels.extend([f'Group {i+1} ({Seperation[i]})'] * len(data))

# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=group_labels, y=combined_non_zero_data)
plt.xlabel('Group (based on Seperation)')
plt.ylabel('Non-zero Intensity Data')
plt.title('Violin Plot of Non-zero Intensity Data Grouped by Seperation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display the plot
plt.show()
    
#%%        
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming Non_zero_data and Seperation are already populated as per your code
# Non_zero_data should be a list of intensity values
# Seperation should be a list of corresponding separation values (e.g., number of non-zero intensities)

# Flatten Non_zero_data and generate labels
combined_non_zero_data = []
combined_group_labels = []

# Combine every 3 groups
group_counter = 0
for i in range(0, len(Seperation), 3):
    # Define the label for the combined group
    combined_label = f'Combined Group {group_counter+1}'
    
    # Combine the Non_zero_data and assign to the combined group
    combined_data = []
    for j in range(i, min(i+3, len(Seperation))):
        combined_data.extend(Non_zero_data[j])
        combined_group_labels.extend([combined_label] * len(Non_zero_data[j]))

    combined_non_zero_data.extend(combined_data)
    group_counter += 1

# Create a violin plot with the combined groups
plt.figure(figsize=(12, 8))
sns.violinplot(x=combined_group_labels, y=combined_non_zero_data )
plt.xlabel('Combined Groups')
plt.ylabel('Non-zero Intensity Data')
plt.title('Violin Plot of Non-zero Intensity Data Grouped by Every 3 Combined Groups')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.yscale('log')
# Display the plot
plt.show()

        
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Assuming Non_zero_data and Seperation are already populated
order = [0, 1, 2, 3, 4, 5]
colors = plt.cm.tab20(np.linspace(0, 0.36, len(order)))
# Flatten Non_zero_data and generate labels
combined_non_zero_data = []
combined_group_labels = []

# Combine every 3 groups
group_counter = 0
for i in range(0, len(Seperation), 3):
    # Define the label for the combined group
    combined_label = f'Combined Group {group_counter+1}'
    
    # Combine the Non_zero_data and assign to the combined group
    combined_data = []
    for j in range(i, min(i+3, len(Seperation))):
        combined_data.extend(Non_zero_data[j])
        combined_group_labels.extend([combined_label] * len(Non_zero_data[j]))

    combined_non_zero_data.extend(combined_data)
    group_counter += 1

# Define the desired order
order = [0, 1, 2, 4, 3, 5]
reordered_labels = []
reordered_data = []

# Create a mapping from group names to indices
# group_mapping = {f'Combined Group {i+1}': i for i in range(len(order))}
newgroup_mapping = ['10MHz','2MHz','4MHz','2x2MHz','8MHz','2x2x2MHz']
# Reorder labels and data according to the specified order
for i,idx in enumerate(order):
    label = newgroup_mapping[i]
    reordered_labels.extend([label] * combined_group_labels.count(label))
    reordered_data.extend([combined_non_zero_data[i] for i in range(len(combined_group_labels)) if combined_group_labels[i] == label])

# Create a DataFrame for Seaborn
df = pd.DataFrame({
    'Group': reordered_labels,
    'Intensity': reordered_data
})

# Create a violin plot with the reordered groups
plt.figure(figsize=(8, 6))
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
sns.violinplot(x='Group', y='Intensity', data=df, inner="quartile", palette=colors)
plt.yscale('log')
plt.ylabel('Intensity (a.u.) (log)',fontsize = 12)
plt.xlabel('Repetition rate',fontsize = 12)
# plt.title('Violin Plot of Non-zero Intensity Data Grouped by Every 3 Combined Groups')
plt.xticks(rotation=0, ha='center')
plt.tight_layout()

# Display the plot
plt.show()


        
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming Non_zero_data and Seperation are already populated as per your code
order = [0, 1, 2, 3, 4, 5]
colors = plt.cm.tab20(np.linspace(0, 0.36, len(order)))
# Create new group labels by combining every 3 original groups
combined_group_labels = []
combined_non_zero_data = []
group_means = []
group_data_list = []
group_counter = 0

for i in range(0, len(Seperation), 3):
    # Define the label for the combined group
    combined_label = f'Combined Group {group_counter+1}'

    # Combine the Non_zero_data and assign to the combined group
    group_data = []
    for j in range(i, min(i+3, len(Seperation))):
        combined_group_labels.extend([combined_label] * Seperation[j])
        group_data.extend(Non_zero_data[sum(Seperation[:j]):sum(Seperation[:j+1])])

    combined_non_zero_data.extend(group_data)
    group_data_list.append(group_data)
    group_means.append(np.mean(group_data))  # Calculate the mean for this combined group
    group_counter += 1

# Create a violin plot with the combined groups
order = [0, 1, 2, 4, 3, 5]
reordered_labels = []
reordered_data = []
reordered_means = []

test_name_pool = ['10MHz','2MHz','4MHz','2x2MHz','8MHz','2x2x2MHz']
for i,idx in enumerate(order):
    label = test_name_pool[i]
    reordered_labels.extend([label] * len(group_data_list[idx]))
    reordered_data.extend(group_data_list[idx])
    reordered_means.append(group_means[idx])

# Create a violin plot with the reordered groups

plt.figure(figsize=(8, 6))
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
# sns.violinplot(x=reordered_labels, y=reordered_data, inner="quartile")  # inner=None to hide default statistics
sns.violinplot(x=reordered_labels, y=reordered_data, inner="quartile", palette=colors)  # inner=None to hide default statistics
plt.yscale('log')
# Overlay mean values with a pointplot
# sns.pointplot(x=reordered_labels, y=reordered_data, 
#               estimator=np.mean, ci=None, color='red', markers='o', linestyles='')
# test_name_pool = ['8MHz', '2x2x2MHz', '4MHz', '4x2MHz']

# Annotate the precise mean values on the plot
# for i, mean_val in enumerate(reordered_means):
#     plt.text(i, mean_val, f'{mean_val:.2f}', color='black', ha='right', va='center_baseline')
# Annotate mean values at the top of the figure
for i, mean_val in enumerate(reordered_means):
    plt.text(i, plt.gca().get_ylim()[1] * 1.01, f'{mean_val:.2f}', 
             color='black', ha='center', va='bottom', fontsize=10)

# plt.xlabel('Combined Groups')
plt.ylabel('Intensity (a.u.)',fontsize = 12)
plt.xlabel('Repetition rate',fontsize = 12)
# plt.title('Violin Plot of Non-zero Intensity Data Grouped by Combined Seperation with Mean Values Marked')
plt.xticks(rotation=0, ha='center')
plt.tight_layout()

# Display the plot
plt.show()