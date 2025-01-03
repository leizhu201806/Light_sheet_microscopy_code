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
DIR = 'F://2_singal_intensity/21-12-07/'
Main_name_pool = ['2MHz','4MHz','8MHz','10MHz','BS_2MHz','BS_4MHz']
# Main_name_pool = ['2MHz']
Non_zero_data = []
Seperation = []
for index_file, test_name in enumerate(Main_name_pool):
    data_arr = []

    # Get all folders at the requested rep. rate
    for val in [x[0] for x in os.walk(DIR)]:
        if test_name in val:
            if os.path.isdir(os.path.join(DIR, val)):
                data_arr.append(os.path.basename(val))

    # Non_zero_data = []
    # Seperation = []
    for data_name in data_arr[0:3]:
        print("############### #" + data_name)
        # Where my data is stored.
        # # Where I want to save it.
        # Tiff_location = '//129.104.18.21/Public/Lei/Processed data/' + date + '/'
    
        # # If the folder doesn't exist, make it. The folder name is the same as the one entered in date_arr
        # if not os.path.exists(Tiff_location):
        #     os.makedirs(Tiff_location)

        # Go through the list of folders inside the data_dir folder

        # if len(get_immediate_subdirectories(data_dir)) >= 7:
        # data_name = get_immediate_subdirectories(data_dir)[2]  # Get the seventh subdirectory (index 6)    
 

        plt.close('all')
        plt.rcParams['figure.figsize'] = [10, 10]
        #First, open the whole stack and average every 10 images. Use the center image of the stack for analysis
        #It is assumed the beads in the center of the stack are the ones best in focus
        # DIR = 'O://Lei/Processed data/PSF NA0.04'
        
        # data_name = 'agarose+redfluo V3'
        #%
        import pandas as pd
        import xlwings as xw
        
        path = DIR+'/'+data_name+'/'+'Param_Acquisition'+'.xlsx'
        path2 = DIR+'/'+data_name+'\\renamed_'+'Param_Acquisition'+'.xlsx'  
        
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
        #%
        radius_threshold = 3
        z_stack = int(df.iloc[0]['Nimg Z']) # Directly take the parameter form the .XLSX file.
        
        # DIR = 'M:\\2016_2020_ZebrafishHeartProject\\Dale - Selected Data\\22-07-25 opera psf'
        
        # data_name = 'red beads agarose 4MHz 1070nm 50mW 1dg 1dp 10z psf'
        
        Vertical = 1
        Horizontal = 0
        # img_arr = io.imread(os.path.join(DIR, data_name, data_name + '.tiff'))[:, :, 100:-100]
        img_arr = np.moveaxis(io.imread(os.path.join(DIR, data_name, data_name + '.tiff')),0,2)

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
            
            # selected_ref = 14
            #Point tracking will take place on the middle image. Beads there are assumed to be well centered
            
        Fwhm_selected = []    
        for trial, selected_ref in enumerate(test_name_pool):    
            img_grayscale = Bin_img_arr[:,:,selected_ref]
            
            # img_grayscale = np.mean(Bin_img_arr,1)
            # fig,ax = plt.subplots()
            # plt.imshow(img_grayscale, cmap='gray', vmax = 0.5*np.max(img_grayscale))
            # plt.colorbar()
            
            # Blur the image to get rid of weird pixels
            blurred = cv2.GaussianBlur(img_grayscale, (11, 11), 0)
            mean = np.mean(blurred)
            std = np.std(blurred)
            
            # fig = plt.figure()
            # plt.imshow(blurred, cmap='gray', vmax = mean+std)
            
            #Take the threshold of the image, based on mean+std/num of all the pixels to remove the background
            mean = np.mean(blurred)
            std = np.std(blurred)
            
            thresh = cv2.threshold(blurred, mean+std/10, 255, cv2.THRESH_BINARY)[1]
            # fig,ax = plt.subplots()
            # plt.imshow(thresh, cmap='gray')
            
            #Find each unique blob, only if it's big enough. IE, it needs to be more than 3 pixels
            labels = measure.label(thresh, connectivity=2.0, background=0)
            mask = np.zeros(thresh.shape, dtype="uint8")
            
            
            plt.figure(figsize=(8, 6))
            
            # First subplot: normalized image
            plt.subplot(1, 2, 1)
            plt.imshow(blurred, cmap='gray', vmax = mean+std)
            cbar1 = plt.colorbar(shrink=0.4, aspect=10, pad=0.02)  # Adjust colorbar size
            cbar1.ax.tick_params(labelsize=10)  # Set fontsize for colorbar ticks
            plt.title('Normalized Image', fontsize=14)
            plt.xlabel('X-axis label', fontsize=12)
            plt.ylabel('Y-axis label', fontsize=12)
            plt.tick_params(axis='both', labelsize=10)
            
            # Second subplot: labels
            plt.subplot(1, 2, 2)
            plt.imshow(labels)
            cbar2 = plt.colorbar(shrink=0.4, aspect=10, pad=0.02)  # Adjust colorbar size
            cbar2.ax.tick_params(labelsize=10)  # Set fontsize for colorbar ticks
            plt.title('Labels', fontsize=14)
            plt.xlabel('X-axis label', fontsize=12)
            plt.ylabel('Y-axis label', fontsize=12)
            plt.tick_params(axis='both', labelsize=10)
            
            # Show the plot with tight layout
            plt.tight_layout()
            plt.show()
            
            for label in np.unique(labels):
                if label == 0:
                    continue
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)
                # fig,ax = plt.subplots()
                # plt.imshow(labelMask, cmap='gray', vmax = np.max(labelMask))
                # plt.colorbar()
                if numPixels > 100 and numPixels < np.shape(img_arr)[0] - 10:
                    mask = cv2.add(mask, labelMask)
            
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = contours.sort_contours(cnts)[0]
            
            #Label all the beads that were found
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(img_grayscale, cmap='gray', vmax = 0.5*np.max(img_grayscale))
            for (i, c) in enumerate(cnts):
                ((cX, cY), radius) = cv2.minEnclosingCircle(c)
                if (cX > 20 and cX < np.shape(img_arr)[0]-20) and (cY > 20 and cY < np.shape(img_arr)[1]-20) and radius >=radius_threshold:
                    ax.add_patch(patches.Circle((int(cX), int(cY)), radius=int(radius), color='r', fill=0))
            
            fig = plt.figure()
            projected_intensity = np.zeros((len(cnts), int(num_images)))
            # Create a mask for the circle
            tt = 0
            for i, c in enumerate(cnts):
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                radius = int(radius)

                
                # Check if the circle is within the image boundaries
                if (x > 20 and x < np.shape(img_arr)[1] - 20) and (y > 20 and y < np.shape(img_arr)[0] - 20 and radius >=radius_threshold):
                    for j in range(int(num_images)):
                        mask = np.zeros(img_arr[:, :, j].shape, dtype=np.uint8)
                        cv2.circle(mask, center, radius, 1, thickness=-1)  # Create a filled circle mask
                        # fig,ax = plt.subplots()
                        # plt.imshow(img_arr, cmap='gray', vmax = .01*np.max(img_arr))
                        # plt.colorbar()
                        # Calculate the mean intensity within the circle
                        mean_intensity = cv2.mean(img_arr[:, :, j], mask=mask)[0]
                        projected_intensity[i, j] = mean_intensity
                        if projected_intensity[i, j] >0:
                            tt = tt+1
                            Non_zero_data.append(projected_intensity[i, j])
                    
                    # # Normalize the projected intensities
                    # min_intensity = np.min(projected_intensity[i, :])
                    # max_intensity = np.max(projected_intensity[i, :])
                    # if max_intensity > min_intensity:  # Avoid division by zero
                    #     projected_intensity[i, :] = (projected_intensity[i, :] - min_intensity) / (max_intensity - min_intensity)
                    
                    # # Plot the normalized projected intensity
                    # plt.plot(projected_intensity[i, :])
            Seperation.append(tt)
            print(projected_intensity.shape)
#%%       
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming Non_zero_data and Seperation are already populated as per your code
# Non_zero_data should be a list of intensity values
# Seperation should be a list of corresponding separation values (e.g., number of non-zero intensities)

# If Seperation is supposed to group Non_zero_data based on number of intensities, you can generate the corresponding group labels:
group_labels = []

# Create labels based on Seperation values
for i, count in enumerate(Seperation):
    group_labels.extend([f'Group {i+1} ({count})'] * count)

# Create a violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x=group_labels, y=Non_zero_data)
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

# Create new group labels by combining every 3 original groups
combined_group_labels = []
combined_non_zero_data = []
group_counter = 0

for i in range(0, len(Seperation), 3):
    # Define the label for the combined group
    combined_label = f'Combined Group {group_counter+1}'

    # Combine the Non_zero_data and assign to the combined group
    for j in range(i, min(i+3, len(Seperation))):
        combined_group_labels.extend([combined_label] * Seperation[j])
        combined_non_zero_data.extend(Non_zero_data[sum(Seperation[:j]):sum(Seperation[:j+1])])

    group_counter += 1

# Create a violin plot with the combined groups
plt.figure(figsize=(10, 6))
sns.violinplot(x=combined_group_labels, y=combined_non_zero_data)
plt.xlabel('Combined Groups')
plt.ylabel('Non-zero Intensity Data')
plt.title('Violin Plot of Non-zero Intensity Data Grouped by Combined Seperation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display the plot
plt.show()
        
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming Non_zero_data and Seperation are already populated as per your code

# Create new group labels by combining every 3 original groups
combined_group_labels = []
combined_non_zero_data = []
group_means = []
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
    group_means.append(np.mean(group_data))  # Calculate the mean for this combined group
    group_counter += 1

# Create a violin plot with the combined groups
plt.figure(figsize=(10, 6))
sns.violinplot(x=combined_group_labels, y=combined_non_zero_data)  # inner=None to hide default statistics

# Overlay mean values with a pointplot
sns.pointplot(x=combined_group_labels, y=combined_non_zero_data, 
              estimator=np.mean, ci=None, color='red', markers='o', linestyles='')

# Annotate the precise mean values on the plot
for i, mean_val in enumerate(group_means):
    plt.text(i, mean_val, f'{mean_val:.2f}', color='black', ha='center', va='bottom')

plt.xlabel('Combined Groups')
plt.ylabel('Non-zero Intensity Data')
plt.title('Violin Plot of Non-zero Intensity Data Grouped by Combined Seperation with Mean Values Marked')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display the plot
plt.show()
        
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming Non_zero_data and Seperation are already populated as per your code
# Order and colors
order = [0, 1, 2, 3, 4, 5]
colors = plt.cm.tab20(np.linspace(0, 0.36, len(order)))
# Create new group labels by combining every 3 original groups
combined_group_labels = []
combined_non_zero_data = []
group_means = []
group_data_list = []
group_counter = 0

for i in range(0, len(Seperation), 3):
    combined_label = f'Combined Group {group_counter}'

    # Combine the Non_zero_data and assign to the combined group
    group_data = []
    for j in range(i, min(i + 3, len(Seperation))):
        combined_group_labels.extend([combined_label] * Seperation[j])
        group_data.extend(Non_zero_data[sum(Seperation[:j]):sum(Seperation[:j + 1])])

    combined_non_zero_data.extend(group_data)
    group_means.append(np.mean(group_data))  # Calculate the mean for this combined group
    group_data_list.append(group_data)  # Store group data for reordering
    group_counter += 1

# Reorder according to the specified order [0, 1, 2, 4, 3, 5]
# order = [0, 1, 2, 4, 3, 5]
order = [2, 0, 1, 4, 3, 5]
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

plt.figure(figsize=(4.8, 3.87))
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
# sns.violinplot(x=reordered_labels, y=reordered_data, inner="quartile")  # inner=None to hide default statistics
sns.violinplot(x=reordered_labels, y=reordered_data, inner="quartile", palette=colors)  # inner=None to hide default statistics
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
plt.ylabel('2P signal (a.u.)',fontsize = 12)
plt.xlabel('Repetition rate',fontsize = 12)
# plt.title('Violin Plot of Non-zero Intensity Data Grouped by Combined Seperation with Mean Values Marked')
plt.xticks(rotation=0, ha='center')
plt.tight_layout()

# Display the plot
plt.show()
#%%
# Exclude "10MHz" data from the order and labels
order = [0, 1, 4, 3, 5]  # Exclude the index corresponding to 10MHz
reordered_labels = []
reordered_data = []
reordered_medians = []
colors = plt.cm.tab20(np.linspace(0, 0.36, 6))

# Adjust the test_name_pool to exclude "10MHz"
test_name_pool = ['2MHz', '4MHz', '2x2MHz', '8MHz', '2x2x2MHz']

# Calculate medians for each group
for i, idx in enumerate(order):
    label = test_name_pool[i]
    reordered_labels.extend([label] * len(group_data_list[idx]))
    reordered_data.extend(group_data_list[idx])
    reordered_medians.append(np.median(group_data_list[idx]))  # Calculate median

# Create a violin plot with the reordered groups
plt.figure(figsize=(4.8, 3.80))
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2

# Create violin plot
sns.violinplot(x=reordered_labels, y=reordered_data, palette=colors[1:])

# for i, mean_val in enumerate(reordered_means):
#     plt.text(i, plt.gca().get_ylim()[1] * 1.01, f'{mean_val:.2f}', 
#              color='black', ha='center', va='bottom', fontsize=10)

# Connect the median values with a line
x_positions = range(len(reordered_medians))  # x-coordinates for the groups
plt.plot(x_positions, reordered_medians, color='k', linestyle='--', linewidth=1.5, marker='o', markersize=5, label="Median")

    
plt.ylim(-50, 300)
# Add labels and formatting
plt.ylabel('2P signal (a.u.)', fontsize=12)
plt.xlabel('Repetition rate', fontsize=12)
plt.xticks(rotation=0, ha='center')

# Modify the x-tick labels with LaTeX formatting for "MHz"
x_labels = [r'$\mathit{' + label.replace("MHz", r'\ MHz') + '}$' for label in test_name_pool]
plt.gca().set_xticklabels(x_labels, fontsize=10)

plt.tight_layout()
plt.show()


#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assuming `order`, `group_data_list`, `group_means`, `colors` are defined and initialized

# Your existing code for setting up reordered labels and data
test_name_pool_label = ['10 MHz', '2 MHz', '4 MHz', '2x2 MHz', '8 MHz', '2x2x2 MHz']
reordered_labels = []
reordered_data = []
reordered_means = []

for i, idx in enumerate(order):
    label = test_name_pool_label[i]
    reordered_labels.extend([label] * len(group_data_list[idx]))
    reordered_data.extend(group_data_list[idx])
    reordered_means.append(group_means[idx])

# Plotting
plt.figure(figsize=(4.8, 3.87))
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

# Violin plot with custom color palette
sns.violinplot(x=reordered_labels, y=reordered_data, inner="quartile", palette=colors)

# Annotate mean values at the top of each group
for i, mean_val in enumerate(reordered_means):
    plt.text(i, plt.gca().get_ylim()[1] * 1.01, f'{mean_val:.2f}', 
             color='black', ha='center', va='bottom',  font = 'Calibri', fontsize=12)

# Setting axis labels
plt.ylabel('2P signal (a.u.)', font = 'Calibri',fontsize=14)
plt.xlabel('Repetition rate', font = 'Calibri' ,fontsize=14)
plt.xticks(rotation=0, ha='center')

# Modify the x-tick labels with LaTeX formatting for "MHz"
x_labels = [r'$\mathit{' + label.replace(" MHz", r'\ MHz') + '}$' for label in test_name_pool_label]
plt.gca().set_xticklabels(x_labels,fontsize=10)

plt.tight_layout()
plt.show()

