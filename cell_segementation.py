# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:43:54 2024

@author: zhu
"""

from cellpose import models
from cellpose.io import imread
from cellpose import plot, utils, io
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
from tifffile import imread
import os

def mask_overlay(img, masks, colors=None):
    """Overlay masks on image (set image to grayscale).

    Args:
        img (int or float, 2D or 3D array): Image of size [Ly x Lx (x nchan)].
        masks (int, 2D array): Masks where 0=NO masks; 1,2,...=mask labels.
        colors (int, 2D array, optional): Size [nmasks x 3], each entry is a color in 0-255 range.

    Returns:
        RGB (uint8, 3D array): Array of masks overlaid on grayscale image.
    """
    if colors is not None:
        if colors.max() > 1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim > 2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)

    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:, :, 2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max() + 1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks == n + 1).nonzero()
        if colors is None:
            HSV[ipix[0], ipix[1], 0] = hues[n]
        else:
            HSV[ipix[0], ipix[1], 0] = colors[n, 0]
        HSV[ipix[0], ipix[1], 1] = 1.0
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

#%%
flile = r'C:\\Users\zhu\Desktop\Figure\Quantification\raw_data'
# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=True, model_type='cyto2')

files = [os.path.join(flile, 'Z04_4dpf_mcherry_XYTZ_1070nm_2x2x4MHz_158mWz29_r1-1.tif')]
imgs = [imread(f) for f in files]
masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0],
                                         flow_threshold=0.5, do_3D=False)
plt.figure()
plt.imshow(masks[0])

files = [os.path.join(flile, 'Z04_4dpf_mcherry_XYTZ_1070nm_2x2x4MHz_158mWz29_r1.tiff')]

image = imread(files)

image = np.moveaxis(image, 0, -1)

labels = measure.label(masks[0], connectivity=2, background=0)
Averaged_electrons = np.zeros([len(np.unique(labels)),1])
for label in np.unique(labels):
    # if label == 0:
    #     continue
    cell_data = []
    labelMask = np.zeros(masks[0].shape, dtype="uint8")
    labelMask[labels == label] = 255
    indices = np.where(labelMask == 255)
    cell_data = np.mean(image[indices[0], indices[1],:], axis=0)-200
    Averaged_electrons[label,0] = np.max(cell_data)
    if label == 2:
        plt.figure(figsize=(4.3, 3.85))

        plt.plot(cell_data/9.1,color=(104/255, 238/255, 23/255))
        print(max(cell_data)/9.1)
        plt.ylabel('Average Electrons per Pixel',fontsize = 12)  # Y-axis label
        plt.xlabel('Frame',fontsize = 12)  # Y-axis label
        plt.gca().tick_params(axis='y', labelsize=12)  # Set y-tick label font size to 12
        plt.gca().tick_params(axis='x', labelsize=12)  # Set y-tick label font size to 12
        plt.Figure()
        plt.imshow()

plt.Figure()
plt.imshow(masks[0])
#%% denoise test

from cellpose import denoise
dn = denoise.DenoiseModel(model_type="denoise_cyto3", gpu=True)
imgs_dn = dn.eval(imgs, channels=None, diameter=100.)

plt.figure()
plt.imshow(imgs_dn[0][:,:,0])

from cellpose import denoise
model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3",
             restore_type="denoise_cyto3", chan2_restore=True)
masks, flows, styles, imgs_dn = model.eval(imgs, channels=[1,2], diameter=50.)

plt.figure()
plt.imshow(imgs_dn[0][:,:,0])

#%%
labels = measure.label(masks[0], connectivity=2, background=0)
Averaged_electrons = np.zeros([len(np.unique(labels)),1])
for label in np.unique(labels):
    # if label == 0:
    #     continue
    cell_data = []
    labelMask = np.zeros(masks[0].shape, dtype="uint8")
    labelMask[labels == label] = 255
    indices = np.where(labelMask == 255)
    cell_data = np.mean(data[0][:,indices[0], indices[1]], axis=1)
    Averaged_electrons[label,0] = np.max(cell_data)
plt.figure() 
plt.plot(cell_data)   
    
outlines = utils.masks_to_outlines(masks[0])

outX, outY = np.nonzero(outlines)
imgout = masks[0].copy()

if len(imgout.shape) == 2:  # If it's a single-channel image (grayscale)
    imgout = np.stack([imgout] * 3, axis=-1)  # Convert to 3-channel RGB

# Set the red color for the specific pixels
color = [255, 0, 0]
imgout[outX, outY] = np.array(color)  # pure red
plt.figure()
plt.imshow(imgout)

Overlapped_image = mask_overlay(imgs[0], masks[0])
plt.figure()

# Display the base image (e.g., the first image in imgs)
plt.imshow(imgs[0],vmin=200,vmax= 300,cmap='gray')

# Overlay the second image (e.g., Overlapped_image) with transparency (alpha)
plt.imshow(Overlapped_image,alpha=0.4)  # Adjust alpha for transparency

plt.axis('off')  # Optionally turn off the axis if not needed
plt.show()

color = [255, 0, 0]

mask_RGB = plot.mask_overlay(imgs[22], masks[22])

plt.figure()
plt.imshow(mask_RGB)

# plot image with outlines overlaid in red
outlines = utils.outlines_list(masks[22])
plt.imshow(imgs[22],cmap=('gray'))
for o in outlines:
    plt.plot(o[:,0], o[:,1], color='r')

#%%
model = models.Cellpose(gpu=True, model_type='cyto')
flile = r'C:\\Users\zhu\Desktop\Figure\Quantification\raw_data'
# data_files = ['C:\\Users\zhu\Desktop\Z05_4dpf_mcherry_XYTZ_1070nm_2x2x4MHz_158mWz29_r1.tiff']
data_files = [os.path.join(flile, 'Z04_4dpf_mcherry_XYTZ_1070nm_2x2x4MHz_158mWz29_r1.tiff')]
data = [imread(f) for f in data_files] [0]
data1 = np.moveaxis(data, 0, -1)
data= data1[:,:,100:550:15]

imgs = [data[:, :, i] for i in range(data.shape[2])]

masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0],
                                         flow_threshold=0.5, do_3D=False)

# nimg = len(imgs)
# for idx in range(nimg):
#     maski = masks[idx]
#     flowi = flows[idx][0]

#     fig = plt.figure(figsize=(12,5))
#     plot.show_segmentation(fig, imgs[idx], maski, flowi)
#     plt.tight_layout()
#     plt.show()

index_frame = range(0,30)  # Selecting frames 8 to 12
Final_electrons1 = [] # List to store the final results
for index_frame in index_frame:
    
    labels = measure.label(masks[index_frame], connectivity=2, background=0)  # Label connected regions
    # plt.figure()
    # plt.imshow(masks[index_frame])
    Averaged_electrons = np.zeros([len(np.unique(labels)),1])  # Initialize the result array for each frame
    for label in np.unique(labels):
        if label == 0: # Optionally skip background if needed
            continue
        # Create a mask for the current label
        cell_data = []
        labelMask = np.zeros(masks[index_frame].shape, dtype="uint8")
        labelMask[labels == label] = 255
        #
        # plt.figure()
        # plt.imshow(labelMask)
        # Find indices of the current label
        indices = np.where(labelMask == 255)
        
        # Calculate the mean across the third axis of data1 for the selected indices
        cell_data = np.mean(data1[indices[0], indices[1],:], axis=0)
        
        Averaged_electrons[label,0] = np.max(cell_data)-200
        if np.max(cell_data)-200 >=30:
            Final_electrons1.append(np.max(cell_data)/9.1-200/9.1)                
 
plt.figure() 
plt.plot(Final_electrons1)   


#%%
import seaborn as sns

# Setting up matplotlib parameters
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2

# Create a figure to visualize the results
plt.figure(figsize=(4.3, 3.55))

# Create a violin plot of the Final_electrons data
sns.violinplot(data=Final_electrons1, color=(63/255, 171/255, 71/255))
# Set plot labels and title
plt.xlabel('Average Electrons/cell')
plt.ylabel('Pixel-Averaged Electron Value')
# plt.title('Violin Plot of Averaged Electron Values Across Frames')

# Adjust the layout for better visibility and show the plot
plt.tight_layout()
plt.show()

#%%
model = models.Cellpose(gpu=True, model_type='cyto')
flile = r'C:\\Users\zhu\Desktop\Figure\Quantification\raw_data'
# data_files = ['C:\\Users\zhu\Desktop\Z05_4dpf_mcherry_XYTZ_1070nm_2x2x4MHz_158mWz29_r1.tiff']
data_files = [os.path.join(flile, 'Z05_4dpf_mcherry_XYTZ_1070nm_2x2x4MHz_158mWz29_r1.tiff')]
data = [imread(f) for f in data_files] [0]
data1 = np.moveaxis(data, 0, -1)
data= data1[:,:,355:800:15]

imgs = [data[:, :, i] for i in range(data.shape[2])]

masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0],
                                         flow_threshold=0.5, do_3D=False)

# nimg = len(imgs)
# for idx in range(nimg):
#     maski = masks[idx]
#     flowi = flows[idx][0]

#     fig = plt.figure(figsize=(12,5))
#     plot.show_segmentation(fig, imgs[idx], maski, flowi)
#     plt.tight_layout()
#     plt.show()

index_frame = range(0,30)  # Selecting frames 8 to 12
Final_electrons = [] # List to store the final results
for index_frame in index_frame:
    
    labels = measure.label(masks[index_frame], connectivity=2, background=0)  # Label connected regions
    # plt.figure()
    # plt.imshow(masks[index_frame])
    Averaged_electrons = np.zeros([len(np.unique(labels)),1])  # Initialize the result array for each frame
    for label in np.unique(labels):
        if label == 0: # Optionally skip background if needed
            continue
        # Create a mask for the current label
        cell_data = []
        labelMask = np.zeros(masks[index_frame].shape, dtype="uint8")
        labelMask[labels == label] = 255
        #
        # plt.figure()
        # plt.imshow(labelMask)
        # Find indices of the current label
        indices = np.where(labelMask == 255)
        
        # Calculate the mean across the third axis of data1 for the selected indices
        cell_data = np.mean(data1[indices[0], indices[1],:], axis=0)
        
        Averaged_electrons[label,0] = np.max(cell_data)-200
        if np.max(cell_data)-200 >=30:
            Final_electrons.append(np.max(cell_data)/9.1-200/9.1)                
 
plt.figure() 
plt.plot(Final_electrons)   
#%%
combined_data = [Final_electrons, Final_electrons1]

plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2

# Create a figure to visualize the results
plt.figure(figsize=(5.5, 4.5))

# Create a violin plot of the combined data
sns.violinplot(data=combined_data, palette=["#501d8a", "#e55709"], inner="quartile", density_norm="count")

# Set plot labels and title
plt.xticks([0, 1], ['Embryo #1', 'Embryo #2'],fontsize = 12)  # Labeling the x-ticks
plt.ylabel('Average Electrons per Pixel',fontsize = 12)  # Y-axis label
# plt.title('Violin Plot of Pixel-Averaged Electron Values',fontsize = 12)  # Title
plt.gca().tick_params(axis='y', labelsize=12)  # Set y-tick label font size to 12
# Adjust the layout for better visibility and show the plot
plt.tight_layout()
plt.show()
ax1.text(i, mean_value, f'$\hat{{\mu}}$ ={mean_value:.4f}', color='k', ha='center', va='bottom')

#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

combined_data = [Final_electrons, Final_electrons1]
sample_counts = [len(data) for data in combined_data]

plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2

# Create a figure to visualize the results
plt.figure(figsize=(4.3, 3.73))

# Create a violin plot of the combined data
ax = sns.violinplot(data=combined_data, palette=["#501d8a", "#e55709"], inner="quartile", scale="count")

# Set plot labels and title
plt.xticks([0, 1], ['Embryo #1', 'Embryo #2'], fontsize=12)
plt.ylabel('Average Electrons per Pixel', fontsize=12)
plt.gca().tick_params(axis='y', labelsize=12)
y_index1 = 11
y_index2 = 12
# Calculate, mark, and annotate the mean values and sample counts
for i, data in enumerate(combined_data):
    mean_value = np.mean(data)
    # Mark the mean with a black dot
    # ax.scatter(i, mean_value, color='k', s=50, zorder=3)
    # Annotate the mean value and sample count
    ax.text(i+0.07, y_index1, f'$\mu$ ={mean_value:.2f}', color='k', ha='left', va='top', fontsize=12)
    ax.text(i+0.07, y_index2, f'$N$ = {sample_counts[i]:.0f}', color='k', ha='left',va='top', fontsize=12)

# Adjust the layout for better visibility and show the plot
plt.tight_layout()
plt.show()

#%% saving data
import pandas as pd

# Assuming combined_data is a list of lists
combined_data = [Final_electrons, Final_electrons1]  # Example data

# Create a DataFrame
df = pd.DataFrame(combined_data).T  # Transpose to have embryos as columns

# Set column names
df.columns = ['Embryo #1', 'Embryo #2']  # Adjust as necessary

# Save to CSV
df.to_csv('combined_data.csv', index=False)
        