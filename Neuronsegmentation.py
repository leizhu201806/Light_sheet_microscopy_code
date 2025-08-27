# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:59:46 2025

@author: ZHU
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

model = models.Cellpose(gpu=True, model_type='cyto')
flile = r'C:\\Users\zhu\Desktop\Neuron data\Z08_4dpf_jrgeco_XYT_1070nm_2x2x4MHz_145mW'
# data_files = ['C:\\Users\zhu\Desktop\Z05_4dpf_mcherry_XYTZ_1070nm_2x2x4MHz_158mWz29_r1.tiff']
data_files = [os.path.join(flile, 'Z08_4dpf_jrgeco_XYT_1070nm_2x2x4MHz_145mWz0_r1.tiff')]
data = [imread(f) for f in data_files] [0]
data1 = np.moveaxis(data, 0, -1)
data= data1[:,:,100:550:15]

imgs = [data[:, :, i] for i in range(data.shape[2])]

masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=[0,0],
                                         flow_threshold=0.5, do_3D=False)
