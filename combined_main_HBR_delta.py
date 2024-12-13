import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import interpolate
import os
from numpy import genfromtxt
import csv

plt.rcParams['figure.figsize'] = [10, 5]

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def signaltonoise(a, axis=0):
    m = a.mean(axis)
    ma = np.amax(a, axis)
    return ma / m

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# data_dir = 'Z:\\2016_2020_ZebrafishHeartProject\\Dale - Selected Data\\'
data_dir = 'O://Lei/Processed data/'
# save_dir = 'M:\\Lei\\Processed_data_results\\'
save_dir = 'O://Lei/Processed data/'

dates = ['18062024 HBR']

# Define a list of colors for plotting

for date in dates:

    # Get all data names from the folders in the date directory
    files = []
    for data_name in get_immediate_subdirectories(data_dir + date):
        if data_name[:3] == 'Z04':
            files.append(data_name)
    colors = plt.cm.viridis(np.linspace(0, 1, np.count_nonzero(files)))
    added_labels = set()
    # Iterate plotting through all the data files in the folder
    # for data_name in files:
    for index, data_name in enumerate(files):
        current_folder = data_dir + date + '\\' + data_name + '\\'
        im = io.imread(current_folder + data_name + '.tiff')
       
        #%%
        # Use the timestamps of the frames to calculate the framerate of the camera
        
        # timestamp = genfromtxt(current_folder + 'timestamp.csv', delimiter=',') #% old code
        
        # Initialize an empty list to store timestamps #% new code
        timestamps = []
        file_path = current_folder + 'timestamp.csv'
        # Open and read the CSV file using the csv module
        with open(file_path, mode='r', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
               # Assuming each row contains a single timestamp value
               timestamps.append(float(row[0]))

        # Convert the list of timestamps to a NumPy array
        timestamp = np.array(timestamps)
        #%%        
        # frame_rate = timestamp[-2, 1] / timestamp[-2, 0]
        frame_rate = np.round(1/(timestamp[2]-timestamp[1]))
        # frame_rate = timestamp[2]-timestamp[1]
        # print(data_name)
        im = im-np.mean(im)
        # im_rotated = np.moveaxis(im[:,10:16,8:13], 0,2)
        im_rotated = np.moveaxis(im, 0,2)
        
        # image = im[10000,8:16,6:13]
        # plt.figure()
        # plt.imshow(image,cmap='gray', vmax = 1*np.max(image))]

        # Creates a mask of the pixels with the best snr in the fourier domain. Good pixels are left with val > 0
        # 3000 measures the heart beat before the laser turns on...since sometimes pixels burn
        # 5:200 avoids the low resonance peak in the FT
        Shutter_closed = 9500
        # arr_fft = fft(im_rotated[..., -Shutter_closed:])
        arr_fft = fft(im_rotated[..., :Shutter_closed])
        y = 2.0 / Shutter_closed * np.abs(arr_fft[..., :Shutter_closed // 2])
        # 60 corresponds to 3.0 Hz which would be too fast for the fish. We therefore select pixels that perform well
        # within a reasonable frequency range
        snr = signaltonoise(y[..., 5:2000], axis=2) #% option i noise calculation 
        # snr = np.mean(y[..., 5:2000], axis=2) / np.std(y[..., 5:2000], axis=2) #% option ii noise calculation 
        mask = np.abs((snr > np.partition(snr.flatten(), -10)[-10]) * snr)
        mask[mask > 0] = 1
        
        N = int(frame_rate*10)

        T = 1.0 / frame_rate
        xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
        hanning = np.hanning(N)

        # For every element inside the mask that's positive (good snr)
        binned = rolling_window(im_rotated[0,0,:], N)
        frequencies = np.zeros((np.count_nonzero(mask), len(binned[::100])))
        for (idx, i) in enumerate(np.argwhere(mask > 0)):
            # Create a rolling window N long along the Z axis of the image. This is the heartbeat
            binned = rolling_window(im_rotated[i[0],i[1],:], N)
            for (idx2, j) in enumerate(binned[::100]):
                yf = fft(j * hanning)
                y = 2.0/N * np.abs(yf[:N//2])
                # Only consider frequencies up to 4Hz, because otherwise you catch the second harmonic sometimes
                max_index = np.argmax(y[10:np.argwhere(xf > 4)[0,0]]) + 10
                xnew = np.linspace(xf[max_index-5],xf[max_index+5],2000)
                # Perform spline interpolation +/- 5 pixelsfrom the peak in the plot
                tck = interpolate.splrep(xf[max_index-5:max_index+5], y[max_index-5:max_index+5])
                ynew = interpolate.splev(xnew, tck)
                # Limit the peak finding in case the spline fitting shoots up at the end
                max_index2 = np.argmax(ynew[300:-300])
                frequencies[idx, idx2] = xnew[300:-300][int(max_index2)]
            #The 5 compensates for the 5 second window that we use for the ft
            # plt.plot(np.linspace(5, timestamp[-2, 0] - 5, len(frequencies[idx])), frequencies[idx], '.')
            # label = data_name if data_name not in added_labels else ""
            # plt.plot(np.linspace(5, timestamp[-2] - 5, len(frequencies[idx])), frequencies[idx], '.', color=colors[index], label=label)
        avg = np.linspace(5, timestamp[-2] - 5, len(frequencies[idx]))
        label = data_name if data_name not in added_labels else ""
        label = label[-5:]
        # plt.plot(np.linspace(5, timestamp[-2] - 5, len(frequencies[idx])),
        #           np.mean(frequencies,axis=(0)), '.', color=colors[index], label=label)
        # plt.plot(np.linspace(5, timestamp[-2] - 5, len(frequencies[idx])),
        #          np.mean(frequencies,axis=(0))-np.mean(frequencies[:,:25]), '.', color=colors[index], label=label)
        
        mean_freq = np.mean(frequencies, axis=0)
        std_freq = np.std(frequencies, axis=0)

        plt.plot(np.linspace(5, timestamp[-2] - 5, len(mean_freq)), mean_freq -  np.mean(mean_freq [:20], axis=0), '.', color=colors[index], label=label)
        # plt.errorbar(np.linspace(5, timestamp[-2] - 5, len(mean_freq)), mean_freq, yerr=std_freq, fmt='o--',
        #               color=colors[index], label=label, alpha=0.2)
        # print(data_name + "," + data_name.split('mW', 1)[0].replace('.', '').upper()[-3:] +','+ str(np.mean(frequencies[...,150:])) + ',' + str(np.mean(frequencies[...,55:86])))
    filname = data_name.replace("Z", "C")    
    combined_save_folder = os.path.join(save_dir, date, f'{filname}_combined')
    if not os.path.exists(combined_save_folder):
        os.makedirs(combined_save_folder)    
    plt.title(''r'$\Delta$ Frequency for ' + data_name[:-6])
    plt.xlabel('Time (s)')
    # plt.ylabel('Freq (Hz)')
    plt.ylabel(''r'$\Delta$ Freq (Hz)')
    plt.ylim((0,0.8))
    plt.legend()  # Add legend
    # plt.savefig(combined_save_folder +'\\' + data_name[:-6] + 'DELTA.png')
    # plt.close()  


        