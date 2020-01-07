# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:06:59 2019

benchmark dataset loading program

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
from numpy import transpose
import scipy.io as io
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

import os

import mne
from mne.filter import filter_data
from sklearn.linear_model import LinearRegression

import signal_processing_function as SPF 

#%% prevent ticking 'F5'
???

#%% Load benchmark dataset & relative information
# load data from .mat file
eeg = io.loadmat(r'E:\dataset\data\S15.mat')
info = io.loadmat(r'E:\dataset\Freq_Phase.mat')

data = eeg['data']
# (64, 1500, 40, 6) = (n_chans, n_times, n_events, n_blocks)
# total trials = n_conditions x n_blocks = 240
# all epochs are down-sampled to 250 Hz, HOLLY SHIT!

# reshape data array: (n_events, n_epochs, n_chans, n_times)
data = data.transpose((2, 3, 0, 1))  

# combine data array: np.concatenate(X, Y, axis=)

# condition infomation
sfreq = 250
freqs = info['freqs'].T
phases = info['phases'].T
del eeg, info


#%% load channels information from .txt file
channels = {}
file = open(r'F:\SSVEP\dataset\channel_info\weisiwen_chans.txt')
for line in file.readlines():
    line = line.strip()
    v = str(int(line.split(' ')[0]) - 1)
    k = line.split(' ')[1]
    channels[k] = v
file.close()

del v, k, file, line       # release RAM
     

#%% Load multiple data file & also can be used to process multiple data
# CAUTION: may lead to RAM crash (5-D array takes more than 6125MB)
# Now I know why people need 32G's RAM...PLEASE SKIP THIS PART!!!
filepath = r'E:\dataset\data'

filelist = []
for file in os.listdir(filepath):
    full_path = os.path.join(filepath, file)
    filelist.append(full_path)

i = 0
eeg = np.zeros((35, 64, 1500, 40, 6))
for file in filelist:
    temp = io.loadmat(file)
    eeg[i,:,:,:,:] = temp['data']
    i += 1
    
# add more codes here to achieve multiple data processing (PLEASE DON'T)
    
del temp, i, file, filelist, filepath, full_path
