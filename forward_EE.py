# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:55:52 2019

Forward Estimate Extraction algorithm to improve short-time SSVEP's SNR

@author: Brynhildr
"""

#%% import 3rd-part module
import numpy as np
import scipy.io as io
import pandas as pd

import os
import time

import mne
from mne.filter import filter_data

from sklearn.linear_model import LinearRegression

import signal_processing_function as SPF 


#%% timing
start = time.clock()


#%% load data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')

data = eeg['raw_data'][0,:,:,:]
#[0,:,:,:]
data *= 1e6  # reset unit

del eeg

# in future versions, chan_info will be combined into raw_data.mat
chans = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\chan_info.mat')
chans = chans['chan_info'].tolist()

# basic info
sfreq = 1000


#%% Data preprocessing
# filtering
f_data = np.zeros((data.shape[0], data.shape[1], 3640))
for i in range(data.shape[0]):
    f_data[i,:,:] = filter_data(data[i,:,:3640], sfreq=sfreq, l_freq=5,
                      h_freq=40, n_jobs=1)

del i, sfreq

# get data for linear regression
w = f_data[:,:,2000:3000]          

# get data for comparision
signal_data = f_data[:,:,3140:]   # 500ms after 140ms duration

del f_data, data

end_pre = time.clock()
print('Data preprocessing: ' + str(end_pre-start) + 's')


#%% Forward Estimate Extraction
target = signal_data[:, chans.index('OZ '), :]
signal_data = np.delete(signal_data, chans.index('OZ '), axis=1)

#%%

