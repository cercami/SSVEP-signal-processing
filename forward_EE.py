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

import matplotlib.pyplot as plt


#%% timing
start = time.clock()

#%% load data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')

data = eeg['raw_data'][1,:,:,:]

data *= 1e6  # reset unit

del eeg

# in future versions, chan_info will be combined into raw_data.mat
chans = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\chan_info.mat')
chans = chans['chan_info'].tolist()

# basic info
sfreq = 1000

#%% Data preprocessing
# filtering
f_data = np.zeros((data.shape[0], data.shape[1], 3700))
for i in range(data.shape[0]):
    f_data[i,:,:] = filter_data(data[i,:,:3700], sfreq=sfreq, l_freq=5,
                      h_freq=40, n_jobs=1)

del i, sfreq

# get data for linear regression
w = f_data[:,:,2000:3000]          

# get data for comparision
signal_data = f_data[:,:,3200:]   # 500ms after 200ms duration

del f_data, data

#%% define function
# multi-linear regression
def mlr(model_input, model_target, data_input, data_target):
    '''
    model_input: (n_trials, n_chans, n_times) (n_chans, n_trials, n_times)
    model_target: (n_trials, n_times)
    data_input: (n_trials, n_chans, n_times) (n_chans, n_trials, n_times)
    data_target: (n_trials, n_times)
    '''
    if model_input.ndim == 3:
        # (n_trials)
        RI = np.zeros((model_input.shape[1]))
        # (n_trials, n_times)
        estimate = np.zeros((model_input.shape[1], data_input.shape[2]))
        # (n_trials, n_chans)
        RC = np.zeros((model_input.shape[1], model_input.shape[0]))
        for i in range(model_input.shape[1]):    # i for trials
            L = LinearRegression().fit(model_input[:,i,:].T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = (np.mat(RC) * np.mat(data_input[:,i,:])).A + RI
    
    elif model_input.ndim == 2:
        # (n_trials)
        RI = np.zeros((model_input.shape[0]))
        # (n_trials, n_times)
        estimate = np.zeros((model_input.shape[0], data_input.shape[1]))
        # (n_trials)
        RC = np.zeros((model_input.shape[0]))
        for i in range(model_input.shape[0]):    # i for trials
            L = LinearRegression().fit(np.mat(model_input[i,:]).T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = RC * data_input[i,:] + RI
    
    extract = data_target - estimate
    
    return extract, estimate

# time-domain snr
def snr_time(data):
    '''
    data:(n_trials, n_times)
    '''
    snr = np.zeros((data.shape[1]))             # (n_times)
    ex = np.mat(np.mean(data, axis=0))          # one-channel data (1, n_times)
    temp = np.mat(np.ones((1, data.shape[0])))     # (1, n_trials)
    minus = (temp.T * ex).A                     # (n_trials, n_times)
    ex = (ex.A) ** 2
    var = np.mean((data - minus)**2, axis=0)
    snr = ex/var
    
    return snr

#%% Forward Estimate Extraction
data_target = signal_data[:, chans.index('POZ'), :]
signal_data = np.delete(signal_data, chans.index('POZ'), axis=1)

w_target = w[:,chans.index('POZ'),:]
w = np.delete(w, chans.index('POZ'), axis=1)

del chans[chans.index('POZ')]

# initialization
snr = snr_time(data_target)
msnr = np.mean(snr)
compare_snr = np.zeros((len(chans)))
max_loop = len(chans)

remain_chans = []
snr_change = []

temp_snr = []
core_data = []
core_w = []

j = 1

# begin loop
active = True
while active and len(chans) <= max_loop:
    # initialization
    compare_snr = np.zeros((len(chans)))
    mtemp_snr = np.zeros((len(chans)))
    # add 1 channel respectively and compare the snr with original one
    for i in range(len(chans)):
        # avoid reshape error in multi-dimension array
        if j == 1:
            temp_w = w[:,i,:]
            temp_data = signal_data[:,i,:]
        else:
            temp_w = np.zeros((j, w.shape[0], w.shape[2]))
            temp_w[:j-1, :, :] = core_w
            temp_w[j-1, :, :] = w[:,i,:]
        
            temp_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
            temp_data[:j-1, :, :] = core_data
            temp_data[j-1, :, :] = signal_data[:,i,:]
        # multi-linear regression & snr computation
        temp_extract, temp_estimate = mlr(temp_w, w_target, temp_data, data_target)
        temp_snr = snr_time(temp_extract)
        mtemp_snr[i] = np.mean(temp_snr)
        compare_snr[i] = mtemp_snr[i] - msnr
    # keep the channels which can improve snr
    chan_index = np.max(np.where(compare_snr == np.max(compare_snr)))
    remain_chans.append(chans.pop(chan_index))
    snr_change.append(np.max(compare_snr))
    # refresh data & avoid reshape error in multi-dimension array
    if j == 1:
        core_w = w[:, chan_index, :]
        core_data = signal_data[:, chan_index, :]
        
        # refresh data
        signal_data = np.delete(signal_data, chan_index, axis=1)
        w = np.delete(w ,chan_index, axis=1)
        # significant loop mark
        print('Complete ' + str(j) + 'th loop')
        j += 1
    else:
        temp_core_w = np.zeros((j, w.shape[0], w.shape[2]))
        temp_core_w[:j-1, :, :] = core_w
        temp_core_w[j-1, :, :] = w[:, chan_index, :]
        core_w = temp_core_w.copy()
            
        temp_core_data = np.zeros((j, signal_data.shape[0], signal_data.shape[2]))
        temp_core_data[:j-1, :, :] = core_data
        temp_core_data[j-1, :, :] = signal_data[:, chan_index, :]
        core_data = temp_core_data.copy()
        if snr_change[j-1] < np.max(snr_change):
            end = time.clock()
            print('Forward EE complete!')
            print('Total running time: ' + str(end - start) + 's')
            active = False
        else:
            # refresh data
            signal_data = np.delete(signal_data, chan_index, axis=1)
            w = np.delete(w ,chan_index, axis=1)
            # significant loop mark
            print('Complete ' + str(j) + 'th loop')
            j += 1

# Algorithm operating results
remain_chans = remain_chans[:len(remain_chans)-1]
        
