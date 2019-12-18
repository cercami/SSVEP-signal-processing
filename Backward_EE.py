# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:57:21 2019

Backward Estimate Extraction algorithm to improve short-time SSVEP's SNR

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

import copy

#%% timing
start = time.clock()

#%% load data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')

data = eeg['raw_data'][2,:,:,:]
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
def mlr(model_input, model_target, data_input, data_target):
    '''
    model_input: (n_trials, n_chans, n_times)
    model_target: (n_trials, n_times)
    data_input: (n_trials, n_chans, n_times)
    data_target: (n_trials, n_times)
    '''
    # (n_trials)
    RI = np.zeros((model_input.shape[0]))
    # (n_trials, n_times)  
    if model_input.ndim == 3:
        estimate = np.zeros((model_input.shape[0], data_input.shape[2]))
        # (n_trials, n_chans)
        RC = np.zeros((model_input.shape[0], model_input.shape[1]))
        for i in range(model_input.shape[0]):    # i for trials
            L = LinearRegression().fit(model_input[i,:,:].T, model_target[i,:].T)
            RI = L.intercept_
            RC = L.coef_
            estimate[i,:] = (np.mat(RC) * np.mat(data_input[i,:,:])).A + RI
    
    elif model_input.ndim == 2:
        estimate = np.zeros((model_input.shape[0], data_input.shape[1]))
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

#%% Backward Estimate Extraction
data_target = signal_data[:, chans.index('POZ'), :]
signal_data = np.delete(signal_data, chans.index('POZ'), axis=1)

w_target = w[:,chans.index('POZ'),:]
w = np.delete(w, chans.index('POZ'), axis=1)

del chans[chans.index('POZ')]

# initialization
snr = snr_time(data_target)
msnr = np.mean(snr)


#%% Backward EE
def backward_EE(chans, msnr, w, w_target, signal_data, data_target):
    '''
    '''
    # initialize variables
    print('Running Backward EE...')
    start = time.clock()
    compare_snr = np.zeros((len(chans)))
    j = 0

    delete_chans = []
    snr_change = []
    # begin loop
    active = True
    while active:
        if len(chans) > 1:
            compare_snr = np.zeros((len(chans)))
            mtemp_snr = np.zeros((len(chans)))
            # delete 1 channel respectively and compare the snr with original one
            for i in range(len(chans)):
                # initialization
                temp_chans = copy.deepcopy(chans)
                temp_data = copy.deepcopy(signal_data)
                temp_w = copy.deepcopy(w)
                # delete one channel
                del temp_chans[i]
                temp_data = np.delete(temp_data, i, axis=1)
                temp_w = np.delete(temp_w, i, axis=1)
                # compute snr
                temp_extract, temp_estimate = mlr(temp_w, w_target, temp_data, data_target)
                temp_snr = snr_time(temp_extract)
                mtemp_snr[i] = np.mean(temp_snr)
                compare_snr[i] = mtemp_snr[i] - msnr
            # keep the channels which can improve snr forever
            chan_index = np.max(np.where(compare_snr == np.max(compare_snr)))
            delete_chans.append(chans.pop(chan_index))
            snr_change.append(np.max(compare_snr))
            # refresh data
            signal_data = np.delete(signal_data, chan_index, axis=1)
            w = np.delete(w, chan_index, axis=1)
            # significant loop mark
            j += 1 
            print('Complete ' + str(j) + 'th loop')
        # Backward EE complete
        else:
            end = time.clock()
            print('Backward EE complete!')
            print('Recursive running time: ' + str(end - start) + 's')
            active = False
        
    model_chans = chans + delete_chans[-2:]
    return model_chans, snr_change
    
#%% Algorithm operating result
model_chans, snr_change = backward_EE(chans=chans, msnr=msnr, w=w, w_target=w_target,
                          signal_data=signal_data, data_target=data_target)

