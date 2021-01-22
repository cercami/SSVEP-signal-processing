'''
Author: your name
Date: 2020-12-28 22:31:31
LastEditTime: 2021-01-01 20:56:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \SSVEP-signal-processing\boruikang_loadData.py
'''
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:07:27 2020

read data from boruikang device

@author: Brynhildr
"""

# %%
import os
import scipy.io as io

import numpy as np
from numpy import newaxis as NA

import mne
from mne import Epochs
from mne.filter import filter_data

import copy
import string

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import mcee

# %%
data_path = r'D:\SSVEP\program\code_1bits.mat'
code_data = io.loadmat(data_path)
code_series = code_data['VEPSeries_1bits']  # (n_codes, n_elements)
data = np.zeros((32, 5, 8, 4001))
# %%
data_path = r'D:\SSVEP\boruikang test\32 target\32_05.mat'
brk = io.loadmat(data_path)
eeg = brk['EEG']
raw_data = eeg['data'][0,0]
del brk, data_path

tmin, tmax = -1, 3
sfreq = 1000

event_id, event = [], []
temp_data_1, temp_data_2 = np.zeros((1, 8, 2001)), np.zeros((1, 8, 2001))

for i in range(32):
    event_id = int(eeg['event'][0,0][i,0][0])
    event = int(eeg['event'][0,0][i,0][1])

    start_point = event + tmin*sfreq - 1
    end_point = event + tmax*sfreq
  
    data[i, 4, ...] = raw_data[:, start_point:end_point]

# %%
f60p0 = filter_data(data_1, sfreq=sfreq, l_freq=50, h_freq=70, n_jobs=8, method='iir')
f60p1 = filter_data(data_2, sfreq=sfreq, l_freq=50, h_freq=70, n_jobs=8, method='iir')

# %%
for i in range(32):
    data[i, ...] = filter_data(data[i, ...], sfreq=sfreq, l_freq=50, h_freq=70, n_jobs=8, method='iir')

# %%
ori_data = np.concatenate((f60p0[NA,...], f60p1[NA,...]), axis=0)

# %%
import mcee
from numpy import newaxis as NA
tar_chan_index = [45,51,52,53,54,55,58,59,60]
data_path = r'D:\SSVEP\dataset\preprocessed_data\cvep_8\wuqiaoyi\fir_50_70.mat'
eeg = io.loadmat(data_path)
test_data = eeg['f_data'][:, :, tar_chan_index, :]
del data_path, eeg
acc_ori_trca = np.zeros((6))
acc_ori_etrca = np.zeros_like(acc_ori_trca)
data = test_data[..., 1140:2940]

for cv in range(6):
    print('CV: %d turn...' %(cv+1))
    test_list = [i+cv*10 for i in range(10)]
    trainData = np.delete(data, test_list, axis=1)
    testData = data[:, test_list, ...]
    # _, acc_trca = mcee.split_TRCA(600, testData, trainData)
    # _, acc_etrca = mcee.split_eTRCA(600, testData, trainData)
    acc_trca = mcee.TRCA(testData, trainData)
    acc_etrca = mcee.eTRCA(testData, trainData)
    acc_ori_trca[cv] = acc_trca
    acc_ori_etrca[cv] = acc_etrca
    print(str(cv+1) + 'th cross-validation complete!\n')

# %%
model = mcee.sin_model(60, 1, 1)
data_path = r'D:\SSVEP\dataset\preprocessed_data\60&80\zhaowei\fir_50_90.mat'
eeg = io.loadmat(data_path)
data = eeg['f_data'][[0,2],:,:,1140:2140][...,[45,51,52,53,54,55,58,59,60],:]
del eeg
# %%
w_Y = mcee.CCA_compute(data[0,0,...], model, mode='model')