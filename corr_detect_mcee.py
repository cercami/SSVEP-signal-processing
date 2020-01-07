# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:17:22 2020

MCEE + Correlation detection

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io
from scipy import signal

import mcee

import copy

import signal_processing_function as SPF 
import matplotlib.pyplot as plt

import seaborn as sns
  
#%% load mcee data
# 9 chans, 1140-1340(300ms)
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_2.mat')
# 0(500),1(400),2(300),3(200),4(180),5(160),6(140),7(120),8(100),9(80),10(60)
mcee_sig = eeg['mcee_sig']
tar_chans = eeg['chan_info'].tolist()
del eeg

# (n_events, n_trials, n_times)
pz = mcee_sig[:,:,0,1140:1440]   # 39
po5 = mcee_sig[:,:,1,1140:1440]  # 45
po3 = mcee_sig[:,:,2,1140:1440]  # 46
poz = mcee_sig[:,:,3,1140:1440]  # 47
po4 = mcee_sig[:,:,4,1140:1440]  # 48
po6 = mcee_sig[:,:,5,1140:1440]  # 49
o1 = mcee_sig[:,:,6,1140:1440]   # 52
oz = mcee_sig[:,:,7,1140:1440]   # 53
o2 = mcee_sig[:,:,8,1140:1440]   # 54

n_events = mcee_sig.shape[0]
n_trials = mcee_sig.shape[1]

del mcee_sig

#%% load origin data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
ori_sig = eeg['f_data'][:,:,:,3140:3440] * 1e6
chan_info = eeg['chan_info'].tolist()
del eeg

# (n_events, n_trials, n_times)
pz = ori_sig[:,:,39,:]
po5 = ori_sig[:,:,45,:]
po3 = ori_sig[:,:,46,:]
poz = ori_sig[:,:,47,:]
po4 = ori_sig[:,:,48,:]
po6 = ori_sig[:,:,49,:]
o1 = ori_sig[:,:,52,:]
oz = ori_sig[:,:,53,:]
o2 = ori_sig[:,:,54,:]

n_events = ori_sig.shape[0]
n_trials = ori_sig.shape[1]

del ori_sig
#del chan_info

#%% correlation detection of single-channel data
acc, mr, rou = mcee.corr_detect(test_data=pz, template=np.mean(pz, axis=1))

#%% reshape results
acc/=300
acc*=100

#%% check waveform
sns.set(style='whitegrid')
fig, ax = plt.subplots(3,1,figsize=(4,12))

ax[0].plot(poz[0,:,:].T, linewidth=0.5)
ax[0].plot(np.mean(poz[0,:,:], axis=0), linewidth=3, color='black')
ax[0].set_ylim(-20,20)

ax[1].plot(poz[1,:,:].T, linewidth=0.5)
ax[1].plot(np.mean(poz[1,:,:], axis=0), linewidth=3, color='black')
ax[1].set_ylim(-20,20)

ax[2].plot(poz[2,:,:].T, linewidth=0.5)
ax[2].plot(np.mean(poz[2,:,:], axis=0), linewidth=3, color='black')
ax[2].set_ylim(-20,20)

#%% check waveform
plt.plot(template[0,:], color='tab:orange')
plt.plot(poz[0,76,:], color='tab:blue')

