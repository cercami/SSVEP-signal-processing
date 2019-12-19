# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:50:32 2019
MCEE test:
    (1)MCEE
    (2)Cross-validation

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
import mcee

import copy

#%% load local data (extract from .cnt file)
freq = 2  # 0 for 8Hz, 1 for 10Hz, 2 for 15Hz

eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
f_data = eeg['f_data'][freq, :, :, :]
chans = eeg['chan_info'].tolist()
f_data *= 1e6  

del eeg

sfreq = 1000

n_trials = f_data.shape[0]
n_chans = f_data.shape[1]
n_times = f_data.shape[2]

w = f_data[:, :, 2000:3000]
signal_data = f_data[:, :, 3200:3700]   

del n_chans, n_times
del f_data

# initialization
w_o = w[:, chans.index('POZ'), :]
w_i = np.delete(w, chans.index('POZ'), axis=1)

sig_o = signal_data[:, chans.index('POZ'), :]
sig_i = np.delete(signal_data, chans.index('POZ'), axis=1)

#%% N-fold cross-validation
# set cross-validation's folds
N = 10

# check numerical error
if n_trials%N != 0:
    print('Numerical error! Please check the folds again!')
    
cv_model_chans = []
cv_snr_change = []
    
for i in range(N):
    # divide data into N folds: test dataset
    a = i*10
    
    te_w_i = w_i[a:a+int(n_trials/N), :, :]
    te_w_o = w_o[a:a+int(n_trials/N), :, :]
    te_sig_i = sig_i[a:a+int(n_trials/N), :, :]
    te_sig_o = sig_o[a:a+int(n_trials/N), :, :]
    
    # training dataset
    tr_w_i = copy.deepcopy(w_i)
    tr_w_i = np.delete(tr_w_i, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=0)
    tr_w_o = copy.deepcopy(w_o)
    tr_w_o = np.delete(tr_w_o, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=0)
    
    tr_sig_i = copy.deepcopy(sig_i)
    tr_sig_i = np.delete(tr_sig_i, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=0)
    tr_sig_o = copy.deepcopy(sig_o)
    tr_sig_o = np.delete(tr_sig_o, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=0)
    
    # MCEE
    mcee_chans = copy.deepcopy(chans)
    del mcee_chans[mcee_chans.index('POZ')]
    
    snr = mcee.snr_time(tr_sig_o)
    msnr = np.mean(snr)
    
    model_chans, snr_change = mcee.stepwise_MCEE(chans=mcee_chans, msnr=msnr,
            w=tr_w_i, w_target=tr_w_o, signal_data=tr_sig_i, data_target=tr_sig_o)
    cv_model_chans.append(str(i)+'th')
    cv_model_chans.append(model_chans)
    cv_snr_change.append(str(i)+'th')
    cv_snr_change.append(snr_change)
    
    del snr, msnr, tr_w_i, tr_w_o, tr_sig_i, tr_sig_o, snr_change
    
    # pick channels chosen from MCEE (in test dataset)
    te_w_i = np.zeros((int(n_trials/N), len(model_chans), w_i.shape[2]))
    te_w_o = w_o[a:a+int(n_trials/N), chans.index('POZ'), :]
    
    te_sig_i = np.zeros((int(n_trials/N), len(model_chans), sig_i.shape[2]))
    te_sig_o = sig_o[a:a+int(n_trials/N), chans.index('POZ'), :]
    
    for j in range(len(model_chans)):
        te_w_i[:, j, :] = w_i[a:a+int(n_trials/N), chans.index(model_chans[j]), :]
        te_sig_i[:, j, :] = sig_i[a:a+int(n_trials/N), chans.index(model_chans[j]), :]
    
    del j
    
    # multi-linear regression
    
