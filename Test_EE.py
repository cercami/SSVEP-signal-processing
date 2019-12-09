# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:55:44 2019

Manually test the effect of EE

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

import mne
from mne.filter import filter_data
from sklearn.linear_model import LinearRegression

import signal_processing_function as SPF 

#%% load local data (extract from .cnt file)
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')

data = eeg['raw_data']
data *= 1e6  

del eeg

chans = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\chan_info.mat')
chans = chans['chan_info'].tolist()

sfreq = 1000

n_events = data.shape[0]
n_trials = data.shape[1]
n_chans = data.shape[2]
n_times = data.shape[3]

#%% Data preprocessing
f_data = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_events):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=5,
                      h_freq=40, n_jobs=6)

del i

w = f_data[:,:,:,2000:3000]
signal_data = f_data[:,:,:,3000:]   

del f_data, data, n_chans, n_events, n_times, n_trials

#%% pick channels
w_i = w[:,:,[chans.index('POZ'), chans.index('P8 '), chans.index('PO8')], :]
w_o = w[:,:,chans.index('OZ '), :]

sig_i = signal_data[:,:,[chans.index('POZ'), chans.index('P8 '), chans.index('PO8')], :]
sig_o = signal_data[:,:,chans.index('OZ '), :]

del w, signal_data

#%% multi-linear regression
rc, ri, r2 = SPF.mlr_analysis(w_i, w_o)
w_es_s, w_ex_s = SPF.sig_extract_mlr(rc, sig_i, sig_o, ri)
del ri, rc, r2

#%% psd
w_p, fs = SPF.welch_p(w_ex_s[:,:,:700], sfreq=sfreq, fmin=0, fmax=50, n_fft=1024,
                      n_overlap=0, n_per_seg=1024)
sig_p, fs = SPF.welch_p(sig_o[:,:,:700], sfreq=sfreq, fmin=0, fmax=50, n_fft=1024,
                        n_overlap=0, n_per_seg=1024)

#%% check waveform
plt.plot(np.mean(sig_o[1,:,:], axis=0), label='origin', color='tab:blue', linewidth=1.5)
plt.plot(np.mean(w_es_s[1,:,:], axis=0), label='estimation', color='tab:green', linewidth=1)
plt.plot(np.mean(w_ex_s[1,:,:], axis=0), label='extraction', color='tab:orange', linewidth=1)
plt.legend(loc='best')

#%% check time-domain snr
sig_snr_t = SPF.snr_time(sig_o, mode='time')
w_snr_t = SPF.snr_time(w_ex_s, mode='time')
#ws_snr_t = SPF.snr_time(w_es_s, mode='time')

plt.plot(sig_snr_t[1,:700], label='origin', color='tab:blue', linewidth=1.5)
plt.plot(w_snr_t[1,:700], label='extraction', color='tab:orange', linewidth=1)
#plt.plot(ws_snr_t[2,:700], label='estimation', color='tab:green', linewidth=1)
plt.legend(loc='best')

snr_t_raise = np.mean(w_snr_t[2,:] - sig_snr_t[2,:])
#snr_t_raise_s = np.mean(ws_snr_t[2,:] - sig_snr_t[2,:])

#%% check frequecy-domain snr
from math import log
def snr_freq(X, k):
    '''
    X: (n_epochs, n_freqs)
    '''
    snr = np.zeros((X.shape[1]))
    if k == 0:
        for i in range(X.shape[1]):
            snr[i] = np.sum(X[k,i,8:10]) / (np.sum(X[k,i,6:8]) + np.sum(X[k,i,10:12]))
            snr[i] = 10 * log(snr[i], 10)
    if k == 1:
        for i in range(X.shape[1]):
            snr[i] = np.sum(X[k,i,10:12]) / (np.sum(X[k,i,8:10]) + np.sum(X[k,i,12:14]))
            snr[i] = 10 * log(snr[i], 10)
    if k == 2:
        for i in range(X.shape[1]):
            snr[i] = np.sum(X[k,i,15:17]) / (np.sum(X[k,i,13:15]) + np.sum(X[k,i,17:19]))
            snr[i] = 10 * log(snr[i], 10)

    return snr

sig_snr_f = snr_freq(sig_p, k=1)
w_snr_f = snr_freq(w_p, k=1)
snr_f_raise = np.mean(w_snr_f - sig_snr_f)
snr_f_raise_std = np.std(w_snr_f - sig_snr_f)


#%% check psd
plt.plot(fs[1,1,:], np.mean(sig_p[1,:,:], axis=0), label='origin', color='tab:blue', linewidth=1.5)
plt.plot(fs[1,1,:], np.mean(w_p[1,:,:], axis=0), label='extraction', color='tab:orange', linewidth=1)
plt.legend(loc='best')


