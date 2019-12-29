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

import os

import mne
from mne.filter import filter_data
from sklearn.linear_model import LinearRegression

import signal_processing_function as SPF 
import mcee

import copy

#%% load local data (extract from .cnt file)
eeg = io.loadmat(r'G:\dataset\preprocessed_data\weisiwen\f_data.mat')
f_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
f_data *= 1e6  

del eeg

sfreq = 1000

n_events = f_data.shape[0]
n_trials = f_data.shape[1]
n_chans = f_data.shape[2]
n_times = f_data.shape[3]

w = f_data[:, :, :, 2000:3000]
signal_data = f_data[:, :, :, 3200:3700]   

del n_chans, n_times, n_trials, n_events
#del f_data

#%% MCEE
# initialization
freq = 0  # 0 for 8Hz, 1 for 10Hz, 2 for 15Hz
channel_target = 'OZ '

mcee_data_target = signal_data[freq, :, chans.index(channel_target), :]
mcee_sig_data = signal_data[freq, :, :, :]
mcee_sig_data = np.delete(mcee_sig_data, chans.index(channel_target), axis=1) 

mcee_w_target = w[freq, :, chans.index(channel_target), :]
mcee_w = w[freq, :, :, :]
mcee_w = np.delete(mcee_w, chans.index(channel_target), axis=1)

mcee_chans = copy.deepcopy(chans)

del mcee_chans[mcee_chans.index(channel_target)]

snr = mcee.snr_time(mcee_data_target)
msnr = np.mean(snr)

# MCEE optimization
model_chans, snr_change = mcee.stepwise_MCEE(chans=mcee_chans, msnr=msnr, w=mcee_w,
                        w_target=mcee_w_target, signal_data=mcee_sig_data,
                        data_target=mcee_data_target)

del mcee_data_target, mcee_sig_data, mcee_w_target, mcee_w

#%% pick channels
w_i = w[:,:,[chans.index('POZ'), chans.index('CP1')], :]
w_o = w[:,:,chans.index('OZ '), :]

sig_i = f_data[:,:,[chans.index('POZ'), chans.index('CP1')], 2800:3700]
sig_o = f_data[:,:,chans.index('OZ '), 2800:3700]

del w, signal_data

#%% multi-linear regression
rc, ri, r2 = SPF.mlr_analysis(w_i, w_o)
w_es_s, w_ex_s = SPF.sig_extract_mlr(rc, sig_i, sig_o, ri)
del ri, rc, r2
del w_o, w_i, sig_i

#%% psd
w_p, fs = SPF.welch_p(w_ex_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=1024,
                      n_overlap=0, n_per_seg=1024)
sig_p, fs = SPF.welch_p(sig_o, sfreq=sfreq, fmin=0, fmax=50, n_fft=1024,
                        n_overlap=0, n_per_seg=1024)
ws_p, fs = SPF.welch_p(w_es_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=1024,
                        n_overlap=0, n_per_seg=1024)

#%% check waveform
w = freq
plt.plot(np.mean(sig_o[w,:,:], axis=0), label='origin', color='tab:blue', linewidth=1.5)
plt.plot(np.mean(w_es_s[w,:,:], axis=0), label='estimation', color='tab:green', linewidth=1)
plt.plot(np.mean(w_ex_s[w,:,:], axis=0), label='extraction', color='tab:orange', linewidth=1)
plt.legend(loc='best')

#%% check time-domain snr
k = freq
sig_snr_t = SPF.snr_time(sig_o, mode='time')
w_snr_t = SPF.snr_time(w_ex_s, mode='time')
ws_snr_t = SPF.snr_time(w_es_s, mode='time')

plt.plot(sig_snr_t[k,:], label='origin', color='tab:blue', linewidth=1.5)
plt.plot(w_snr_t[k,:], label='extraction', color='tab:orange', linewidth=1)
plt.legend(loc='best')

snr_t_raise = np.mean(w_snr_t[k,:] - sig_snr_t[k,:])
snr_ts_raise = np.mean(ws_snr_t[k,:] - sig_snr_t[k,:])

percent_t = snr_t_raise/np.mean(sig_snr_t[k,:])*100
percent_ts = snr_ts_raise/np.mean(sig_snr_t[k,:])*100

#%% check frequecy-domain snr
f = freq
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

sig_snr_f = snr_freq(sig_p, k=f)
w_snr_f = snr_freq(w_p, k=f)
ws_snr_f = snr_freq(ws_p, k=f)

snr_f_raise = np.mean(w_snr_f - sig_snr_f)
snr_fs_raise = np.mean(ws_snr_f - sig_snr_f)

percent_f = snr_f_raise/np.mean(sig_snr_f)*100
percent_fs = snr_fs_raise/np.mean(sig_snr_f)*100

#%% check psd
p = freq
plt.plot(fs[1,1,:], np.mean(sig_p[p,:,:], axis=0), label='origin', color='tab:blue', linewidth=1.5)
plt.plot(fs[1,1,:], np.mean(w_p[p,:,:], axis=0), label='extraction', color='tab:orange', linewidth=1)
plt.plot(fs[1,1,:], np.mean(ws_p[p,:,:], axis=0), label='estimation', color='tab:green', linewidth=1)
plt.title('Stepwise')
plt.legend(loc='best')

#%%
sns.set(style='whitegrid')

fig, ax = plt.subplots(2, 1, figsize=(12, 12))

ax[0].plot(sig_o[freq, :, :].T, linewidth=1)
ax[0].plot(np.mean(sig_o[freq, :, :], axis=0), linewidth=4, color='black', label='mean')
ax[0].set_title('8Hz & OZ origin', fontsize=18)
ax[0].set_xlabel('Time/ms', fontsize=16)
ax[0].set_ylabel('Amplitude/uV', fontsize=16)
ax[0].vlines(208.3, -20, 30, linestyle='dashed', color='black', label='start')
ax[0].set_ylim([-20, 20])
ax[0].legend(loc='best', fontsize=16)

ax[1].plot(w_ex_s[freq, :, :].T, linewidth=1)
ax[1].plot(np.mean(w_ex_s[freq, :, :], axis=0), linewidth=4, color='black', label='mean')
ax[1].set_title('8Hz & OZ extraction', fontsize=18)
ax[1].set_xlabel('Time/ms', fontsize=16)
ax[1].set_ylabel('Amplitude/uV', fontsize=16)
ax[1].vlines(208.3, -20, 30, linestyle='dashed', color='black', label='start')
ax[1].set_ylim([-20, 20])
ax[1].legend(loc='best', fontsize=16)

fig.tight_layout()
plt.show()
plt.savefig(r'E:\8-oz.png', dpi=300)