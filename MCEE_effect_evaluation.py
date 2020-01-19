# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:50:32 2019
MCEE effect evaluation:
    (1)MCEE main process
    (2)cross-validation
    (3)figures (waveform & psd, etc)
    

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
from numpy import transpose
import scipy.io as io
import pandas as pd

import signal_processing_function as SPF 
import mcee

import copy

#import fp_growth as fpg

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

#%%******************** Holistic evaluation ********************%%#
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee_3.mat')
mcee_sig = eeg['mcee_sig'][:,:,:,1140:1540]
chans = eeg['chan_info'].tolist()
del eeg

p0 = mcee_sig[0,:,:,:]
p1 = mcee_sig[1,:,:,:]
del mcee_sig

# plot figures
plt.plot(np.mean(p0[:,0,:], axis=0), label='PZ')
plt.plot(np.mean(p0[:,1,:], axis=0), label='PO5')
plt.plot(np.mean(p0[:,2,:], axis=0), label='PO3')
plt.plot(np.mean(p0[:,3,:], axis=0), label='POZ')
plt.plot(np.mean(p0[:,4,:], axis=0), label='PO4')
plt.plot(np.mean(p0[:,5,:], axis=0), label='PO6')
plt.plot(np.mean(p0[:,6,:], axis=0), label='O1')
plt.plot(np.mean(p0[:,7,:], axis=0), label='OZ')
plt.plot(np.mean(p0[:,8,:], axis=0), label='O2')
plt.legend(loc='best')

#%% load local data (extract from .cnt file)
freq = 0  # 0 for 8Hz, 1 for 10Hz, 2 for 15Hz
target_channel = 'O2 '

eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
f_data = eeg['f_data'][freq, :, :, :]
chans = eeg['chan_info'].tolist()
f_data *= 1e6  

del eeg

sfreq = 1000

n_trials = f_data.shape[0]
n_chans = f_data.shape[1]
n_times = f_data.shape[2]

w = f_data[:, :, 2000:3000]
signal_data = f_data[:, :, 3140:3240]   

del n_chans, n_times
del f_data

# initialization
w_o = w[:, chans.index(target_channel), :]
w_i = np.delete(w, chans.index(target_channel), axis=1)

sig_o = signal_data[:, chans.index(target_channel), :]
sig_i = np.delete(signal_data, chans.index(target_channel), axis=1)

del chans[chans.index(target_channel)]

#%% N-fold cross-validation
# set cross-validation's folds
N = 10

# check numerical error
if n_trials%N != 0:
    print('Numerical error! Please check the folds again!')

# initialize variables
cv_model_chans = []
cv_snr_change = []

gf = np.zeros((N, int(n_trials/N)))

snr_t_raise = np.zeros((N))
percent_t = np.zeros((N))

snr_f_raise = np.zeros((N))
percent_f = np.zeros((N))
    
for i in range(N):
    # divide data into N folds: test dataset
    a = i*10
    
    te_w_i = w_i[a:a+int(n_trials/N), :, :]
    te_w_o = w_o[a:a+int(n_trials/N), :]
    te_sig_i = sig_i[a:a+int(n_trials/N), :, :]
    te_sig_o = sig_o[a:a+int(n_trials/N), :]
    
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
    
    snr = mcee.snr_time(tr_sig_o)
    msnr = np.mean(snr)
    
    model_chans, snr_change = mcee.stepwise_MCEE(chans=mcee_chans, msnr=msnr,
            w=tr_w_i, w_target=tr_w_o, signal_data=tr_sig_i, data_target=tr_sig_o)
    cv_model_chans.append(model_chans)
    cv_snr_change.append(snr_change)
    
    del snr, msnr, tr_w_i, tr_w_o, tr_sig_i, tr_sig_o, snr_change
    
    # pick channels chosen from MCEE (in test dataset)
    te_w_i = np.zeros((int(n_trials/N), len(model_chans), w_i.shape[2]))
    te_w_o = w_o[a:a+int(n_trials/N), :]
    
    te_sig_i = np.zeros((int(n_trials/N), len(model_chans), sig_i.shape[2]))
    te_sig_o = sig_o[a:a+int(n_trials/N), :]
    
    for j in range(len(model_chans)):
        te_w_i[:, j, :] = w_i[a:a+int(n_trials/N), chans.index(model_chans[j]), :]
        te_sig_i[:, j, :] = sig_i[a:a+int(n_trials/N), chans.index(model_chans[j]), :]
    
    del j
    
    # multi-linear regression
    rc, ri, r2 = SPF.mlr_analysis(te_w_i, te_w_o)
    w_es_s, w_ex_s = SPF.sig_extract_mlr(rc, te_sig_i, te_sig_o, ri)
    gf[i,:] = r2
    del rc, ri, te_w_i, te_w_o, te_sig_i, r2, w_es_s
    
    # power spectrum density
    w_p, fs = SPF.welch_p(w_ex_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=1024,
                          n_overlap=0, n_per_seg=1024)
    sig_p, fs = SPF.welch_p(te_sig_o, sfreq=sfreq, fmin=0, fmax=50, n_fft=1024,
                          n_overlap=0, n_per_seg=1024)
    
    # time-domain snr
    sig_snr_t = SPF.snr_time(te_sig_o)
    w_snr_t = SPF.snr_time(w_ex_s)
    snr_t_raise[i] = np.mean(w_snr_t - sig_snr_t)
    percent_t[i] = snr_t_raise[i] / np.mean(sig_snr_t) * 100
    del sig_snr_t, w_snr_t
    
    # frequency-domain snr
    sig_snr_f = SPF.snr_freq(sig_p, k=freq)
    w_snr_f = SPF.snr_freq(w_p, k=freq)
    snr_f_raise[i] = np.mean(w_snr_f - sig_snr_f)
    percent_f[i] = snr_f_raise[i] / np.mean(sig_snr_f) * 100
    del sig_snr_f, w_snr_f, w_p, sig_p, fs, w_ex_s
    
    # release RAM
    del mcee_chans, te_sig_o, model_chans, 
    
    # loop mart
    print(str(i+1) + 'th cross-validation complete!')
    
# release RAM
del N, a, freq, i, sfreq, sig_i, sig_o, signal_data, w, w_i, w_o, n_trials

#%% FP-Growth
if __name__ == '__main__':
    '''
    Call function 'find_frequent_itemsets()' to form frequent items
    '''
    frequent_itemsets = fpg.find_frequent_itemsets(cv_model_chans, minimum_support=5,
                                                   include_support=True)
    #print(type(frequent_itemsets))
    result = []
    # save results from generator into list
    for itemset, support in frequent_itemsets:  
        result.append((itemset, support))
    # ranking
    result = sorted(result, key=lambda i: i[0])
    print('FP-Growth complete!')


#%%******************** Details evaluation ********************%%#
#%% load local data (extract from .cnt file)
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
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
signal_data = f_data[:, :, :, 3140:3240]   

del n_chans, n_times, n_trials, n_events
#del f_data
freq = 1  # 10Hz

#%% pick channels
w_i = w[:,:,[chans.index('P2 '), chans.index('CP2'), chans.index('C2 ')], :]
w_o = w[:,:,chans.index('O2 '), :]

sig_i = signal_data[:,:,[chans.index('P2 '), chans.index('CP2'), chans.index('C2 ')], :]
sig_o = signal_data[:,:,chans.index('O2 '), :]

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
sig_snr_t = SPF.snr_time(sig_o)
w_snr_t = SPF.snr_time(w_ex_s)
ws_snr_t = SPF.snr_time(w_es_s)

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

#%% check waveform
sns.set(style='whitegrid')

fig, ax = plt.subplots(2, 1, figsize=(12, 12))

ax[0].plot(sig_o[freq, :, :].T, linewidth=1)
ax[0].plot(np.mean(sig_o[freq, :, :], axis=0), linewidth=4, color='black', label='mean')
ax[0].set_title('8Hz & POZ origin', fontsize=18)
ax[0].set_xlabel('Time/ms', fontsize=16)
ax[0].set_ylabel('Amplitude/uV', fontsize=16)
ax[0].set_ylim([-20, 20])
ax[0].legend(loc='best', fontsize=16)

ax[1].plot(w_ex_s[freq, :, :].T, linewidth=1)
ax[1].plot(np.mean(w_ex_s[freq, :, :], axis=0), linewidth=4, color='black', label='mean')
ax[1].set_title('8Hz & POZ extraction', fontsize=18)
ax[1].set_xlabel('Time/ms', fontsize=16)
ax[1].set_ylabel('Amplitude/uV', fontsize=16)
ax[1].set_ylim([-20, 20])
ax[1].legend(loc='best', fontsize=16)

fig.tight_layout()
plt.show()
#plt.savefig(r'F:\8-poz.png', dpi=600)
