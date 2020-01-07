# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:46:50 2019

Three methods to build filter bank for TRCA or FB-CCA target identification algorithm

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io
from scipy import signal

import copy

import matplotlib.pyplot as plt
import seaborn as sns

from mne.filter import filter_data

#%% Load data
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')
data = eeg['raw_data']
chans = eeg['chan_info'].tolist()
data *= 1e6  

del eeg

sfreq = 1000

sig = data[:, :, :, 3140:3640]   

n_events = sig.shape[0]
n_trials = sig.shape[1]
n_times = sig.shape[3]

del data

# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
n_chans = len(tar_chans)

pk_sig = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_chans):
    pk_sig[:,:,i,:] = sig[:, :, chans.index(tar_chans[i]), :]
del i

#%% Filter bank:8-88Hz (10 bands/ 8Hz each)
#%% IIR method
# each band contain 8Hz's useful information with 1Hz's buffer zone
n_bands = 10
# for example, 1st band is actually 7Hz-17Hz
sl_freq = [(8*(x+1)-4) for x in range(10)]  # the lowest stop frequencies
pl_freq = [(8*(x+1)-2) for x in range(10)]  # the lowest pass frequencies
ph_freq = [(8*(x+2)+2) for x in range(10)]  # the highest pass frequencies
sh_freq = [(8*(x+2)+4) for x in range(10)]  # the highest stop frequencies

# 5-D tension
fb_data = np.zeros((n_bands, n_events, n_trials, n_chans, n_times))

# make data through filter bank
for i in range(n_bands):
    # design Chebyshev-II iir band-pass filter
    b, a = signal.iirdesign(wp=[pl_freq[i], ph_freq[i]], ws=[sl_freq[i], sh_freq[i]],
                        gpass=3, gstop=18, analog=False, ftype='cheby1', fs=sfreq)
    # filter data forward and backward to achieve zero-phase
    fb_data[i,:,:,:,:] = signal.filtfilt(b=b, a=a, x=pk_sig, axis=-1)
    del a, b
    
del i, sl_freq, pl_freq, ph_freq, sh_freq
print('Filter bank construction complete!')

#%% IIR method 2
# each band from m*8Hz to 90Hz
n_bands = 10
l_freq = [(8*(x+1)-2) for x in range(n_bands)]
h_freq = 90

sns.set(style='whitegrid')

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

# 5-D tension
fb_data = np.zeros((n_events, n_bands, n_trials, n_chans, n_times))

# design Chebyshev I bandpass filter
for i in range(n_bands):
    b, a = signal.cheby1(N=5, rp=0.5, Wn=[l_freq[i], h_freq], btype='bandpass',
                         analog=False, output='ba', fs=sfreq)
    # filter data forward and backward to achieve zero-phase
    fb_data[:,i,:,:,:] = signal.filtfilt(b=b, a=a, x=pk_sig, axis=-1)
    # plot figures
    w, h = signal.freqz(b, a)
    freq = (w*sfreq) / (2*np.pi)
    
    ax[0].plot(freq, 20*np.log10(abs(h)), label='band %d'%(i+1))
    ax[0].set_title('Frequency Response', fontsize=16)
    ax[0].set_ylabel('Amplitude/dB', fontsize=16)
    ax[0].set_xlabel('Frequency/Hz', fontsize=16)
    ax[0].set_xlim([0, 200])
    ax[0].set_ylim([-60, 5])
    ax[0].legend(loc='best')

    ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, label='band %d'%(i+1))
    ax[1].set_ylabel('Anlge(degrees)', fontsize=16)
    ax[1].set_xlabel('Frequency/Hz', fontsize=16)
    ax[1].set_xlim([0, 200])
    ax[1].legend(loc='best')
    
    del b, a, w, h, freq
del i
plt.show()

#%% FIR method
# each band contain 8Hz's useful information with 1Hz's buffer zone
n_bands = 10

l_freq = [(8*(x+1)-2) for x in range(n_bands)]
h_freq = 90

# 5-D tension
fb_data = np.zeros((n_bands, n_events, n_trials, n_chans, n_times))

# make data through filter bank
for i in range(n_bands):
    for j in range(pk_sig.shape[0]):
        fb_data[i,j,:,:,:] = filter_data(pk_sig[j,:,:,:], sfreq=sfreq,
               l_freq=l_freq[i], h_freq=h_freq[i], n_jobs=4, filter_length='500ms',
               l_trans_bandwidth=2, h_trans_bandwidth=2, method='fir',
               phase='zero', fir_window='hamming', fir_design='firwin2',
               pad='reflect_limited')
