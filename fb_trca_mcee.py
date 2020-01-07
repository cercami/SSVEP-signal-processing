# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:46:35 2020

TRCA + MCEE + 10-fold cross-validation

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

# initialization
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
#tar_chans = ['PZ ','POZ','O1 ','OZ ','O2 ']
#tar_chans = ['POZ', 'O1 ', 'OZ ', 'O2 ']
#tar_chans = ['O1 ', 'OZ ', 'O2 ']
#tar_chans = ['POZ', 'OZ ']

N = 10
n_bands = 10

n_events = 3
n_trials = 100
n_chans = len(tar_chans)

acc_cv = np.zeros((11,n_bands))
for nt in range(11):
    if nt == 0:  # 500ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_0.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6  # all chans(9), 1140-1640(500ms)
        ep = 1640
    elif nt == 1:  # 400ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_1.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1540
    elif nt == 2:  # 300ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_2.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1440
    elif nt == 3:  # 200ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_3.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1340
    elif nt == 4:  # 180ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_4.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1320
    elif nt == 5:  # 160ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_5.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1300
    elif nt == 6:  # 140ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_6.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1280
    elif nt == 7:  # 120ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_7.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1260
    elif nt == 8:  # 100ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_8.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1240
    elif nt == 9:  # 80ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_9.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1220
    elif nt == 10:  # 60ms
        eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_10.mat')
        mcee_sig = eeg['mcee_sig'] * 1e6
        ep = 1200
                
    chans = eeg['chan_info'].tolist() 
    del eeg

    # basic information
    sfreq = 1000

    # trca
    for nb in range(n_bands):
        # filter bank: 8-88Hz | IIR method, each band has 2Hz's buffer
        # change number of filter banks
        l_freq = [(8*(x+1)-2) for x in range(nb+1)]
        h_freq = 90
    
        # 5-D tension: (n_events, n_bands, n_trials, n_chans, n_times)
        fb_data = np.zeros((n_events, nb+1, n_trials, n_chans, mcee_sig.shape[3]))
    
        # apply Chebyshev I bandpass filter
        for i in range(nb+1):
            b, a = signal.cheby1(N=5, rp=0.5, Wn=[l_freq[i], h_freq], btype='bandpass',
                                 analog=False, output='ba', fs=sfreq)
            fb_data[:,i,:,:,:] = signal.filtfilt(b=b, a=a, x=mcee_sig, axis=-1)
            del b, a
        del i
    
        fb_data = fb_data[:,:,:,:,1140:ep]
    
        # cross-validation
        print('Running TRCA algorithm...')
        acc_temp = []
        for i in range(N):   
            # divide dataset
            a = i * 10
            # training dataset: (n_events, n_bands, n_trials, n_chans, n_times)
            tr_fb_data = fb_data[:,:,a:a+int(n_trials/N),:,:]
            # test dataset: (n_events, n_bands, n_trials, n_chans, n_times)
            te_fb_data = copy.deepcopy(fb_data)
            te_fb_data = np.delete(te_fb_data, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=2)
            # template data: (n_events, n_bands, n_chans, n_times)| basic unit: (n_chans, n_times)
            template = np.mean(tr_fb_data, axis=2)

            # main loop of TRCA
            acc = mcee.trca(tr_fb_data, te_fb_data)
            acc_temp.append(np.sum(acc))
            del acc
            # end                                
            print(str(i+1) + 'th cross-validation complete!')
    
        acc_cv[nt,nb] = np.sum(acc_temp)
        del acc_temp
        print(str(nb+1) + ' bands complete!\n')

#%% reshape results
acc_cv = acc_cv.T
acc_cv/=2700
acc_cv*=100