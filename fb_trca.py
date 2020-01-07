# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:04:13 2020

TRCA + 10-fold cross-validation

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io
from scipy import signal

import copy

import mcee

import matplotlib.pyplot as plt
import seaborn as sns

#%% load data (origin)
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
data = eeg['f_data'][:,:,:,2000:3640]
chans = eeg['chan_info'].tolist()
data *= 1e6  
del eeg

#%% initialization
n_events = data.shape[0]
n_trials = data.shape[1]

# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
tar_chans = ['PZ ', 'POZ', 'O1 ', 'OZ ', 'O2 ']
#tar_chans = ['POZ', 'O1 ', 'OZ ', 'O2 ']
#tar_chans = ['O1 ', 'OZ ', 'O2 ']
#tar_chans = ['POZ', 'OZ ']

n_chans = len(tar_chans)
n_times = data.shape[3]

#%% cross validation
sfreq = 1000

N = 10
n_bands = 10
acc_cv = np.zeros((11, n_bands))

# pick specific channels from origin data
pk_sig = np.zeros((data.shape[0], data.shape[1], n_chans, data.shape[3]))
for i in range(n_chans):
    pk_sig[:,:,i,:] = data[:,:,chans.index(tar_chans[i]),:]
del i, data 

# main loop for origin data | fixed channels, nb bands, nt times 
for nb in range(n_bands):  # form filter banks before extracting data
    # filter bank: 8-88Hz | IIR method, each band has 2Hz's buffer
    # change number of filter banks
    l_freq = [(8*(x+1)-2) for x in range(nb+1)]
    h_freq = 90
    # 5-D tension
    fb_data = np.zeros((n_events, nb+1, n_trials, n_chans, pk_sig.shape[3]))
    # apply Chebyshev I bandpass filter
    for i in range(nb+1):
        b, a = signal.cheby1(N=5, rp=0.5, Wn=[l_freq[i], h_freq], btype='bandpass',
                                 analog=False, output='ba', fs=sfreq)
        fb_data[:,i,:,:,:] = signal.filtfilt(b=b, a=a, x=pk_sig, axis=-1)
        del b, a
    del i
    
    # extract different lengths' data
    for nt in range(11):
        if nt == 0: # 500ms
            trca_data = fb_data[:,:,:,:,1140:]
            print('Data length: 500ms')
        elif nt == 1:  # 400ms
            trca_data = fb_data[:,:,:,:,1140:1540]
            print('Data length: 400ms')
        elif nt == 2:  # 300ms
            trca_data = fb_data[:,:,:,:,1140:1440]
            print('Data length: 300ms')
        elif nt == 3:  # 200ms
            trca_data = fb_data[:,:,:,:,1140:1340]
            print('Data length: 200ms')
        elif nt == 4:  # 180ms
            trca_data = fb_data[:,:,:,:,1140:1320]
            print('Data length: 180ms')
        elif nt == 5:  # 160ms
            trca_data = fb_data[:,:,:,:,1140:1300]
            print('Data length: 160ms')
        elif nt == 6:  # 140ms
            trca_data = fb_data[:,:,:,:,1140:1280]
            print('Data length: 140ms')
        elif nt == 7:  # 120ms
            trca_data = fb_data[:,:,:,:,1140:1260]
            print('Data length: 120ms')
        elif nt == 8:  # 100ms
            trca_data = fb_data[:,:,:,:,1140:1240]
            print('Data length: 100ms')
        elif nt == 9:  # 80ms
            trca_data = fb_data[:,:,:,:,1140:1220]
            print('Data length: 80ms')
        elif nt == 10:  # 60ms
            trca_data = fb_data[:,:,:,:,1140:1200]
            print('Data length: 60ms')
            
        # cross-validation + TRCA
        print('Running TRCA algorithm...')
        acc_temp = []
        for i in range(N):   
            # divide dataset
            a = i * 10
            # training dataset: (n_events, n_bands, n_trials, n_chans, n_times)
            tr_fb_data = trca_data[:,:,a:a+int(n_trials/N),:,:]
            # test dataset: (n_events, n_bands, n_trials, n_chans, n_times)
            te_fb_data = copy.deepcopy(trca_data)
            te_fb_data = np.delete(te_fb_data, [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9], axis=2)
            # template data: (n_events, n_bands, n_chans, n_times)| basic unit: (n_chans, n_times)
            template = np.mean(tr_fb_data, axis=2)

            # main loop of TRCA
            acc = mcee.trca(tr_fb_data, te_fb_data)
            acc_temp.append(np.sum(acc))
            del acc
            # end                                
            print(str(i+1) + 'th cross-validation complete!')
    
        acc_cv[nt, nb] = np.sum(acc_temp)
        del acc_temp
        print(str(nb+1) + ' bands complete!\n')
        
#%% reshape results
acc_cv = acc_cv.T
acc_cv/=2700
acc_cv*=100

#%%
k=0
np.where(acc_cv[:,k]==np.min(acc_cv[:,k]))
