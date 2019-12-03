# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:24:52 2019

This program is used to timing
@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io
import pandas as pd

from mne.filter import filter_data

import signal_processing_function as SPF 

import time


#%% timing
start = time.clock()


#%% load local data (extract from .cnt file)
# BEGIN FROM HERE!
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')

data = eeg['raw_data']
data *= 1e6  # reset unit

del eeg

chans = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\chan_info.mat')
chans = chans['chan_info'].tolist()

del chans

# basic info
sfreq = 1000

n_events = data.shape[0]
n_trials = data.shape[1]
n_chans = data.shape[2]
n_times = data.shape[3]


#%% Data preprocessing
# filtering
f_data = np.zeros((n_events, n_trials, n_chans, n_times))
for i in range(n_events):
    f_data[i,:,:,:] = filter_data(data[i,:,:,:], sfreq=sfreq, l_freq=5,
                      h_freq=40, n_jobs=6)

del i

# get data for linear regression
w1 = f_data[:,:,:,0:1000]           # 0-1s
w2 = f_data[:,:,:,1000:2000]        # 1-2s
w3 = f_data[:,:,:,2000:3000]        # 2-3s
w = f_data[:,:,:,0:3000]

# get data for comparision
signal_data = f_data[:,:,:,3000:]   # 3-6s

del f_data, data
del n_events, n_trials, n_times, n_chans

print('Filtering finished')


#%% Inter-channel correlation analysis: Spearman correlation coefficient
def target_corr(X, chan, method):
    '''
    choose one channel to compute inter-channel correlation
    X: (n_chans, n_times)
    chan: channel's serial number
    method: str, pearson or spearman
    '''
    corr = np.zeros((X.shape[0]))
    target = pd.DataFrame(np.mat(X[chan,:]))
    for i in range(X.shape[0]):
        temp = pd.DataFrame(np.mat(X[i,:]))
        corr[i] = target.corrwith(temp, axis=1, method=method)
                
    del temp
    return corr
    
#%%
#w1_corr = SPF.corr_coef(w1, 'spearman')
#w2_corr = SPF.corr_coef(w2, 'spearman')
#w3_corr = SPF.corr_coef(w3, 'spearman')

w_corr = target_corr(w[0,0,:,:], chan=47, method='spearman')

sig_corr = target_corr(signal_data[0,0,:,:], chan=47, method='spearman')

#compare_w1 = binarization(w1_corr - sig_corr)
#compare_w2 = binarization(w2_corr - sig_corr)
#compare_w3 = binarization(w2_corr - sig_corr)
compare_corr = w_corr - sig_corr

# save data
data_path = r'F:\64chan_corr.mat'
io.savemat(data_path, {'signal':sig_corr, 'w':w_corr, 'w_sub':compare_corr})
    
#del w1_corr, w2_corr, w3_corr, w_corr, sig_corr
#del comprare_w1, compare_w2, compare_w3, compare
#del w_corr, sig_corr, compare_corr, w

print('Correlation computation finished')


#%% reload full-chan-correlation data
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\64chan_corr.mat')
w_sub = data['w_sub']

del data


#%% Automatically pick estimate channel and target channel


# pick input channels: C1, Cz, C2, C4, CP5
# choose output channels: Oz

# w1 model data: 0-1000ms
w1_i = w1[:,:,[34,35,36,43,44],:]
w1_o = w1[:,:,53,:]
w1_total = w1[:,:,[34,35,36,43,44,53],:]

# w2 model data: 1000-2000ms
w2_i = w2[:,:,[34,35,36,43,44],:]
w2_o = w2[:,:,53,:]
w2_total = w2[:,:,[34,35,36,43,44,53],:]

# w3 model data: 2000-3000ms
w3_i = w3[:,:,[34,35,36,43,44],:]
w3_o = w3[:,:,53,:]
w3_total = w3[:,:,[34,35,36,43,44,53],:]

# signal part data: 3000-6000ms
sig_i = signal_data[:,:,[34,35,36,43,44],:]
sig_o = signal_data[:,:,53,:]
sig_total = signal_data[:,:,[34,35,36,43,44,53],:]

# save data
data_path = r'F:\model_data.mat'
io.savemat(data_path, {'w1_i':w1_i, 'w1_o':w1_o, 'w1_total':w1_total,
                       'w2_i':w2_i, 'w2_o':w2_o, 'w2_total':w2_total,
                       'w3_i':w3_i, 'w3_o':w3_o, 'w3_total':w3_total,
                       'sig_i':sig_i, 'sig_o':sig_o, 'sig_total':sig_total})
    
# release RAM
del w1, w2, w3, signal_data
del w_sub

print('Channels seletion finished')


#%% Prepare for checkboard plot (Spearman method)
w1_pick_corr = SPF.corr_coef(w1_total, 'spearman')
w2_pick_corr = SPF.corr_coef(w2_total, 'spearman')
w3_pick_corr = SPF.corr_coef(w3_total, 'spearman')

sig_pick_corr = SPF.corr_coef(sig_total, 'spearman')

data_path = r'F:\pick_chan_corr.mat'
io.savemat(data_path, {'w1':w1_pick_corr,
                       'w2':w2_pick_corr,
                       'w3':w3_pick_corr,
                       'sig':sig_pick_corr})
    
del w1_pick_corr, w2_pick_corr, w3_pick_corr, sig_pick_corr
del w1_total, w2_total, w3_total, sig_total

print('Checkboard materials preparation finished')


#%% Spatial filter: inverse array method
# filter coefficient
sp_w1 = SPF.inv_spa(w1_i, w1_o)
# w1 estimate & extract data: (n_events, n_epochs, n_times)
w1_es_w1, w1_ex_w1 = SPF.sig_extract_ia(sp_w1, w1_i, w1_o)
# w1 model's goodness of fit
gf_w1 = SPF.fit_goodness(w1_o, w1_es_w1, chans=5)

# w1-w2:
w1_es_w2, w1_ex_w2 = SPF.sig_extract_ia(sp_w1, w2_i, w2_o)

# w1-w3:
w1_es_w3, w1_ex_w3 = SPF.sig_extract_ia(sp_w1, w3_i, w3_o)

# w2-w2:
sp_w2 = SPF.inv_spa(w2_i, w2_o)
w2_es_w2, w2_ex_w2 = SPF.sig_extract_ia(sp_w2, w2_i, w2_o)
gf_w2 = SPF.fit_goodness(w2_o, w2_es_w2, chans=5)

# w2-w3:
w2_es_w3, w2_ex_w3 = SPF.sig_extract_ia(sp_w2, w3_i, w3_o)

# w3-w3:
sp_w3 = SPF.inv_spa(w3_i, w3_o)
w3_es_w3, w3_ex_w3 = SPF.sig_extract_ia(sp_w3, w3_i, w3_o)
gf_w3 = SPF.fit_goodness(w3_o, w3_es_w3, chans=5)

# w1-s
w1_es_s, w1_ex_s = SPF.sig_extract_ia(sp_w1, sig_i, sig_o)
# w2-s
w2_es_s, w2_ex_s = SPF.sig_extract_ia(sp_w2, sig_i, sig_o)
# w3-s
w3_es_s, w3_ex_s = SPF.sig_extract_ia(sp_w3, sig_i, sig_o)

# save data
data_path = r'F:\inver_array_model.mat'
io.savemat(data_path, {'sp_w1':sp_w1, 'sp_w2':sp_w2, 'sp_w3':sp_w3,
                       'gf_w1':gf_w1, 'gf_w2':gf_w2, 'gf_w3':gf_w3,
                       'w1_es_w1':w1_es_w1, 'w1_es_w2':w1_es_w2, 'w1_es_w3':w1_es_w3,
                       'w2_es_w2':w2_es_w2, 'w2_es_w3':w2_es_w3, 
                       'w3_es_w3':w3_es_w3,
                       'w1_ex_w1':w1_ex_w1, 'w1_ex_w2':w1_ex_w2, 'w1_ex_w3':w1_ex_w3,
                       'w2_ex_w2':w2_ex_w2, 'w2_ex_w3':w2_ex_w3, 
                       'w3_ex_w3':w3_ex_w3,
                       'w1_es_s':w1_es_s, 'w2_es_s':w1_es_s, 'w3_es_s':w3_es_s,
                       'w1_ex_s':w1_ex_s, 'w2_ex_s':w2_ex_s, 'w3_ex_s':w3_ex_s})
    
# release RAM
del sp_w1, sp_w2, sp_w3, gf_w1, gf_w2, gf_w3
del w1_ex_w1, w1_ex_w2, w1_ex_w3, w2_ex_w2, w2_ex_w3, w3_ex_w3
del w1_i, w2_i, w3_i, sig_i

print('Inverse array model finished')


#%% Cosine similarity (background part)
# w1 estimate (w1 model) & w1 original, mlr, normal similarity, the same below
w1_w1_nsim = SPF.cos_sim(w1_o, w1_es_w1, mode='normal')
w1_w2_nsim = SPF.cos_sim(w2_o, w1_es_w2, mode='normal')
w1_w3_nsim = SPF.cos_sim(w3_o, w1_es_w3, mode='normal')

w2_w2_nsim = SPF.cos_sim(w2_o, w2_es_w2, mode='normal')
w2_w3_nsim = SPF.cos_sim(w3_o, w2_es_w3, mode='normal')

w3_w3_nsim = SPF.cos_sim(w3_o, w3_es_w3, mode='normal')

# w1 estimate (w1 model) & w1 original, mlr, Tanimoto, the same below
w1_w1_tsim = SPF.cos_sim(w1_o, w1_es_w1, mode='tanimoto')
w1_w2_tsim = SPF.cos_sim(w2_o, w1_es_w2, mode='tanimoto')
w1_w3_tsim = SPF.cos_sim(w3_o, w1_es_w3, mode='tanimoto')

w2_w2_tsim = SPF.cos_sim(w2_o, w2_es_w2, mode='tanimoto')
w2_w3_tsim = SPF.cos_sim(w3_o, w2_es_w3, mode='tanimoto')

w3_w3_tsim = SPF.cos_sim(w3_o, w3_es_w3, mode='tanimoto')

# save data
data_path = r'F:\cos_sim_ia.mat'
io.savemat(data_path, {'w1_w1_nsim':w1_w1_nsim, 'w1_w2_nsim':w1_w2_nsim, 'w1_w3_nsim':w1_w3_nsim,
                       'w2_w2_nsim':w2_w2_nsim, 'w2_w3_nsim':w2_w3_nsim,
                       'w3_w3_nsim':w3_w3_nsim,
                       'w1_w1_tsim':w1_w1_tsim, 'w1_w2_tsim':w1_w2_tsim, 'w1_w3_tsim':w1_w3_tsim,
                       'w2_w2_tsim':w2_w2_tsim, 'w2_w3_tsim':w2_w3_tsim,
                       'w3_w3_tsim':w3_w3_tsim})

# release RAM
del w1_w1_nsim, w1_w2_nsim, w1_w3_nsim, w2_w2_nsim, w2_w3_nsim, w3_w3_nsim
del w1_w1_tsim, w1_w2_tsim, w1_w3_tsim, w2_w2_tsim, w2_w3_tsim, w3_w3_tsim
del w1_es_w1, w1_es_w2, w1_es_w3, w2_es_w2, w2_es_w3, w3_es_w3
del w1_o, w2_o, w3_o

print('Cosine similarity computaion finished')


#%% Power spectrum density
w1_p, fs = SPF.welch_p(w1_ex_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                      n_overlap=250, n_per_seg=500)
w2_p, fs = SPF.welch_p(w2_ex_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                      n_overlap=250, n_per_seg=500)
w3_p, fs = SPF.welch_p(w3_ex_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                      n_overlap=250, n_per_seg=500)
sig_p, fs = SPF.welch_p(sig_o, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                       n_overlap=250, n_per_seg=500)

# save data
data_path = r'F:\psd_ia.mat'
io.savemat(data_path, {'w1':w1_p, 'w2':w2_p, 'w3':w3_p, 'sig':sig_p, 'fs':fs})

del w1_p, w2_p, w3_p, sig_p, fs
del sfreq

print('PSD computation finished')


#%% SNR in time domain
# original signal snr
snr_o = SPF.snr_time(sig_o, mode='time')
# w1-s
snr_w1 = SPF.snr_time(w1_ex_s, mode='time')
# w2-s
snr_w2 = SPF.snr_time(w2_ex_s, mode='time')
# w3-s
snr_w3 = SPF.snr_time(w3_ex_s, mode='time')

# save data
data_path = r'F:\snr_t_ia.mat'
io.savemat(data_path, {'origin':snr_o, 'w1':snr_w1, 'w2':snr_w2, 'w3':snr_w3})

# release RAM
del snr_o, snr_w1, snr_w2, snr_w3
del sig_o
del w1_es_s, w2_es_s, w3_es_s, w1_ex_s, w2_ex_s, w3_ex_s

print('Time-domain SNR computation finished')
print('Program complete')

#%% timing
end = time.clock()
print('Running time: ' + str(end - start) + 's')