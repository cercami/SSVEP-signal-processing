# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:06:59 2019

This program is used to process data

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

#%% prevent ticking 'F5'
???

#%% Load benchmark dataset & relative information
# load data from .mat file
eeg = io.loadmat(r'E:\dataset\data\S15.mat')
info = io.loadmat(r'E:\dataset\Freq_Phase.mat')

data = eeg['data']
# (64, 1500, 40, 6) = (n_chans, n_times, n_events, n_blocks)
# total trials = n_conditions x n_blocks = 240
# all epochs are down-sampled to 250 Hz, HOLLY SHIT!

# reshape data array: (n_events, n_epochs, n_chans, n_times)
data = data.transpose((2, 3, 0, 1))  

# combine data array: np.concatenate(X, Y, axis=)

# condition infomation
sfreq = 250
freqs = info['freqs'].T
phases = info['phases'].T
del eeg, info


#%% load channels information from .txt file
channels = {}
file = open(r'F:\SSVEP\dataset\channel_info\weisiwen_chans.txt')
for line in file.readlines():
    line = line.strip()
    v = str(int(line.split(' ')[0]) - 1)
    k = line.split(' ')[1]
    channels[k] = v
file.close()

del v, k, file, line       # release RAM
     

#%% Load multiple data file & also can be used to process multiple data
# CAUTION: may lead to RAM crash (5-D array takes more than 6125MB)
# Now I know why people need 32G's RAM...PLEASE SKIP THIS PART!!!
filepath = r'E:\dataset\data'

filelist = []
for file in os.listdir(filepath):
    full_path = os.path.join(filepath, file)
    filelist.append(full_path)

i = 0
eeg = np.zeros((35, 64, 1500, 40, 6))
for file in filelist:
    temp = io.loadmat(file)
    eeg[i,:,:,:,:] = temp['data']
    i += 1
    
# add more codes here to achieve multiple data processing (PLEASE DON'T)
    
del temp, i, file, filelist, filepath, full_path


#%% load local data (extract from .cnt file)
# BEGIN FROM HERE!
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\raw_data.mat')

data = eeg['raw_data']
data *= 1e6  # reset unit

del eeg

chans = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\chan_info.mat')
chans = chans['chan_info'].tolist()


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

data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat'
io.savemat(data_path, {'f_data':f_data})

#%% Correlation binarization
def binarization(X):
    compare = np.zeros((X.shape[0], X.shape[1]))
    for i in range(n_chans):
        for j in range(n_chans):
            if X[i,j] < 0:
                compare[i,j] = 0
            else:
                compare[i,j] = X[i,j]
    return compare

#%% Inter-channel correlation analysis: Spearman correlation coefficient
w1_corr = SPF.corr_coef(w1, 'spearman')
w2_corr = SPF.corr_coef(w2, 'spearman')
w3_corr = SPF.corr_coef(w3, 'spearman')
w_corr = SPF.corr_coef(w, 'spearman')

sig_corr = SPF.corr_coef(signal_data, mode='spearman')

compare_w1 = binarization(w1_corr - sig_corr)
compare_w2 = binarization(w2_corr - sig_corr)
compare_w3 = binarization(w2_corr - sig_corr)
compare = binarization(w_corr - sig_corr)

# save data
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\64chan_corr.mat'
io.savemat(data_path, {'signal':sig_corr,
                       'w1':w1_corr,
                       'w2':w2_corr,
                       'w3':w3_corr,
                       'w':w_corr,
                       'w1_sub':compare_w1,
                       'w2_sub':compare_w2,
                       'w3_sub':compare_w3,
                       'w_sub':compare})

del w1_corr, w2_corr, w3_corr, w_corr, sig_corr
del comprare_w1, compare_w2, compare_w3, compare


#%% reload full-chan-correlation data
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\64chan_corr.mat')
w_sub = data['w_sub']

del data


#%% Automatically pick estimate channel and target channel


# pick input channels: C1, Cz, C2, C4, CP5
# choose output channels: Oz

# w1 model data: 0-1000ms
w1_i = w1[:,:,[45,54],:]
w1_o = w1[:,:,47,:]
w1_total = w1[:,:,[34,35,36,43,44,47],:]
w_total = w[:,:,[34,35,36,43,44,47],:]

# w2 model data: 1000-2000ms
w2_i = w2[:,:,[34,35,36,43,44],:]
w2_o = w2[:,:,47,:]
w2_total = w2[:,:,[34,35,36,43,44,47],:]

# w3 model data: 2000-3000ms
w3_i = w3[:,:,[34,35,36,43,44],:]
w3_o = w3[:,:,47,:]
w3_total = w3[:,:,[34,35,36,43,44,47],:]

# signal part data: 3000-6000ms
sig_i = signal_data[:,:,[34,35,36,43,44],:]
sig_o = signal_data[:,:,47,:]
sig_total = signal_data[:,:,[34,35,36,43,44,47],:]

# save data
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\model_data.mat'
io.savemat(data_path, {'w1_i':w1_i, 'w1_o':w1_o, 'w1_total':w1_total,
                       'w2_i':w2_i, 'w2_o':w2_o, 'w2_total':w2_total,
                       'w3_i':w3_i, 'w3_o':w3_o, 'w3_total':w3_total,
                       'sig_i':sig_i, 'sig_o':sig_o, 'sig_total':sig_total})
    
# release RAM
#del w1, w2, w3, signal_data
del w1_total, w2_total, w3_total
#del w_sub

#%% Prepare for checkboard plot (Spearman method)
#w1_pick_corr = SPF.corr_coef(w1_total, 'spearman')
#w2_pick_corr = SPF.corr_coef(w2_total, 'spearman')
#w3_pick_corr = SPF.corr_coef(w3_total, 'spearman')
w_pick_corr = SPF.corr_coef(w_total, 'spearman')

sig_pick_corr = SPF.corr_coef(sig_total, 'spearman')

data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\pick_chan_corr.mat'
io.savemat(data_path, {'w':w_pick_corr, 'sig':sig_pick_corr})
    
#del w1_pick_corr, w2_pick_corr, w3_pick_corr, sig_pick_corr
del w_pick_corr, sig_pick_corr, sig_total


#%% Spatial filter: multi-linear regression method

# regression coefficient, intercept, R^2
rc_w1, ri_w1, r2_w1 = SPF.mlr_analysis(w1_i, w1_o)
# w1 estimate & extract data: (n_events, n_epochs, n_times)
w1_es_w1, w1_ex_w1 = SPF.sig_extract_mlr(rc_w1, w1_i, w1_o, ri_w1)

# w1-w2
w1_es_w2, w1_ex_w2 = SPF.sig_extract_mlr(rc_w1, w2_i, w2_o, ri_w1)

# w1-w3
w1_es_w3, w1_ex_w3 = SPF.sig_extract_mlr(rc_w1, w3_i, w3_o, ri_w1)

# w2-w2
rc_w2, ri_w2, r2_w2 = SPF.mlr_analysis(w2_i, w2_o)
w2_es_w2, w2_ex_w2 = SPF.sig_extract_mlr(rc_w2, w2_i, w2_o, ri_w2)

# w2-w3
w2_es_w3, w2_ex_w3 = SPF.sig_extract_mlr(rc_w2, w3_i, w3_o, ri_w2)

# w3-w3
rc_w3, ri_w3, r2_w3 = SPF.mlr_analysis(w3_i, w3_o)
w3_es_w3, w3_ex_w3 = SPF.sig_extract_mlr(rc_w3, w3_i, w3_o, ri_w3)

# w1-s
w1_es_s, w1_ex_s = SPF.sig_extract_mlr(rc_w1, sig_i, sig_o, ri_w1)
# w2-s
w2_es_s, w2_ex_s = SPF.sig_extract_mlr(rc_w2, sig_i, sig_o, ri_w2)
# w3-s
w3_es_s, w3_ex_s = SPF.sig_extract_mlr(rc_w3, sig_i, sig_o, ri_w3)

# save data
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\MLR_model.mat'
io.savemat(data_path, {'rc_w1':rc_w1, 'rc_w2':rc_w2, 'rc_w3':rc_w3,
                       'ri_w1':ri_w1, 'ri_w2':ri_w2, 'ri_w3':ri_w3,
                       'r2_w1':r2_w1, 'r2_w2':r2_w2, 'r2_w3':r2_w3,
                       'w1_es_w1':w1_es_w1, 'w1_es_w2':w1_es_w2, 'w1_es_w3':w1_es_w3,
                       'w2_es_w2':w2_es_w2, 'w2_es_w3':w2_es_w3, 
                       'w3_es_w3':w3_es_w3,
                       'w1_ex_w1':w1_ex_w1, 'w1_ex_w2':w1_ex_w2, 'w1_ex_w3':w1_ex_w3,
                       'w2_ex_w2':w2_ex_w2, 'w2_ex_w3':w2_ex_w3, 
                       'w3_ex_w3':w3_ex_w3,
                       'w1_es_s':w1_es_s, 'w2_es_s':w1_es_s, 'w3_es_s':w3_es_s,
                       'w1_ex_s':w1_ex_s, 'w2_ex_s':w2_ex_s, 'w3_ex_s':w3_ex_s})
    
# release RAM
del rc_w1, rc_w2, rc_w3, ri_w1, ri_w2, ri_w3, r2_w1, r2_w2, r2_w3
del w1_ex_w1, w1_ex_w2, w1_ex_w3, w2_ex_w2, w2_ex_w3, w3_ex_w3


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
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\inver_array_model.mat'
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
#data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\cos_sim_mlr.mat'
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\cos_sim_ia.mat'
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
#del w1_i, w2_i, w3_i, w1_o, w2_o, w3_o, sig_i


#%% Power spectrum density
# load data
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\MLR_model.mat')
w1_ex_s = data['w1_ex_s']
w2_ex_s = data['w2_ex_s']
w3_ex_s = data['w3_ex_s']
del data

data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\34-35-36-43-44__POZ\model_data.mat')
sig_o = data['sig_o']
del data

sfreq = 1000

w1_p, fs = SPF.welch_p(w1_ex_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                      n_overlap=1000, n_per_seg=2000)
w2_p, fs = SPF.welch_p(w2_ex_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                      n_overlap=1000, n_per_seg=2000)
w3_p, fs = SPF.welch_p(w3_ex_s, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                      n_overlap=1000, n_per_seg=2000)
sig_p, fs = SPF.welch_p(sig_o, sfreq=sfreq, fmin=0, fmax=50, n_fft=3000,
                       n_overlap=1000, n_per_seg=2000)
#plt.plot(fs[1,1,:], np.mean(sig_p[0,:,:], axis=0), label='origin')
#plt.plot(fs[1,1,:], np.mean(w1_p[0,:,:], axis=0), label='w1')
#plt.plot(fs[1,1,:], np.mean(w2_p[0,:,:], axis=0), label='w2')
#plt.plot(fs[1,1,:], np.mean(w3_p[0,:,:], axis=0), label='w3')
#plt.legend(loc='best')

# save data
data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\psd_mlr.mat'
#data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\psd_ia.mat'
io.savemat(data_path, {'w1':w1_p, 'w2':w2_p, 'w3':w3_p, 'sig':sig_p, 'fs':fs})


#%% Precise FFT transform


#%% SNR in time domain
def snr(X):
    '''
    Two method for SNR computation
        (1) Compute SNR and return time sequency
        (2) Use superimposed average method:
    :param X: input data (one-channel) (n_events, n_epochs, n_times) 
    :param mode: choose method
    '''
    snr = np.zeros((X.shape[0], X.shape[2]))    # (n_events, n_times)
    # basic object's size: (n_epochs, n_times)
    for i in range(X.shape[0]):                 # i for n_events
        ex = np.mat(np.mean(X[i,:,:], axis=0))  # (1, n_times)
        temp = np.mat(np.ones((1, X.shape[1]))) # (1, n_epochs)
        minus = (temp.T * ex).A                 # (n_epochs, n_times)
        ex = (ex.A) ** 2
        var = np.mean((X[i,:,:] - minus)**2, axis=0)
        snr[i,:] = ex/var
    return snr
                
# load data
data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__POZ\model_data.mat')
sig_o = data['sig_o']
del data

data = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\14_18__POZ\MLR_model.mat')
w1 = data['w1_ex_s']
w2 = data['w2_ex_s']
w3 = data['w3_ex_s']
del data

# original signal snr
snr_o = snr(sig_o)
# w1-s
snr_w1 = snr(w1)
# w2-s
snr_w2 = snr(w2)
# w3-s
snr_w3 = snr(w3)

# save data
#data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\snr_t_mlr.mat'
#data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\snr_t_ia.mat'
#io.savemat(data_path, {'origin':snr_o, 'w1':snr_w1, 'w2':snr_w2, 'w3':snr_w3})

# release RAM
#del snr_o, snr_w1, snr_w2, snr_w3
#del sig_o
#del w1_es_s, w2_es_s, w3_es_s, w1_ex_s, w2_ex_s, w3_ex_s 


#%% SNR in frequency domain

# save data
#data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\snr_f_mlr.mat'
#data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\snr_f_ia.mat'
#io.savemat(data_path, {'origin':snr_o, 'w1':snr_w1, 'w2':snr_w2, 'w3':snr_w3})

# release RAM
del w1_p, w2_p, w3_p, sig_p
del fs
