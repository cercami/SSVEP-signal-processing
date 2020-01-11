# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:14:37 2020

@author: lenovo
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

#%% load data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw_data.mat')
# (n_events, n_trials, n_chans, n_times)
raw_data = eeg['raw_data'] * 1e6
chans = eeg['chan_info'].tolist()
del eeg

#%% filtering respectively
n_events = int(raw_data.shape[0]/2)
n_trials = raw_data.shape[1]
n_chans = raw_data.shape[2]
n_times = raw_data.shape[3]
sfreq = 1000

f_data_60 = np.zeros((n_events, n_trials, n_chans, n_times))
f_data_40 = np.zeros((n_events, n_trials, n_chans, n_times))

for i in range(n_events):
    f_data_60[i,:,:,:] = filter_data(raw_data[i,:,:,:], sfreq=sfreq, l_freq=50,
             h_freq=90, n_jobs=6)
del i

for j in range(n_events):
    f_data_40[j,:,:,:] = filter_data(raw_data[j+2,:,:,:], sfreq=sfreq, l_freq=30,
             h_freq=90, n_jobs=6)
del j

del raw_data, n_times, n_chans

#%% mcee optimization
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

n_chans = len(tar_chans)
n_times = 2140

# (n_events, n_trials, n_chans, n_times)
mcee_sig = np.zeros((n_events, n_trials, n_chans, n_times))

# mcee optimization
for nt in range(11):
    # stepwise
    for ntc in range(len(tar_chans)):
        for nf in range(2):
            stim = nf  # 0 for 0 phase, 1 for pi phase
            target_channel = tar_chans[ntc]
            # load local data (extract from .cnt file)
            f_data = f_data_60[nf,:,:,2140:3140]
            w = f_data_60[:,:,1000:2000]
            signal_data = f_data[:,:,0:int(100*(nt+1))]
            ep = 1640

            # basic information
            sfreq = 1000

            # variables initialization
            w_o = w[:,chans.index(target_channel),:]
            w_temp = copy.deepcopy(w)
            w_i = np.delete(w_temp, chans.index(target_channel), axis=1)
            del w_temp

            sig_o = signal_data[:,chans.index(target_channel),:]
            f_sig_o = f_data[:,chans.index(target_channel),:]
            sig_temp = copy.deepcopy(signal_data)
            sig_i = np.delete(sig_temp, chans.index(target_channel), axis=1)
            del sig_temp

            mcee_chans = copy.deepcopy(chans)
            del mcee_chans[chans.index(target_channel)]

            snr = mcee.snr_time(sig_o)
            msnr = np.mean(snr)
  
            # use stepwise method to find channels
            model_chans, snr_change = mcee.stepwise_MCEE(chans=mcee_chans,
                    msnr=msnr, w=w, w_target=w_o, signal_data=sig_i, data_target=sig_o)
            del snr_change
            del w_i, sig_i, mcee_chans, snr, msnr

            # pick channels chosen from stepwise
            w_i = np.zeros((w.shape[0], len(model_chans), w.shape[2]))
            f_sig_i = np.zeros((f_data.shape[0], len(model_chans), f_data.shape[2]))

            for nc in range(len(model_chans)):
                w_i[:,nc,:] = w[:,chans.index(model_chans[nc]),:]
                f_sig_i[:,nc,:] = f_data[:,chans.index(model_chans[nc]),:]
            del nc

            # mcee main process
            rc, ri, r2 = SPF.mlr_analysis(w_i, w_o)
            w_es_s, w_ex_s = SPF.sig_extract_mlr(rc, f_sig_i, f_sig_o, ri)
            del rc, ri, r2, w_es_s

            # save optimized data
            mcee_sig[nf,:,ntc,:] = w_ex_s
            del w_ex_s, model_chans, f_sig_i, sig_o, w_i, w_o, w, signal_data, f_sig_o
            
    data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_%d.mat'%(nt)
    io.savemat(data_path, {'mcee_sig': mcee_sig,
                           'chan_info': tar_chans})