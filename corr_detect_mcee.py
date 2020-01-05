# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:17:22 2020

MCEE + Correlation detection

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

#%% prepare data
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','POZ','O1 ','OZ ','O2 ']
#tar_chans = ['POZ', 'O1 ', 'OZ ', 'O2 ']
#tar_chans = ['O1 ', 'OZ ', 'O2 ']
#tar_chans = ['POZ', 'OZ ']
# (n_events, n_trials, n_chans, n_times)
mcee_sig = np.zeros((3, 100, len(tar_chans), 1640))

N = 10

n_events = 3
n_trials = 100
n_chans = len(tar_chans)

#acc_cv = np.zeros((11))

# mcee optimization
for nt in range(11):
    # stepwise
    for ntc in range(len(tar_chans)):
        for nf in range(3):
            freq = nf  # 0 for 8Hz, 1 for 10Hz, 2 for 15Hz
            target_channel = tar_chans[ntc]
            # load local data (extract from .cnt file)
            eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
            f_data = eeg['f_data'][freq,:,:,2000:3640] * 1e6
            w = f_data[:,:,0:1000]
            if nt == 0:  # 500ms
                signal_data = f_data[:,:,1140:1640]
                ep = 1640
            elif nt == 1:  # 400ms
                signal_data = f_data[:,:,1140:1540]
                ep = 1540
            elif nt == 2:  # 300ms
                signal_data = f_data[:,:,1140:1440]
                ep = 1440
            elif nt == 3:  # 200ms
                signal_data = f_data[:,:,1140:1340]
                ep = 1340
            elif nt == 4:  # 180ms
                signal_data = f_data[:,:,1140:1320]
                ep = 1320
            elif nt == 5:  # 160ms
                signal_data = f_data[:,:,1140:1300]
                ep = 1300
            elif nt == 6:  # 140ms
                signal_data = f_data[:,:,1140:1280]
                ep = 1280
            elif nt == 7:  # 120ms
                signal_data = f_data[:,:,1140:1260]
                ep = 1260
            elif nt == 8:  # 100ms
                signal_data = f_data[:,:,1140:1240]
                ep = 1240
            elif nt == 9:  # 80ms
                signal_data = f_data[:,:,1140:1220]
                ep = 1220
            elif nt == 10:  # 60ms
                signal_data = f_data[:,:,1140:1200]
                ep = 1200
                
            chans = eeg['chan_info'].tolist() 
            del eeg

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
            
    data_path = r'I:\SSVEP\dataset\preprocessed_data\weisiwen\mcee_5chan_%d.mat'%(nt)
    io.savemat({'mcee_sig': mcee_sig})