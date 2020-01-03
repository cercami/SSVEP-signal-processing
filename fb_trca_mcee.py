# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:46:35 2020

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

#%% initialization
tar_chans = ['PZ ','POZ','O1 ','OZ ','O2 ']
# (n_events, n_trials, n_chans, n_times)
mcee_sig = np.zeros((3, 100, len(tar_chans), 160))

#%% loop condition
for ntc in range(len(tar_chans)):
    for nf in range(3):
        freq = nf  # 0 for 8Hz, 1 for 10Hz, 2 for 15Hz
        target_channel = tar_chans[ntc]

        # load local data (extract from .cnt file)
        eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
        f_data = eeg['f_data'][freq,:,:,2000:3640] * 1e6
        w = f_data[:,:,0:1000]
        signal_data = f_data[:,:,1140:1300]

        #for nt in range(5):
            #f_data = eeg['f_data'][freq,:,:,2000:3140+int((nt+1)*100)] * 1e6
            #w = f_data[:,:,0:1000]
            #signal_data = f_data[:,:,1140:1140+int((nt+1)*100)]
        
        chans = eeg['chan_info'].tolist() 
        del eeg

        # basic information
        sfreq = 1000

        # variables initialization
        w_o = w[:, chans.index(target_channel), :]
        w_temp = copy.deepcopy(w)
        w_i = np.delete(w_temp, chans.index(target_channel), axis=1)
        del w_temp

        sig_o = signal_data[:, chans.index(target_channel), :]
        sig_temp = copy.deepcopy(signal_data)
        sig_i = np.delete(sig_temp, chans.index(target_channel), axis=1)
        del sig_temp

        mcee_chans = copy.deepcopy(chans)
        del mcee_chans[chans.index(target_channel)]

        snr = mcee.snr_time(sig_o)
        msnr = np.mean(snr)
  
        # use stepwise method to find channels
        model_chans, snr_change = mcee.stepwise_MCEE(chans=mcee_chans, msnr=msnr, w=w,
                                 w_target=w_o, signal_data=sig_i, data_target=sig_o)
        del snr_change
        del w_i, sig_i, mcee_chans, snr, msnr

        # pick channels chosen from stepwise
        w_i = np.zeros((w.shape[0], len(model_chans), w.shape[2]))
        sig_i = np.zeros((signal_data.shape[0], len(model_chans), signal_data.shape[2]))

        for nc in range(len(model_chans)):
            w_i[:,nc,:] = w[:,chans.index(model_chans[nc]),:]
            sig_i[:,nc,:] = signal_data[:,chans.index(model_chans[nc]),:]
        del nc

        # mcee main process
        rc, ri, r2 = SPF.mlr_analysis(w_i, w_o)
        w_es_s, w_ex_s = SPF.sig_extract_mlr(rc, sig_i, sig_o, ri)
        del rc, ri, r2, w_es_s

        # save optimized data
        mcee_sig[nf,:,ntc,:] = w_ex_s
        del w_ex_s, model_chans, sig_i, sig_o, w_i, w_o, w, signal_data

#%% trca
N = 10
n_bands = 3
acc_cv = np.zeros((n_bands))

pk_sig = mcee_sig

n_events = 3
n_trials = 100
n_chans = len(tar_chans)

  
for nb in range(n_bands):
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
    
    acc_cv[nb] = np.sum(acc_temp)
    del acc_temp
    print(str(nb+1) + ' bands complete!\n')
    


