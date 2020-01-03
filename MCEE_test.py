# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:50:32 2019
MCEE test:
    (1)MCEE
    (2)Cross-validation

@author: Brynhildr
"""

#%% Import third part module
import numpy as np
import scipy.io as io

import signal_processing_function as SPF 
import mcee

import copy

import fp_growth as fpg

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

