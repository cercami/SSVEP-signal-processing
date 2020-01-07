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

import seaborn as sns

#%% prepare data
# pick channels from parietal and occipital areas
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']
#tar_chans = ['POZ', 'O1 ', 'OZ ', 'O2 ']
#tar_chans = ['O1 ', 'OZ ', 'O2 ']
#tar_chans = ['POZ', 'OZ ']

# (n_events, n_trials, n_chans, n_times)
mcee_sig = np.zeros((3, 100, len(tar_chans), 1640))

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
            eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
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
            
    data_path = r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_%d.mat'%(nt)
    io.savemat(data_path, {'mcee_sig': mcee_sig,
                           'chan_info': tar_chans})
    
#%% reload mcee data
# begin from 140ms, 5 chans, 1140-1340(200ms)
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\b_140ms\mcee_5chan_2.mat')
mcee_sig = eeg['mcee_sig']
tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2']
del eeg

# (n_events, n_trials, n_times)
pz = mcee_sig[:,:,0,1140:1540]   # 39
po5 = mcee_sig[:,:,1,1140:1540]  # 45
po3 = mcee_sig[:,:,2,1140:1540]  # 46
poz = mcee_sig[:,:,3,1140:1540]  # 47
po4 = mcee_sig[:,:,4,1140:1540]  # 48
po6 = mcee_sig[:,:,5,1140:1540]  # 49
o1 = mcee_sig[:,:,6,1140:1540]   # 52
oz = mcee_sig[:,:,7,1140:1540]   # 53
o2 = mcee_sig[:,:,8,1140:1540]   # 54
del mcee_sig

n_events = pz.shape[0]
n_trials = pz.shape[1]


#%% load origin data
eeg = io.loadmat(r'F:\SSVEP\dataset\preprocessed_data\weisiwen\f_data.mat')
ori_sig = eeg['f_data'][:,:,:,3140:6140] * 1e6
chan_info = eeg['chan_info'].tolist()
del eeg

# (n_events, n_trials, n_times)
pz = ori_sig[:,:,39,:]
po5 = ori_sig[:,:,45,:]
po3 = ori_sig[:,:,46,:]
poz = ori_sig[:,:,47,:]
po4 = ori_sig[:,:,48,:]
po6 = ori_sig[:,:,49,:]
o1 = ori_sig[:,:,52,:]
oz = ori_sig[:,:,53,:]
o2 = ori_sig[:,:,54,:]
del ori_sig

n_events = pz.shape[0]
n_trials = pz.shape[1]
#del chan_info

#%% correlation detection
acc_cv = []
mr = np.zeros((3,3))
# divide test dataset & training dataset 
te_d = copy.deepcopy(po5)
# target identification template (n_events, n_times)
template = np.mean(po5, axis=1)  

# pick a single trial of test dataset & compute Pearson correlation
# target: all events | input: all events, all trials
def corr_detect(test_data, template):
    '''
    Offline Target identification for single-channel data
        (using Pearson correlation coefficients)
    Parameters:
        test_data: array | (n_events, n_trials, n_times)
        template: array | (n_events, n_times)
    Returns:
        acc: int | the total number of correct identifications
        mr: square | (n_events (test dataset), n_events (template)),
            the mean of Pearson correlation coefficients
    '''
rou = np.zeros((n_events,100,n_events))
for nete in range(n_events):
    for ntte in range(100):
        for netr in range(n_events):
            rou[nete,ntte,netr] = np.sum(np.tril(np.corrcoef(te_d[nete,ntte,:],template[netr,:]),-1))
        del netr
        if np.max(np.where([rou[nete,ntte,:] == np.max(rou[nete,ntte,:])])) == nete:  # correct
            acc_cv.append(1)
    del ntte
del nete
    
for j in range(3):
    for k in range(3):
        mr[j,k] = np.mean(rou[j,:,k])

# reshape result
acc_cv = np.sum(acc_cv)
#%%
acc_cv/=300
acc_cv*=100

#%%
sns.set(style='whitegrid')
fig, ax = plt.subplots(3,1,figsize=(4,12))

ax[0].plot(poz[0,:,:].T, linewidth=0.5)
ax[0].plot(np.mean(poz[0,:,:], axis=0), linewidth=3, color='black')
ax[0].set_ylim(-20,20)

ax[1].plot(poz[1,:,:].T, linewidth=0.5)
ax[1].plot(np.mean(poz[1,:,:], axis=0), linewidth=3, color='black')
ax[1].set_ylim(-20,20)

ax[2].plot(poz[2,:,:].T, linewidth=0.5)
ax[2].plot(np.mean(poz[2,:,:], axis=0), linewidth=3, color='black')
ax[2].set_ylim(-20,20)

#%%
plt.plot(template[0,:], color='tab:orange')
plt.plot(poz[0,76,:], color='tab:blue')





