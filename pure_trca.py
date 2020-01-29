# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:10:16 2020

TRCA method without filter band

@author: Brynhildr
"""

# Import third part module
import numpy as np
import scipy.io as io
from scipy import signal

import copy

import mcee

import matplotlib.pyplot as plt
import seaborn as sns

#%% load origin data
acc_cv = np.zeros((7))
bp = 2140
for i in range(7):
    eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\raw & filtered data\50_hp.mat')
    if i == 0:  # 60ms
        data = eeg['f_data'][:,:,[45,51,52,53,54,55,58,59,60],bp:2200]
        del eeg
        print('\nData length: 60ms')
    elif i == 1 :  # 80ms
        data = eeg['f_data'][:,:,[45,51,52,53,54,55,58,59,60],bp:2220]
        del eeg
        print('\nData length: 80ms')
    elif i == 2:  # 100ms
        data = eeg['f_data'][:,:,[45,51,52,53,54,55,58,59,60],bp:2240]
        del eeg
        print('\nData length: 100ms')
    elif i == 3:  # 200ms
        data = eeg['f_data'][:,:,[45,51,52,53,54,55,58,59,60],bp:2340]
        del eeg
        print('\nData length: 200ms')
    elif i == 4:  # 300ms
        data = eeg['f_data'][:,:,[45,51,52,53,54,55,58,59,60],bp:2440]
        del eeg
        print('\nData length: 300ms')
    elif i == 5:  # 400ms
        data = eeg['f_data'][:,:,[45,51,52,53,54,55,58,59,60],bp:2540]
        del eeg
        print('\nData length: 400ms')
    elif i == 6:  # 500ms
        data = eeg['f_data'][:,:,[45,51,52,53,54,55,58,59,60],bp:2640]
        del eeg
        print('\nData length: 500ms')
    
    # basic information
    n_events = data.shape[0]
    n_trials = data.shape[1]
    
    # 10-fold cross-validation + TRCA
    print('Running TRCA alogrithm...')
    acc = []
    for cv in range(10):
        # divide dataset
        a = cv * 12
        # training dataset: (n_events, n_trials, n_chans, n_times)
        tr_data = data[:,a:a+int(n_trials/10),:,:]  # 12 trials
        # test dataset: (n_events, n_trials, n_chans, n_times)
        te_data = copy.deepcopy(data)
        te_data = np.delete(te_data,
            [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9,a+10,a+11], axis=1)  # 108 trials
        # main loop of TRCA
        acc_temp = mcee.pure_trca(train_data=te_data, test_data=tr_data)
        acc.append(np.sum(acc_temp))
        del acc_temp
        print(str(cv+1) + 'th cross-validation complete!')
    
    acc_cv[i] = np.sum(acc)/(10*12*2)*100
    del acc, tr_data, te_data


#%% load mcee data
bp = 1140
acc_cv = np.zeros((7,10))
for i in range(7):
    if i == 0:  # 60ms
        eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_hp\mcee_6.mat')
        data = eeg['mcee_sig'][:,:,:,bp:1200]
        del eeg
        print('\nData length: 60ms')
    elif i == 1 :  # 80ms
        eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_hp\mcee_5.mat')
        data = eeg['mcee_sig'][:,:,:,bp:1220]
        del eeg
        print('\nData length: 80ms')
    elif i == 2:  # 100ms
        eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_hp\mcee_4.mat')
        data = eeg['mcee_sig'][:,:,:,bp:1240]
        del eeg
        print('\nData length: 100ms')
    elif i == 3:  # 200ms
        eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_hp\mcee_3.mat')
        data = eeg['mcee_sig'][:,:,:,bp:1340]
        del eeg
        print('\nData length: 200ms')
    elif i == 4:  # 300ms
        eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_hp\mcee_2.mat')
        data = eeg['mcee_sig'][:,:,:,bp:1440]
        del eeg
        print('\nData length: 300ms')
    elif i == 5:  # 400ms
        eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_hp\mcee_1.mat')
        data = eeg['mcee_sig'][:,:,:,bp:1540]
        del eeg
        print('\nData length: 400ms')
    elif i == 6:  # 500ms
        eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_hp\mcee_0.mat')
        data = eeg['mcee_sig'][:,:,:,bp:1640]
        del eeg
        print('\nData length: 500ms')
    
    # basic information
    n_events = data.shape[0]
    n_trials = data.shape[1]
    
    # 10-fold cross-validation + TRCA
    print('Running TRCA alogrithm...')
    acc = []
    for cv in range(10):
        # divide dataset
        a = cv * 12
        # training dataset: (n_events, n_trials, n_chans, n_times)
        tr_data = data[:,a:a+int(n_trials/10),:,:]  # 12 trials
        # test dataset: (n_events, n_trials, n_chans, n_times)
        te_data = copy.deepcopy(data)
        te_data = np.delete(te_data,  
            [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9,a+10,a+11], axis=1)  # 108 trials
        # main loop of TRCA
        acc_temp = mcee.pure_trca(train_data=te_data, test_data=tr_data)
        acc.append(np.sum(acc_temp))
        del acc_temp
        print(str(cv+1) + 'th cross-validation complete!')
    
    acc_cv[i,:] = acc
    del acc, tr_data, te_data
    
#%% n_cycle pure TRCA
# initialization
bp = 1140
acc_cv = np.zeros((10,10))  # (n_cycles, n_fold)

# load data (200ms)
eeg = io.loadmat(r'I:\SSVEP\dataset\preprocessed_data\wuqiaoyi\mcee data\begin from 140ms\50_90_bp\mcee_3.mat')

for ncy in range(10):  # no more than 10 cycles
    data = eeg['mcee_sig'][:,:,:,bp:int(bp+(ncy+1)*1000/60)]
    print('\nData length: ' + str(ncy+1) + ' cycles')
    
    # basic information
    n_events = data.shape[0]
    n_trials = data.shape[1]
    
    # 10-fold cross-validation + TRCA
    print('Running TRCA alogrithm...')
    acc = []
    for cv in range(10):
        # divide dataset
        a = cv * 12
        # training dataset: (n_events, n_trials, n_chans, n_times)
        tr_data = data[:,a:a+int(n_trials/10),:,:]  # 12 trials
        # test dataset: (n_events, n_trials, n_chans, n_times)
        te_data = copy.deepcopy(data)
        te_data = np.delete(te_data,  
            [a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9,a+10,a+11], axis=1)  # 108 trials
        # main loop of TRCA
        acc_temp = mcee.pure_trca(train_data=te_data, test_data=tr_data)
        acc.append(np.sum(acc_temp))
        del acc_temp
        print(str(cv+1) + 'th cross-validation complete!')
    
    acc_cv[ncy,:] = acc
    del acc, tr_data, te_data