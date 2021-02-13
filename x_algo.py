# -*- coding: utf-8 -*-
"""
Top Secret
A new data calibration algorithm

P = argmin(||P*Xi - Xmean||_F^2)
P.T = (X @ X) @ (1/N * sum(Xi @ Xi.T))^-1

Refer: Xiaoyu Zhou, Mingpeng Xu
Script author: Brynhildr W

"""

# %%
import numpy as np
from numpy import (sin,pi)
from numpy import linalg as LA

import scipy.io as io

import matplotlib.pyplot as plt
import mcee
%matplotlib auto


# %% pre function
def corr_coef(X, y):
    """

    Parameters
    ----------
    X : (..., n_points)
        input data array. (could be sequence)
    y : (1, n_points)
        input data vector/sequence

    Returns
    -------
    corrcoef : float
        Pearson's Correlation Coefficient(mean of sequence or single number).
    """
    cov_yx = y @ X.T
    # Note: 'int' object has no attribute 'ndim', but the dimension of 'float' object is 0
    if cov_yx.ndim == 0:  
        var_xx = np.sqrt(X @ X.T)
    # try:
    #     dim = cov_yx.ndim  
    # except AttributeError:
    #     var_xx = np.sqrt(X @ X.T)
    else:
        var_xx = np.sqrt(np.diagonal(X @ X.T)) 
    var_yy = np.sqrt(float(y @ y.T))
    corrcoef = cov_yx / (var_xx*var_yy)

    return corrcoef.mean()

def TRCA_compute(data):
    '''
    Task-related component analysis (TRCA)

    Parameters
    ----------
    data : (n_events, n_trials, n_chans, n_times)
        input data array (default z-scored after bandpass filtering).

    Returns
    -------
    w : (n_events, n_chans)
        eigenvector refering to the largest eigenvalue.

    '''
    # basic information
    n_events = data.shape[0]
    n_trials= data.shape[1]
    n_chans = data.shape[2]
    n_times = data.shape[-1]

    # compute spatial filter W 
    w = np.zeros((n_events, n_chans))
    for ne in range(n_events):
        # matrix Q: inter-channel covariance
        q = np.zeros((n_chans, n_chans))
        temp_Q = data[ne, ...].swapaxes(0,1).reshape((n_chans,-1), order='C')
        q = (temp_Q@temp_Q.T) / (n_trials*n_times)
        # matrix S: inter-channels' inter-trial covariance
        s = np.zeros_like(q)
        for nt_i in range(n_trials):
            for nt_j in range(n_trials):
                if nt_i != nt_j:
                    data_i = data[ne, nt_i, ...]
                    data_j = data[ne, nt_j, ...]
                    s += (data_i@data_j.T) / n_times
        # generalized eigenvalue problem
        e_va, e_vec = LA.eig(LA.inv(q)@s)
        w_index = np.argmax(e_va)
        w[ne, :] = e_vec[:, w_index].T
    return w

# For origin data
def TRCA(train_data, test_data):
    '''
    TRCA with target identification process in offline situation

    Parameters
    ----------
    train_data : (n_events, n_trials, n_chans, n_times)
        training dataset.
    test_data : (n_events, n_trials, n_chans, n_times)
        test dataset.

    Returns
    -------
    accuracy : float, 0-1

    '''
    # basic parameters
    n_events, n_tests = test_data.shape[0], test_data.shape[1]
    template = train_data.mean(axis=1)  # template data: (n_events, n_chans, n_times)
    w = TRCA_compute(train_data)        # spatial filter W: (n_events, n_chans)

    # target identification
    r = np.zeros((n_events, n_events, n_tests))
    for nete in range(n_events):  # n_events in test dataset
        for netr in range(n_events):
            for nte in range(n_tests):  # n_events in test dataset
                temp_test = w[netr, :] @ test_data[nete, nte, ...]
                temp_template = w[netr, :] @ template[netr, ...]
                r[nete, netr, nte] = corr_coef(temp_test, temp_template)
    
    # compute accuracy
    accuracy = []
    for nete in range(n_events):  # n_events in training dataset
        for nte in range(n_tests):
            if np.argmax(r[nete,:,nte]) == nete:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return accuracy

# main function
def model_calibration(data):
    """

    Parameters
    ----------
    data : ndarray, (n_trials, n_chans, n_times)

    Returns
    -------
    projection : ndarray, (n_chans, n_chans)
    """
    n_trials = data.shape[0]
    n_chans = data.shape[1]

    template = data.mean(axis=0)  # (n_chans, n_times)
    temp = np.zeros((n_chans, n_chans))
    for ntr in range(n_trials):
        temp += data[ntr,...] @ data[ntr,...].T

    projection = (template@template.T) @ LA.inv(temp/n_trials)

    return projection

def cali_TRCA(train_data, test_data):
    """
    TRCA with model calibration
    Parameters
    ----------
    train_data : (n_events, n_trains, n_chans, n_times)
        training dataset.
    test_data : (n_events, n_tests, n_chans, n_times)
        test dataset.

    Returns
    -------
    accuracy : float, 0-1
    """
    # initialization
    n_events = train_data.shape[0]
    n_trains = train_data.shape[1]
    n_chans = train_data.shape[-2]
    n_times = train_data.shape[-1]

    # data calibration & apply calibration on training dataset
    projection = np.zeros((n_events, n_chans, n_chans))
    model_sig = np.zeros_like(train_data)
    for ne in range(n_events):
        projection[ne,...] = model_calibration(train_data[ne,...])
        for ntr in range(n_trains):
            model_sig[ne,ntr,...] = projection[ne,...] @ train_data[ne,ntr,...]
    del ne, ntr

    # apply different calibration on test dataset
    n_tests = test_data.shape[1]
    target_sig = np.zeros((n_events, n_events, n_tests, n_chans, n_times))
    for nete in range(n_events):  # test dataset
        for netr in range(n_events):  # train dataset
            for nte in range(n_tests):
                target_sig[nete,netr,nte,...] = projection[netr,...] @ test_data[nete,nte,...]
    del nete, netr, nte

    template = model_sig.mean(axis=1)  # (n_events, n_chans, n_times)
    w = TRCA_compute(model_sig)

    # combine target identification
    r = np.zeros((n_events, n_events, n_tests))
    for nete in range(n_events):  # test dataset
        for netr in range(n_events):  # train dataset
            for nte in range(n_tests):
                temp_test = w[netr,:] @ target_sig[nete,netr,nte,...]
                temp_template = w[netr,:] @ template[netr,...]
                r[nete,netr,nte] = corr_coef(temp_test,temp_template)
    del nete, netr, nte

    # compute accuracy
    accuracy = []
    for nete in range(n_events):  # test dataset
        for nte in range(n_tests):
            if np.argmax(r[nete,:,nte]) == nete:
                accuracy.append(1)
    accuracy = np.sum(accuracy) / (n_events*n_tests)

    return projection, model_sig, target_sig, r, accuracy


# %% test with TRCA  
# eeg = io.loadmat(r'D:\SSVEP\dataset\preprocessed_data\xwt_bishe\wuqiaoyi\1-31\f_60.mat')
eeg = io.loadmat(r'D:\SSVEP\dataset\preprocessed_data\xwt_bishe\zhangcongyu\f_60.mat')
f_data = eeg['f_data']  # (n_events, n_trials, n_chans, n_times)
tar_list = [45,51,52,53,54,55,58,59,60]
# tar_list = [47,53,54,55,56,57,60,61,62]
ns = 40

acc_ori_trca, acc_ori_strca = np.zeros((20, 5)), np.zeros((20, 5))
for cv in range(20):
    print('CV : %d turn...' %(cv+1))
    randPick = np.arange(f_data.shape[1])
    np.random.shuffle(randPick)
    for nt in range(5):
        print('Data length: %d00ms' %(nt+1))
        train = f_data[:,randPick[:ns],:,1140:1190+nt*50][...,tar_list,:]
        test = f_data[:,randPick[ns:],:,1140:1190+nt*50][...,tar_list,:]

        # target identification main process
        otrca = TRCA(train, test)
        _,_,_,_,strca = cali_TRCA(train, test)
        # dcpm = mcee.DCPM(train, test, di=['1','2'])


        # save accuracy data
        acc_ori_trca[cv, nt] = otrca
        acc_ori_strca[cv, nt] = strca
        # acc_dcpm[cv, nt] = dcpm


    print(str(cv+1) + 'th cross-validation complete!\n')

# %%
