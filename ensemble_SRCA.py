# -*- coding: utf-8 -*-
"""
an ensemble SRCA demo

@ author: Brynhildr Wu
update: 30/12/2020

"""
# %% load in basic modules
import numpy as np
from numpy import newaxis as NA
from numpy import linalg as LA
from math import pi
from sklearn import linear_model

from copy import deepcopy
from time import perf_counter
import scipy.io as io
import matplotlib.pyplot as plt

# %% define functions
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

def sinw(freq, time, phase, sfreq=1000):
    '''
    make sine wave

    Parameters
    ----------
    freq : float
        frequency / Hz.
    time : float
        time lenght / s.
    phase : float
        0-2.
    sfreq : float/int, optional
        sampling frequency. The default is 1000.

    Returns
    -------
    wave : (time*sfreq,)
        sequence.
    '''
    n_point = int(time*sfreq)
    time_point = np.linspace(0, (n_point-1)/sfreq, n_point)
    wave = np.sin(2*pi*freq*time_point + pi*phase)
    return wave

def template_phase(data, freq, step=100, sfreq=1000):
    '''
    choose best template initial phase for training dataset

    Parameters
    ----------
    data : (n_trials, n_times)
    freq : float
        frequency of template.
    step : int, optional
        stepwidth = 1 / steps. The default is 100.
    sfreq : float, optional
        sampling frequency of signal. The default is 1000.

    Returns
    -------
    best_phase : float
        0-1.
    '''
    time = data.shape[-1] / sfreq
    phase = [2*x/step for x in range(step)]
    corr = np.zeros((step))
    sig_template = np.mean(data, axis=0)
    for i in range(step):
        tar_template = sinw(freq=freq, time=time, phase=phase[i], sfreq=sfreq)
        corr[i] = corr_coef(sig_template, tar_template)
    best_phase = np.argmax(corr)*2/100

    return best_phase

def srca_process(rest_model, rest_target, task_model, task_target, regression='Iva',
                alpha=1.0, l1_ratio=1.0):
    """

    Parameters
    ----------
    rest_model : (n_trials, n_channels, n_points)
        Rest-state data of model channels.
    rest_target : (n_trials, n_points)
        Rest-state data of target channel.
    task_model : (n_trials, n_channels, n_points)
        Task-state data of model channels.
    task_target : (n_trials, n_points)
        Task-state data of target channel.
    regression : str, optional
        Regression methods: 'Iva'(inverse array), 'OLS', 'Ridge', 'Lasso' and 'ElasticNet'.
        Inverse-array method does not consider the baseline drift obtained in the
            multi-linear regression calculation, but matrix operation is the fastest way to
            implement regression and the sacrifice in accuracy is negligible
        The default is 'Iva'.
    alpha : float, optional
        Parameters used in Ridge, Lasso and EN regression. The default is 1.0.
    l1_ratio : float, optional
        Parameters used in EN regression. The default is 1.0.

    Returns
    -------
    extract : (n_trials, n_points)
        SRCA filtered data.

    """
    n_trials = task_target.shape[0]
    n_points = task_target.shape[-1]
    estimate = np.zeros((n_trials, n_points))
    if regression == 'Iva':
        for i in range(n_trials):
            # Y = AX, A = Y*X^T*(X*X.T)^-1
            X, Y = rest_model[i,...], rest_target[i,:][NA,:]  # X: (n_chans, n_points) | Y: (1, n_points)
            matrix_A = Y @ X.T @ LA.inv(X@X.T)
            estimate[i,:] = matrix_A @ task_model[i,...]
    else:  # consider the baseline drift, but much slower
        for i in range(n_trials):  # basic operating unit: (n_points, n_chans) & (n_points,)
            if regression == 'OLS':
                L = linear_model.LinearRegression().fit(rest_model[i,...].T, rest_target[i,:].T)
            elif regression == 'Ridge':
                L = linear_model.Ridge(alpha).fit(rest_model[i,...].T, rest_target[i,:].T)
            elif regression == 'Lasso':
                L = linear_model.Lasso(alpha).fit(rest_model[i,...].T, rest_target[i,:].T)
            elif regression == 'ElasticNet':
                L = linear_model.ElasticNet(alpha, l1_ratio).fit(rest_model[i,...].T, rest_target[i,:].T)
            baseline = L.intercept_
            fitting_coef = L.coef_
            estimate[i,:] = fitting_coef@task_model[i,...] + baseline
    extract = task_target - estimate

    return extract

def fisher_score(data):
    """

    Parameters
    ----------
    data : (2, n_trials, n_points)
        input data array.

    Returns
    -------
    fs : float
        Fisher-Score(mean) of input data.

    """

    # data initialization
    sample_num = data.shape[1]
    data1, data2 = data[0,...], data[1,...]
    group1_mean, group2_mean = data1.mean(axis=0, keepdims=True), data2.mean(axis=0, keepdims=True)
    total_mean = data.mean(axis=0, keepdims=True)

    # inter-class divergence
    ite_d = sample_num * ((group1_mean-total_mean)**2 + (group2_mean-total_mean)**2)

    # intra-class divergence
    itr_d = sample_num * (np.sum((data1-group1_mean)**2, axis=0) + np.sum((data2-group2_mean)**2, axis=0))

    # fisher-score
    fs = ite_d / itr_d

    return fs.mean()

def ensemble_SNR(data):
    """

    Parameters
    ----------
    data : (n_events, n_trials, n_points)

    Returns
    -------
    snr : (n_points,)

    """
    n_events, n_points = data.shape[0], data.shape[-1]
    snr = np.zeros((n_events))
    for i in range(n_events):
        signal = data[i,...].mean(axis=0)                     # (n_points,)
        signal_power = signal**2                              # (n_points,)
        noise_power = ((data[i,...]-signal)**2).mean(axis=0)  # (n_points,)
        snr[i] = np.mean(signal_power/noise_power)
    snr = np.sum(snr)

    return snr

def ensemble_pCORR(data, freq, phase, sfreq=1000):
    """

    Parameters
    ----------
    data : (n_trials, n_times)
    freq : list
        frequency of template.
    phase : list
        0-1, initial phase of template.
    sfreq : float, optional
        sampling frequency of signal. The default is 1000.

    Returns
    -------
    corr : float
        correlation sequence.
    """
    n_events = data.shape[0]
    time_length = data.shape[-1] / sfreq
    template = np.zeros((n_events, time_length))
    corr = np.zeros((n_events))
    for i in range(n_events):
        template[i,:] = sinw(freq[i], time_length, phase[i], sfreq)
        corr[i] = corr_coef(data[i,...], template[i,:])
    corr = np.sum(corr)
    
    return corr


# %% load in training data
data_path = r'D:\SSVEP\dataset\preprocessed_data\cvep_8\wuqiaoyi\fir_50_70.mat'
eeg = io.loadmat(data_path)
train_data = eeg['f_data']
chans = eeg['chan_info'].tolist()
del data_path, eeg

tar_chans = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

# %% SRCA training: initialization
start = perf_counter()

max_loop = len(chans)
remain_chans = []
para_change = []
active = True

# %% first loop: just add best one
compare_para = np.zeros((len(chans)))
