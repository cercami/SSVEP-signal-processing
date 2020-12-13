# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com
Created on Sat Nov 28 14:22:28 2020
Version: 0.1

A toolbox for target identification algorithms using Unified Framework
Refer: Wong C M, et al. Spatial Filtering in SSVEP-based BCIs: Unified Framework and New Improvements
    [J]. IEEE Transactions on Biomedical Engineering, 2020, PP(99):1-1.

Prefunctions:
1. zero_mean:
    zero-mean normalization (if necessary). 
        Generally speaking, the data preprocessed by bandpass filtering from MNE has been
        zero-averaged already. However, who knows how you guys preprocess the EEG data?
2. corr_coef:
    compute Pearson Correlation Coefficient
3. sinw:
    make a piece of sinusoidal wave
4. real_phase:
    compute best initial phase for training dataset
5. time_shift:
    cyclic rotate time sequence to destroy the time correlation of noise

The Unified Framework:
1. spatial_filter (W): class
    framework 1: (Z.T)*D*P*(P.T)*(D.T)*Z*W = Z.T*D*D.T*Z*W*Lamda
    framework 2: (Z.T)*D*P*(P.T)*(D.T)*Z*W = W*Lamda

Target identification functions
1. sCCA: (CCA_compute & sCCA)
    standard Canonical Correlation Analysis
2. itCCA:
    Individual template CCA
3. eCCA:
    extended CCA
4. TRCA: (TRCA_compute & TRCA)
    Task-Related Component Analysis
5. eTRCA:
    ensemble-TRCA
6. TRCA_R:
7. eTRCA_R
8. msCCA
9. stw_TRCA:
    sliding time window TRCA
10. corr_detect:
    correlation detect for single-channel data
    
update: 2020/12/12

"""

# %% basic modules
import numpy as np
from numpy import newaxis as NA
from numpy import linalg as LA
from math import pi
from sklearn import linear_model

from copy import deepcopy
from time import perf_counter
import scipy.io as io
import matplotlib.pyplot as plt

# %% Prefunctions
def zero_mean(data):
    """

    Parameters
    ----------
    data : (n_trials, n_channels, n_times)
        input data array.

    Returns
    -------
    data : (n_trials, n_channels, n_times)
        data after zero-mean normalization.
    """
    data -= data.mean(axis=-1, keepdims=True)

    return data

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
    X, y = zero_mean(X), zero_mean(y)
    cov_yx = y @ X.T
    var_xx, var_yy = np.sqrt(np.diagonal(X @ X.T)), np.sqrt(float(y @ y.T))
    corrcoef = cov_yx / (var_xx*var_yy)

    return corrcoef.mean()

def sinw(freq, time, phase, sfreq=1000):
    """

    Parameters
    ----------
    freq : float
        frequency / Hz.
    time : float
        time length / s.
    phase : float
        0-2.
    sfreq : float/int, optional
        sampling frequency. The default is 1000.

    Returns
    -------
    wave : (time*sfreq,)
        sinusoidal sequence.
    """
    n_point = int(time*sfreq)
    time_point = np.linspace(0, (n_point-1)/sfreq, n_point)
    wave = np.sin(2*pi*freq*time_point + pi*phase)

    return wave

def real_phase(data, freq, step=100, sfreq=1000):
    """

    Parameters
    ----------
    data : (n_trials, n_points)
    freq : float
        frequency of template.
    step : int, optional
        stepwidth = 1 / steps. The default is 100.
    sfreq : float, optional
        sampling frequency of signal. The default is 1000.

    Returns
    -------
    best_phase : float
        0-2.
    """
    time = data.shape[-1] / sfreq
    step = [2*x/step for x in range(step)]
    corr = np.zeros((step))

    signal_mean = data.mean(axis=0)[NA, :]

    for i in range(step):
        template = sinw(freq=freq, time=time, phase=step[i], sfreq=sfreq)
        corr[i] = corr_coef(signal_mean, template[NA,:])

    phase = np.max(np.where(corr == np.max(corr)))
    best_phase = phase*2/100

    return best_phase

def time_shift(data, step, axis=None):
    """

    Parameters
    ----------
    data : (n_channels, n_points)
        Input data array.
    step : int/float
        The length of the scroll. 
    axis : int or tuple of ints
        Dimension of scrolling, 0 - vertical, 1 - horizontal.
        By default(None) , the array will be flattened before being shifted,
            and then restored to its original shape.

    Returns
    -------
    tf_data : (n_channels, n_points)
        Data containing background EEG with corrupted temporal and spatial correlation.

    """
    n_chans = data.shape[0]
    tf_data = np.zeros_like(data)
    tf_data[0,:] = data[0,:]
    for i in range(n_chans-1):
        tf_data[i+1,:] = np.roll(data[i+1,:], shift=round(step*(i+1)), axis=axis)

    return tf_data


# %% The Unified Framework
class spatial_filter:
    """
    The Unified Framework for constructing spatial filters used in SSVEP signal processing

    """
    def __init__(self, )
    pass
