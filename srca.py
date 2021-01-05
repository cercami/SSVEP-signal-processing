# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu, Tuo Liu
@ email: brynhildrwu@gmail.com
Created on Sat Nov 28 14:22:28 2020
Version: 0.1

A toolbox for algorithm 'Spatial Regression Component Analysis (SRCA)'

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
5. sCCA_coef: (Doubtful in principle)
    compute standard canonical correlation coefficient
6. time_shift:
    cyclic rotate time sequence to destroy the time correlation of noise

Target functions:
1. snr_time:
    compute the mean of SSVEP's SNR sequence in time domain
2. pearson_corr:
    compute the mean of Pearson correlation coefficient (each trial & average)
3. fisher_score:
    compute the mean of Fisher-Score sequence
4. phase_corr: (i.e. corr_coef, set X to data, y to template)
    compute correlation coefficient between real signal and phase-modulated artificial template
5. phase_cca: (Doubtful in principle)
    compute CCA coefficient between real signal and phase-modulated artificial template

Target identification functions (Developer-Special Edition, DSE)
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
    

Main function:
1. SRCA: class
    container of SRCA data

update: 2020/12/10

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

def sCCA_coef():
    """

    Parameters
    ----------
    data : (n_trials, n_points)
        input data array.
    group_num : float, optional
        number of groups. The default is 2.
    break_point : int
        The data before the breakpoint belongs to one group
            while the data after it belongs to another.

    Returns
    -------
    fs : (n_points,)
        Fisher-Score sequence of input data.
    """

    pass

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


# %% Target functions
def snr_time(data, mean=True):
    """

    Parameters
    ----------
    data : (n_trials, n_points)

    Returns
    -------
    snr : (n_points,)
        SNR sequence in time domain.
    """
    n_points = data.shape[-1]
    snr = np.zeros((n_points))
    signal = data.mean(axis=0)                       # (n_points,)

    signal_power = signal**2                         # (n_points,)
    noise_power = ((data - signal)**2).mean(axis=0)  # (n_points,)

    snr = signal_power / noise_power                 # (n_points,)

    if mean:
        return snr.mean()
    elif not mean:
        return snr

def pearson_corr(data):
    """

    Parameters
    ----------
    data : (n_trials, n_points)
        input data array.
    
    Returns
    -------
    corr : float
        correlation coefficient.
    """
    template = data.mean(axis=0)  # (n_points,)
    corr = corr_coef(data, template[NA,:])

    return corr
 
def fisher_score(data, break_point, group_num=2):
    """

    Parameters
    ----------
    data : (n_trials, n_points)
        input data array.
    group_num : float, optional
        number of groups. The default is 2.
    break_point : int
        The data before the breakpoint belongs to one group
            while the data after it belongs to another.

    Returns
    -------
    fs : (n_points,)
        Fisher-Score sequence of input data.

    """
    # only support 2 categories
    if group_num != 2:
        raise ValueError('Only support 2 categories in standard Fisher-Score computation')

    # data initialization
    sample_num1 = break_point + 1
    sample_num2 = data.shape[0] - sample_num1

    data1, data2 = data[:break_point,:], data[break_point:,:]

    group1_mean, group2_mean = data1.mean(axis=0, keepdims=True), data2.mean(axis=0, keepdims=True)
    total_mean = data.mean(axis=0, keepdims=True)

    # inter-class divergence
    ite_d = sample_num1 * (group1_mean-total_mean)**2
    ite_d += sample_num2 * (group2_mean-total_mean)**2

    # intra-class divergence
    itr_d = sample_num1 * np.sum((data1-group1_mean)**2, axis=0)
    itr_d += sample_num2 * np.sum((data2-group2_mean)**2, axis=0)

    # fisher-score
    fs = ite_d / itr_d

    return fs.squeeze()

def phase_corr(data, template):
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
        Pearson's Correlation Coefficient(mean of a sequence).
    """
    data, template = zero_mean(data), zero_mean(template)
    cov_td = template @ data.T
    var_dd, var_tt = np.sqrt(np.diagonal(data @ data.T)), np.sqrt(float(template @ template))
    corrcoef = cov_td / (var_dd*var_tt)

    return corr_coef.mean()

def phase_cca():
    pass


# %% return sorted list's label change
# index_change = sorted(range(len(list)), key=lambda k: lis[k])
# data = data[index_change[:], ...]

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
                L = linear_model.Ridge(alpha=alpha).fit(rest_model[i,...].T, rest_target[i,:].T)
            elif regression == 'Lasso':
                L = linear_model.Lasso(alpha=alpha).fit(rest_model[i,...].T, rest_target[i,:].T)
            elif regression == 'ElasticNet':
                L = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(rest_model[i,...].T, rest_target[i,:].T)
            baseline = L.intercept_  # float
            fitting_coef = L.coef_   # vector: ()
            #estimate[i,:] = np.dot(fitting_coef, task_model[i,...]) + baseline
            estimate[i,:] = fitting_coef@task_model[i,...] + baseline
    extract = task_target - estimate

    return extract


# main class
class SRCA:
    """
    Spatial Regression Component Analysis.
    """
    def __init__(self, sample_rate, target_channel, background_end_time, task_end_time = None,
                 background_begin_time = 0, task_begin_time = None, recursive_form = 'stepwise',
                 raw_fitting_channels = None, objective_function = 'fisher_score', n_jobs = 1):
        """

        Parameters
        ----------
        sample_rate : int
            The sample rate.
        target_channel : string list
            A list. The elements are string target channels.
        background_end_time : float
            Background EEG end time, unit is second.
        task_end_time : float, the default is None.
            Task EEG end time, unit is second. 
            When it is None, it is the end time of the whole trial.
        background_begin_time : float, the default is 0.
            Background EEG begin time, unit is second.
        task_begin_time : float, the default is None.
            Task EEG begin time, unit is second.
            When it is None, set it to background EEG end time.
        recursive_form : string, the options are {'traversal','stepwise'},
                         the default is 'stepwise'.
            The recursive form for choose channels from raw fitting channals.
        raw_fitting_channels : string list list, the default is None.
            The raw fitting channels, sometimes it is not required.
            But when it was given, the number of string lists must be 
            equal to number of target channel strings.
        objective_function : string, the options are {'fisher_score','pearson_corr'}, 
                             the default is 'fisher_score'.
            The objective function.
        n_jobs : int, the default is 1.
            Parallel computing, the value is the number of processes.

        Raises
        ------
        ValueError
            Len of raw_fitting_channels must be equal to len of target_channel.

        Returns
        -------
        None.

        """
        self.sample_rate = sample_rate
        self.target_channel = target_channel
        self.background_end_point = int(np.round(self.sample_rate * background_end_time))
        if task_end_time == None:
            self.task_end_point = None
        else:
            self.task_end_point = int(np.round(self.sample_rate * task_end_time))
        self.background_begin_point = int(np.round(self.sample_rate * background_begin_time))
        if task_begin_time == None:
            self.task_begin_point = int(np.round(self.sample_rate * background_end_time))
        else:
            self.task_begin_point = int(np.round(self.sample_rate * task_begin_time))
        self.raw_fitting_channels = raw_fitting_channels
        if self.raw_fitting_channels != None and len(self.raw_fitting_channels) != len(self.target_channel):
            raise ValueError("len of raw_fitting_channels must be equal to len of target_channel.")
        self.recursive_form = recursive_form
        self.objective_function = objective_function
        self.n_jobs = n_jobs # Multiprocess parameter
        
    def fit(self,X,y):
        """

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_points)
            Raw data.
        y : ndarray, shape (n_trials)
            Labels.

        Returns
        -------
        None.

        """
        X = deepcopy(X)
        y = deepcopy(y)
        n_trials, n_channels, n_points = X.shape
        self.best_fitting_channels = self.raw_fitting_channels # The best fitting channels combination.
        pass

    def transform(self,X):
        """

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_points)
            Raw data.

        Returns
        -------
        Residual_X : ndarray, shape (n_trials, n_best_fitting_channels, n_points)
            Data after SRCA.

        """
        Residual_X = deepcopy(X)
        return Residual_X


